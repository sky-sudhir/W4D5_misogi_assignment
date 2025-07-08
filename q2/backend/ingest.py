import fitz  # PyMuPDF
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
import uuid
import logging
from io import BytesIO
import re

from chroma_utils import ChromaManager

logger = logging.getLogger(__name__)

class DocumentIngester:
    """Handles document ingestion and chunking"""
    
    def __init__(self, chroma_manager: ChromaManager):
        self.chroma_manager = chroma_manager
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            # Open PDF from bytes
            doc = fitz.open(stream=content, filetype="pdf")
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            
            doc.close()
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def extract_text_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX content"""
        try:
            # Open DOCX from bytes
            doc = Document(BytesIO(content))
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {str(e)}")
            raise
    
    def extract_text_from_txt(self, content: bytes) -> str:
        """Extract text from TXT content"""
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return content.decode('latin-1')
            except Exception as e:
                logger.error(f"Error extracting text from TXT: {str(e)}")
                raise
    
    def extract_text(self, content: bytes, filename: str) -> str:
        """Extract text based on file extension"""
        file_extension = filename.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            return self.extract_text_from_pdf(content)
        elif file_extension == 'docx':
            return self.extract_text_from_docx(content)
        elif file_extension == 'txt':
            return self.extract_text_from_txt(content)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep legal formatting
        text = re.sub(r'[^\w\s\.\,\;\:\(\)\[\]\-\'\"]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_legal_metadata(self, text: str, law_name: str) -> Dict[str, Any]:
        """Extract legal-specific metadata from text"""
        metadata = {
            "law_name": law_name,
            "sections": [],
            "subsections": [],
            "clauses": [],
            "keywords": []
        }
        
        # Extract sections (e.g., "Section 80C", "Sec. 12A")
        section_pattern = r'(?:Section|Sec\.?)\s+(\d+[A-Z]*)'
        sections = re.findall(section_pattern, text, re.IGNORECASE)
        metadata["sections"] = list(set(sections))
        
        # Extract subsections (e.g., "(1)", "(a)", "(i)")
        subsection_pattern = r'\(([0-9]+[a-z]*|[a-z]+|[ivx]+)\)'
        subsections = re.findall(subsection_pattern, text, re.IGNORECASE)
        metadata["subsections"] = list(set(subsections))
        
        # Extract common legal keywords
        legal_keywords = [
            "act", "rule", "regulation", "provision", "clause", "schedule",
            "amendment", "notification", "circular", "judgment", "order",
            "tax", "income", "gst", "property", "contract", "liability"
        ]
        
        found_keywords = []
        for keyword in legal_keywords:
            if keyword.lower() in text.lower():
                found_keywords.append(keyword)
        
        metadata["keywords"] = found_keywords
        
        return metadata
    
    async def ingest_document(
        self,
        content: bytes,
        filename: str,
        law_name: str,
        document_type: str = "act"
    ) -> Dict[str, Any]:
        """Ingest a document into all collections"""
        try:
            # Extract text
            raw_text = self.extract_text(content, filename)
            
            # Preprocess text
            cleaned_text = self.preprocess_text(raw_text)
            
            # Create chunks
            chunks = self.text_splitter.split_text(cleaned_text)
            
            # Extract legal metadata
            base_metadata = self.extract_legal_metadata(cleaned_text, law_name)
            
            # Prepare documents for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                doc_id = f"{filename}_{uuid.uuid4().hex[:8]}_{i}"
                
                # Create metadata for this chunk (keep under 256 bytes)
                chunk_metadata = {
                    "law": base_metadata["law_name"][:50],  # Truncate to 50 chars
                    "sections": ",".join(base_metadata["sections"][:3]),  # Max 3 sections
                    "keywords": ",".join(base_metadata["keywords"][:5]),  # Max 5 keywords
                    "type": document_type,
                    "chunk_idx": i,
                    "length": len(chunk)
                }
                
                documents.append(chunk)
                metadatas.append(chunk_metadata)
                ids.append(doc_id)
            
            # Add to all collections
            for collection_type in ["cosine", "euclidean", "mmr", "hybrid"]:
                await self.chroma_manager.add_documents(
                    collection_type=collection_type,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            logger.info(f"Successfully ingested {filename} with {len(chunks)} chunks")
            
            return {
                "chunks_created": len(chunks),
                "total_characters": len(cleaned_text),
                "metadata": base_metadata,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document {filename}: {str(e)}")
            raise
    
    def get_legal_terms(self) -> List[str]:
        """Get list of legal terms for entity matching"""
        return [
            # Income Tax Act terms
            "section 80c", "section 80d", "section 24", "section 54",
            "capital gains", "house property", "salary income", "business income",
            "deduction", "exemption", "assessment year", "financial year",
            
            # GST Act terms
            "input tax credit", "output tax", "reverse charge", "composition scheme",
            "gst registration", "place of supply", "taxable supply", "exempt supply",
            "igst", "cgst", "sgst", "utgst",
            
            # General legal terms
            "plaintiff", "defendant", "appellant", "respondent", "writ petition",
            "high court", "supreme court", "civil procedure code", "criminal procedure code",
            "constitution", "fundamental rights", "directive principles",
            
            # Property law terms
            "sale deed", "lease deed", "mortgage", "easement", "title deed",
            "registration", "stamp duty", "property tax", "land revenue"
        ] 