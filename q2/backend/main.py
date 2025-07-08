from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import uvicorn
import logging

from ingest import DocumentIngester
from retrieval import RetrievalEngine
from gemini_llm import GeminiLLM
from chroma_utils import ChromaManager

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Indian Legal Document Search API",
    description="Semantic and hybrid search system for Indian legal documents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
chroma_manager = ChromaManager()
document_ingester = DocumentIngester(chroma_manager)
retrieval_engine = RetrievalEngine(chroma_manager)
gemini_llm = GeminiLLM()

@app.on_event("startup")
async def startup_event():
    """Initialize ChromaDB collections on startup"""
    try:
        await chroma_manager.initialize_collections()
        logger.info("ChromaDB collections initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB collections: {e}")
        # Continue anyway - collections will be created on first use

@app.get("/")
async def root():
    return {"message": "Indian Legal Document Search API", "version": "1.0.0"}

@app.post("/upload-document/")
async def upload_document(
    file: UploadFile = File(...),
    law_name: str = Form(..., description="Name of the law (e.g., Income Tax Act)"),
    document_type: str = Form(default="act", description="Type of document (act, judgment, etc.)")
):
    """Upload and ingest a legal document"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.docx', '.txt')):
            raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are supported")
        
        # Read file content
        content = await file.read()
        
        # Ingest document
        result = await document_ingester.ingest_document(
            content=content,
            filename=file.filename,
            law_name=law_name,
            document_type=document_type
        )
        
        return {
            "message": "Document uploaded and ingested successfully",
            "filename": file.filename,
            "chunks_created": result["chunks_created"],
            "law_name": law_name,
            "document_type": document_type
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/search/")
async def search_documents(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(default=5, description="Number of results to return"),
    similarity_methods: List[str] = Query(
        default=["cosine", "euclidean", "mmr", "hybrid"],
        description="Similarity methods to use"
    )
):
    """Search legal documents using multiple similarity methods"""
    try:
        results = await retrieval_engine.search_all_methods(
            query=query,
            top_k=top_k,
            methods=similarity_methods
        )
        
        return {
            "query": query,
            "top_k": top_k,
            "results": results,
            "total_methods": len(similarity_methods)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")

@app.post("/summarize/")
async def summarize_results(
    request: Dict[str, Any]
):
    """Summarize search results using Gemini LLM"""
    try:
        query = request.get("query", "")
        documents = request.get("documents", [])
        
        summary = await gemini_llm.summarize_legal_documents(
            query=query,
            documents=documents
        )
        
        return {
            "query": query,
            "summary": summary,
            "document_count": len(documents)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during summarization: {str(e)}")

@app.get("/collections/info/")
async def get_collections_info():
    """Get information about ChromaDB collections"""
    try:
        info = await chroma_manager.get_collections_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collections info: {str(e)}")

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "chroma_connected": await chroma_manager.test_connection(),
        "gemini_available": gemini_llm.test_connection()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("BACKEND_PORT", 8000)),
        reload=True
    ) 