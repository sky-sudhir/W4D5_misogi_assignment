import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

logger = logging.getLogger(__name__)

class GeminiLLM:
    """Handles Gemini 2.0 Flash integration for legal document processing"""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.api_key,
            temperature=0.1,
            max_tokens=1000
        )
        
        # Define prompts
        self.summarization_prompt = PromptTemplate(
            input_variables=["query", "documents"],
            template="""
You are a legal expert specializing in Indian law. Based on the search query and retrieved legal documents, provide a comprehensive summary.

Search Query: {query}

Retrieved Legal Documents:
{documents}

Please provide:
1. A concise summary addressing the query
2. Key legal provisions or sections mentioned
3. Relevant case law or precedents (if any)
4. Practical implications or applications

Focus on accuracy and cite specific sections or provisions where applicable.

Summary:
"""
        )
        
        self.qa_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
You are a legal expert specializing in Indian law. Answer the following question based on the provided legal context.

Question: {question}

Legal Context:
{context}

Provide a detailed answer that:
1. Directly addresses the question
2. Cites relevant legal provisions
3. Explains the legal reasoning
4. Mentions any important exceptions or conditions

Answer:
"""
        )
    
    def test_connection(self) -> bool:
        """Test Gemini API connection"""
        try:
            # Simple test query
            response = self.llm.invoke([
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Hello, please respond with 'Connection successful'")
            ])
            return "Connection successful" in response.content
        except Exception as e:
            logger.error(f"Gemini connection test failed: {str(e)}")
            return False
    
    def _format_documents_for_prompt(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents for inclusion in prompts"""
        formatted_docs = []
        
        for i, doc in enumerate(documents, 1):
            doc_text = doc.get('document', '')
            metadata = doc.get('metadata', {})
            
            law_name = metadata.get('law_name', 'Unknown Law')
            sections = metadata.get('sections', [])
            similarity_score = doc.get('similarity_score', 0.0)
            
            doc_info = f"""
Document {i}:
Law: {law_name}
Sections: {', '.join(sections) if sections else 'Not specified'}
Similarity Score: {similarity_score:.3f}
Content: {doc_text}
---
"""
            formatted_docs.append(doc_info)
        
        return "\n".join(formatted_docs)
    
    async def summarize_legal_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Summarize legal documents based on search query"""
        try:
            if not documents:
                return {
                    "summary": "No documents found for the given query.",
                    "key_provisions": [],
                    "recommendations": []
                }
            
            # Format documents for prompt
            formatted_docs = self._format_documents_for_prompt(documents)
            
            # Create summarization chain
            chain = LLMChain(llm=self.llm, prompt=self.summarization_prompt)
            
            # Generate summary
            response = await chain.arun(
                query=query,
                documents=formatted_docs
            )
            
            # Extract key information from metadata
            all_sections = []
            all_laws = []
            
            for doc in documents:
                metadata = doc.get('metadata', {})
                if metadata.get('sections'):
                    all_sections.extend(metadata['sections'])
                if metadata.get('law_name'):
                    all_laws.append(metadata['law_name'])
            
            # Remove duplicates
            unique_sections = list(set(all_sections))
            unique_laws = list(set(all_laws))
            
            return {
                "summary": response.strip(),
                "key_provisions": unique_sections,
                "laws_covered": unique_laws,
                "document_count": len(documents),
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Error in summarize_legal_documents: {str(e)}")
            raise
    
    async def answer_legal_question(
        self,
        question: str,
        context_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Answer a legal question based on context documents"""
        try:
            if not context_documents:
                return {
                    "answer": "I don't have sufficient legal context to answer this question.",
                    "confidence": 0.0,
                    "sources": []
                }
            
            # Format context
            context = self._format_documents_for_prompt(context_documents)
            
            # Create QA chain
            chain = LLMChain(llm=self.llm, prompt=self.qa_prompt)

            print(context)
            
            # Generate answer
            response = await chain.arun(
                question=question,
                context=context
            )
            
            # Extract sources
            sources = []
            for doc in context_documents:
                metadata = doc.get('metadata', {})
                source_info = {
                    "law_name": metadata.get('law_name', 'Unknown'),
                    "sections": metadata.get('sections', []),
                    "similarity_score": doc.get('similarity_score', 0.0)
                }
                sources.append(source_info)
            
            return {
                "answer": response.strip(),
                "sources": sources,
                "context_document_count": len(context_documents),
                "question": question
            }
            
        except Exception as e:
            logger.error(f"Error in answer_legal_question: {str(e)}")
            raise
    
    async def generate_legal_insights(
        self,
        documents: List[Dict[str, Any]],
        focus_area: str = "general"
    ) -> Dict[str, Any]:
        """Generate legal insights from documents"""
        try:
            if not documents:
                return {"insights": [], "recommendations": []}
            
            insight_prompt = PromptTemplate(
                input_variables=["documents", "focus_area"],
                template="""
Analyze the following legal documents and provide insights focused on {focus_area}.

Legal Documents:
{documents}

Provide:
1. Key legal insights and patterns
2. Important precedents or principles
3. Practical recommendations
4. Potential legal risks or considerations

Focus Area: {focus_area}

Insights:
"""
            )
            
            formatted_docs = self._format_documents_for_prompt(documents)
            
            chain = LLMChain(llm=self.llm, prompt=insight_prompt)
            
            response = await chain.arun(
                documents=formatted_docs,
                focus_area=focus_area
            )
            
            return {
                "insights": response.strip(),
                "focus_area": focus_area,
                "document_count": len(documents)
            }
            
        except Exception as e:
            logger.error(f"Error in generate_legal_insights: {str(e)}")
            raise
    
    async def explain_legal_concept(
        self,
        concept: str,
        context_documents: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Explain a legal concept with optional context"""
        try:
            explanation_prompt = PromptTemplate(
                input_variables=["concept", "context"],
                template="""
Explain the legal concept "{concept}" in the context of Indian law.

{context}

Provide:
1. Clear definition and explanation
2. Legal basis and relevant provisions
3. Practical applications and examples
4. Common misconceptions or important notes

Explanation:
"""
            )
            
            context = ""
            if context_documents:
                context = f"Context from legal documents:\n{self._format_documents_for_prompt(context_documents)}"
            else:
                context = "General explanation without specific document context."
            
            chain = LLMChain(llm=self.llm, prompt=explanation_prompt)
            
            response = await chain.arun(
                concept=concept,
                context=context
            )
            
            return {
                "explanation": response.strip(),
                "concept": concept,
                "has_context": bool(context_documents)
            }
            
        except Exception as e:
            logger.error(f"Error in explain_legal_concept: {str(e)}")
            raise 