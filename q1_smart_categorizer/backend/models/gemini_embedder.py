import numpy as np
import logging
from typing import List
import google.generativeai as genai
import time

from models.base_embedder import BaseEmbedder
from config.settings import MODEL_CONFIGS, get_gemini_api_key

logger = logging.getLogger("model")

class GeminiEmbedder(BaseEmbedder):
    """Gemini text embeddings"""
    
    def __init__(self):
        super().__init__("gemini")
        self.model_name = MODEL_CONFIGS["gemini"]["model_name"]
        self.api_key = None
        self.client = None
        
    def load_model(self):
        """Initialize Gemini client"""
        if self.client is not None:
            return
        
        logger.info("Initializing Gemini client")
        
        try:
            self.api_key = get_gemini_api_key()
            genai.configure(api_key=self.api_key)
            self.client = genai
            
            logger.info("Gemini client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate Gemini embeddings"""
        if self.client is None:
            self.load_model()
        
        logger.info(f"Generating Gemini embeddings for {len(texts)} texts")
        
        embeddings = []
        
        try:
            for i, text in enumerate(texts):
                # Rate limiting - Gemini has API limits
                if i > 0 and i % 10 == 0:
                    time.sleep(1)  # Wait 1 second every 10 requests
                
                try:
                    # Generate embedding
                    result = genai.embed_content(
                        model=f"models/{self.model_name}",
                        content=text,
                        task_type="classification"
                    )
                    
                    embedding = np.array(result['embedding'])
                    embeddings.append(embedding)
                    
                    if (i + 1) % 50 == 0:
                        logger.info(f"Processed {i + 1} texts")
                        
                except Exception as e:
                    logger.warning(f"Error generating embedding for text {i}: {str(e)}")
                    # Use zero vector as fallback
                    embeddings.append(np.zeros(MODEL_CONFIGS["gemini"]["embedding_dim"]))
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error generating Gemini embeddings: {str(e)}")
            raise 