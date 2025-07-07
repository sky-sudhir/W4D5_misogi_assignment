import numpy as np
import logging
from typing import List
from sentence_transformers import SentenceTransformer

from models.base_embedder import BaseEmbedder
from config.settings import MODEL_CONFIGS

logger = logging.getLogger("model")

class SentenceBertEmbedder(BaseEmbedder):
    """Sentence-BERT embeddings"""
    
    def __init__(self):
        super().__init__("sbert")
        self.model = None
        self.model_name = MODEL_CONFIGS["sbert"]["model_name"]
        
    def load_model(self):
        """Load Sentence-BERT model"""
        if self.model is not None:
            return
        
        logger.info(f"Loading Sentence-BERT model: {self.model_name}")
        
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Sentence-BERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Sentence-BERT model: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate Sentence-BERT embeddings"""
        if self.model is None:
            self.load_model()
        
        logger.info(f"Generating Sentence-BERT embeddings for {len(texts)} texts")
        
        try:
            # Generate embeddings in batches
            batch_size = 32
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                embeddings.extend(batch_embeddings)
                
                if (i + batch_size) % 100 == 0:
                    logger.info(f"Processed {min(i + batch_size, len(texts))} texts")
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error generating Sentence-BERT embeddings: {str(e)}")
            raise 