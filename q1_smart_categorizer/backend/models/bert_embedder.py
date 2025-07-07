import numpy as np
import logging
from typing import List
import torch
from transformers import BertTokenizer, BertModel

from models.base_embedder import BaseEmbedder
from config.settings import MODEL_CONFIGS

logger = logging.getLogger("model")

class BertEmbedder(BaseEmbedder):
    """BERT embeddings using [CLS] token"""
    
    def __init__(self):
        super().__init__("bert")
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = MODEL_CONFIGS["bert"]["max_length"]
        self.model_name = MODEL_CONFIGS["bert"]["model_name"]
        
    def load_model(self):
        """Load BERT model and tokenizer"""
        if self.model is not None:
            return
        
        logger.info(f"Loading BERT model: {self.model_name}")
        
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"BERT model loaded on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading BERT model: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate BERT embeddings using [CLS] token"""
        if self.model is None:
            self.load_model()
        
        logger.info(f"Generating BERT embeddings for {len(texts)} texts")
        
        embeddings = []
        batch_size = 16  # Process in batches to manage memory
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Get BERT outputs
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Extract [CLS] token embeddings (first token)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                
                # Move back to CPU and convert to numpy
                cls_embeddings = cls_embeddings.cpu().numpy()
                embeddings.extend(cls_embeddings)
                
                if (i + batch_size) % 100 == 0:
                    logger.info(f"Processed {min(i + batch_size, len(texts))} texts")
        
        return np.array(embeddings) 