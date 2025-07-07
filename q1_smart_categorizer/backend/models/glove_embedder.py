import numpy as np
import logging
from typing import List, Dict
from pathlib import Path
import requests
import zipfile
import os

from models.base_embedder import BaseEmbedder
from config.settings import DATA_DIR, MODEL_CONFIGS

logger = logging.getLogger("model")

class GloVeEmbedder(BaseEmbedder):
    """GloVe embeddings using pre-trained vectors"""
    
    def __init__(self):
        super().__init__("glove")
        self.word_vectors = {}
        self.embedding_dim = MODEL_CONFIGS["glove"]["embedding_dim"]
        self.model_file = DATA_DIR / MODEL_CONFIGS["glove"]["model_file"]
        
    def download_glove_vectors(self):
        """Download GloVe vectors if not present"""
        if self.model_file.exists():
            logger.info("GloVe vectors already exist")
            return
        
        logger.info("Downloading GloVe vectors...")
        
        # GloVe 6B vectors URL
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = DATA_DIR / "glove.6B.zip"
        
        try:
            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the specific file we need
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract("glove.6B.300d.txt", DATA_DIR)
            
            # Remove the zip file
            os.remove(zip_path)
            
            logger.info("GloVe vectors downloaded and extracted")
            
        except Exception as e:
            logger.error(f"Error downloading GloVe vectors: {str(e)}")
            raise
    
    def load_model(self):
        """Load GloVe word vectors"""
        if self.word_vectors:
            return
        
        # Download if not present
        self.download_glove_vectors()
        
        logger.info(f"Loading GloVe vectors from {self.model_file}")
        
        try:
            word_count = 0
            with open(self.model_file, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.array(values[1:], dtype='float32')
                    self.word_vectors[word] = vector
                    word_count += 1
                    
                    if word_count % 50000 == 0:
                        logger.info(f"Loaded {word_count} word vectors")
            
            logger.info(f"Loaded {len(self.word_vectors)} GloVe word vectors")
            
        except Exception as e:
            logger.error(f"Error loading GloVe vectors: {str(e)}")
            raise
    
    def get_word_vector(self, word: str) -> np.ndarray:
        """Get vector for a single word"""
        word = word.lower()
        if word in self.word_vectors:
            return self.word_vectors[word]
        else:
            # Return random vector for unknown words
            return np.random.normal(size=self.embedding_dim)
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings by averaging word vectors"""
        if not self.word_vectors:
            self.load_model()
        
        logger.info(f"Generating GloVe embeddings for {len(texts)} texts")
        
        embeddings = []
        
        for text in texts:
            words = text.split()
            if not words:
                # Handle empty text
                embeddings.append(np.zeros(self.embedding_dim))
                continue
            
            # Get vectors for all words
            word_vectors = [self.get_word_vector(word) for word in words]
            
            # Average the vectors
            text_embedding = np.mean(word_vectors, axis=0)
            embeddings.append(text_embedding)
        
        return np.array(embeddings) 