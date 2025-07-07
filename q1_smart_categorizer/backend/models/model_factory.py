from typing import Dict, Type
from models.base_embedder import BaseEmbedder
from models.glove_embedder import GloVeEmbedder
from models.bert_embedder import BertEmbedder
from models.sbert_embedder import SentenceBertEmbedder
from models.gemini_embedder import GeminiEmbedder

class ModelFactory:
    """Factory class for creating embedding models"""
    
    _models: Dict[str, Type[BaseEmbedder]] = {
        "glove": GloVeEmbedder,
        "bert": BertEmbedder,
        "sbert": SentenceBertEmbedder,
        "gemini": GeminiEmbedder
    }
    
    @classmethod
    def create_model(cls, model_name: str) -> BaseEmbedder:
        """Create an embedding model instance"""
        if model_name not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
        
        return cls._models[model_name]()
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model names"""
        return list(cls._models.keys())
    
    @classmethod
    def is_valid_model(cls, model_name: str) -> bool:
        """Check if model name is valid"""
        return model_name in cls._models 