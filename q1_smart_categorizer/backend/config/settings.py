import os
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR.parent / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Dataset configuration
DATASET_PATH = BASE_DIR / "dataset.csv"
CATEGORIES = ["tech", "business", "sport", "politics", "entertainment"]
CATEGORY_MAPPING = {
    "business": "finance",  # Map business to finance as per requirements
    "tech": "tech",
    "sport": "sport", 
    "politics": "politics",
    "entertainment": "entertainment"
}

# Model configurations
MODEL_CONFIGS = {
    "glove": {
        "name": "GloVe",
        "embedding_dim": 300,
        "model_file": "glove.6B.300d.txt",
        "requires_download": True
    },
    "bert": {
        "name": "BERT",
        "model_name": "bert-base-uncased",
        "embedding_dim": 768,
        "max_length": 512
    },
    "sbert": {
        "name": "Sentence-BERT",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384
    },
    "gemini": {
        "name": "Gemini",
        "model_name": "text-embedding-004",
        "embedding_dim": 768,
        "api_key_env": "GEMINI_API_KEY"
    }
}

# Training configuration
TRAIN_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "max_iter": 1000,
    "solver": "liblinear"
}

# API configuration
API_CONFIG = {
    "host": "localhost",
    "port": 8000,
    "title": "Smart Article Categorizer API",
    "description": "API for article classification using multiple embedding models",
    "version": "1.0.0"
}

# UMAP configuration
UMAP_CONFIG = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "n_components": 2,
    "metric": "cosine",
    "random_state": 42
}

# Text preprocessing configuration
TEXT_PREPROCESSING = {
    "max_length": 1000,  # Maximum number of words to keep
    "min_length": 10,    # Minimum number of words required
    "remove_html": True,
    "remove_urls": True,
    "remove_special_chars": True,
    "to_lowercase": True
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_CONFIGS[model_name]

def get_gemini_api_key() -> str:
    """Get Gemini API key from environment"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return api_key 