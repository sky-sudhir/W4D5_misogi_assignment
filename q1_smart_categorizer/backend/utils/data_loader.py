import pandas as pd
import numpy as np
import re
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
import logging

from config.settings import (
    DATASET_PATH, CATEGORIES, CATEGORY_MAPPING, 
    TRAIN_CONFIG, TEXT_PREPROCESSING
)

logger = logging.getLogger("data")

def clean_text(text: str) -> str:
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    if TEXT_PREPROCESSING["to_lowercase"]:
        text = text.lower()
    
    # Remove HTML tags
    if TEXT_PREPROCESSING["remove_html"]:
        text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    if TEXT_PREPROCESSING["remove_urls"]:
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove special characters but keep basic punctuation
    if TEXT_PREPROCESSING["remove_special_chars"]:
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def filter_by_length(text: str) -> bool:
    """Filter text by length requirements"""
    words = text.split()
    word_count = len(words)
    
    return (TEXT_PREPROCESSING["min_length"] <= word_count <= TEXT_PREPROCESSING["max_length"])

def load_dataset() -> pd.DataFrame:
    """Load and preprocess the dataset"""
    logger.info(f"Loading dataset from {DATASET_PATH}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(DATASET_PATH)
        logger.info(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
        
        # Check required columns
        required_columns = ['Text', 'Category']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Filter for valid categories
        valid_categories = set(CATEGORIES)
        df = df[df['Category'].isin(valid_categories)]
        logger.info(f"Filtered to {len(df)} rows with valid categories")
        
        # Clean text
        logger.info("Cleaning text data...")
        df['Text'] = df['Text'].apply(clean_text)
        
        # Filter by text length
        logger.info("Filtering by text length...")
        df = df[df['Text'].apply(filter_by_length)]
        logger.info(f"After length filtering: {len(df)} rows")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['Text'])
        logger.info(f"Removed {initial_count - len(df)} duplicate texts")
        
        # Map categories if needed
        df['Category'] = df['Category'].map(CATEGORY_MAPPING)
        
        # Log category distribution
        category_counts = df['Category'].value_counts()
        logger.info(f"Category distribution: {category_counts.to_dict()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def prepare_train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train and test sets"""
    logger.info("Splitting dataset into train and test sets")
    
    try:
        # Stratified split to maintain category distribution
        train_df, test_df = train_test_split(
            df,
            test_size=TRAIN_CONFIG["test_size"],
            random_state=TRAIN_CONFIG["random_state"],
            stratify=df['Category']
        )
        
        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        # Log category distribution in splits
        train_dist = train_df['Category'].value_counts()
        test_dist = test_df['Category'].value_counts()
        logger.info(f"Train distribution: {train_dist.to_dict()}")
        logger.info(f"Test distribution: {test_dist.to_dict()}")
        
        return train_df, test_df
        
    except Exception as e:
        logger.error(f"Error splitting dataset: {str(e)}")
        raise

def get_category_labels() -> List[str]:
    """Get unique category labels"""
    return list(CATEGORY_MAPPING.values())

def get_dataset_stats() -> Dict[str, any]:
    """Get dataset statistics"""
    try:
        df = load_dataset()
        
        # Convert numpy types to Python native types for JSON serialization
        stats = {
            "total_samples": int(len(df)),
            "categories": get_category_labels(),
            "category_distribution": {k: int(v) for k, v in df['Category'].value_counts().to_dict().items()},
            "avg_text_length": float(df['Text'].apply(lambda x: len(x.split())).mean()),
            "min_text_length": int(df['Text'].apply(lambda x: len(x.split())).min()),
            "max_text_length": int(df['Text'].apply(lambda x: len(x.split())).max())
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting dataset stats: {str(e)}")
        raise 