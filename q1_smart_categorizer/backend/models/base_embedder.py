from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib
import logging
from pathlib import Path

from config.settings import MODELS_DIR, TRAIN_CONFIG

logger = logging.getLogger("model")

class BaseEmbedder(ABC):
    """Base class for all embedding models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embeddings = None
        self.labels = None
        self.classifier = None
        self.is_trained = False
        self.model_path = MODELS_DIR / f"{model_name}_model.pkl"
        self.embeddings_path = MODELS_DIR / f"{model_name}_embeddings.npy"
        self.labels_path = MODELS_DIR / f"{model_name}_labels.npy"
        
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for given texts"""
        pass
    
    @abstractmethod
    def load_model(self):
        """Load the embedding model"""
        pass
    
    def train_classifier(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Train logistic regression classifier and return metrics"""
        logger.info(f"Training classifier for {self.model_name}")
        
        try:
            # Generate embeddings for training data
            logger.info("Generating embeddings for training data...")
            train_texts = train_df['Text'].tolist()
            train_embeddings = self.generate_embeddings(train_texts)
            train_labels = train_df['Category'].tolist()
            
            # Generate embeddings for test data
            logger.info("Generating embeddings for test data...")
            test_texts = test_df['Text'].tolist()
            test_embeddings = self.generate_embeddings(test_texts)
            test_labels = test_df['Category'].tolist()
            
            # Train classifier
            logger.info("Training logistic regression classifier...")
            self.classifier = LogisticRegression(
                max_iter=TRAIN_CONFIG["max_iter"],
                solver=TRAIN_CONFIG["solver"],
                random_state=TRAIN_CONFIG["random_state"]
            )
            
            self.classifier.fit(train_embeddings, train_labels)
            
            # Make predictions
            train_predictions = self.classifier.predict(train_embeddings)
            test_predictions = self.classifier.predict(test_embeddings)
            
            # Calculate metrics
            train_accuracy = accuracy_score(train_labels, train_predictions)
            test_accuracy = accuracy_score(test_labels, test_predictions)
            
            # Detailed metrics for test set
            precision, recall, f1, support = precision_recall_fscore_support(
                test_labels, test_predictions, average='weighted'
            )
            
            # Per-class metrics
            classification_rep = classification_report(
                test_labels, test_predictions, output_dict=True
            )
            
            # Convert numpy types to Python native types for JSON serialization
            def convert_numpy_types(obj):
                """Recursively convert numpy types to Python native types"""
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif hasattr(obj, 'tolist'):  # numpy array
                    return obj.tolist()
                else:
                    return obj
            
            metrics = {
                "model_name": self.model_name,
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "classification_report": convert_numpy_types(classification_rep),
                "train_samples": int(len(train_texts)),
                "test_samples": int(len(test_texts))
            }
            
            # Store embeddings and labels for UMAP visualization
            self.embeddings = np.vstack([train_embeddings, test_embeddings])
            self.labels = train_labels + test_labels
            
            # Save model and data
            self.save_model()
            self.is_trained = True
            
            logger.info(f"Training completed for {self.model_name}")
            logger.info(f"Test accuracy: {test_accuracy:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training classifier for {self.model_name}: {str(e)}")
            raise
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict category for a single text"""
        if not self.is_trained and not self.load_saved_model():
            raise ValueError(f"Model {self.model_name} is not trained")
        
        try:
            # Generate embedding
            embedding = self.generate_embeddings([text])
            
            # Predict
            prediction = self.classifier.predict(embedding)[0]
            probabilities = self.classifier.predict_proba(embedding)[0]
            confidence = float(max(probabilities))
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Error predicting with {self.model_name}: {str(e)}")
            raise
    
    def save_model(self):
        """Save trained model and embeddings"""
        try:
            # Save classifier
            joblib.dump(self.classifier, self.model_path)
            
            # Save embeddings and labels
            if self.embeddings is not None:
                np.save(self.embeddings_path, self.embeddings)
            if self.labels is not None:
                np.save(self.labels_path, self.labels)
                
            logger.info(f"Model saved for {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error saving model for {self.model_name}: {str(e)}")
            raise
    
    def load_saved_model(self) -> bool:
        """Load saved model and embeddings"""
        try:
            if not self.model_path.exists():
                return False
            
            # Load classifier
            self.classifier = joblib.load(self.model_path)
            
            # Load embeddings and labels if they exist
            if self.embeddings_path.exists():
                self.embeddings = np.load(self.embeddings_path)
            if self.labels_path.exists():
                self.labels = np.load(self.labels_path).tolist()
            
            self.is_trained = True
            logger.info(f"Model loaded for {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model for {self.model_name}: {str(e)}")
            return False
    
    def get_embeddings_for_umap(self) -> Tuple[np.ndarray, List[str]]:
        """Get embeddings and labels for UMAP visualization"""
        if self.embeddings is None or self.labels is None:
            if not self.load_saved_model():
                raise ValueError(f"No embeddings available for {self.model_name}")
        
        return self.embeddings, self.labels 