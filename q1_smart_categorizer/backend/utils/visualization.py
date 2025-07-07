import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import umap
import logging

from config.settings import UMAP_CONFIG
from utils.json_encoder import convert_numpy_types

logger = logging.getLogger("model")

class UMAPVisualizer:
    """UMAP visualization for embeddings"""
    
    def __init__(self):
        self.reducer = umap.UMAP(
            n_neighbors=UMAP_CONFIG["n_neighbors"],
            min_dist=UMAP_CONFIG["min_dist"],
            n_components=UMAP_CONFIG["n_components"],
            metric=UMAP_CONFIG["metric"],
            random_state=UMAP_CONFIG["random_state"]
        )
    
    def fit_transform(self, embeddings: np.ndarray, labels: List[str]) -> Dict[str, any]:
        """Fit UMAP and transform embeddings to 2D"""
        logger.info(f"Fitting UMAP on {len(embeddings)} embeddings")
        
        try:
            # Fit and transform embeddings
            embedding_2d = self.reducer.fit_transform(embeddings)
            
            # Create DataFrame for easier handling
            df = pd.DataFrame({
                'x': embedding_2d[:, 0],
                'y': embedding_2d[:, 1],
                'label': labels
            })
            
            # Calculate cluster centers for each category
            cluster_centers = df.groupby('label')[['x', 'y']].mean().to_dict('index')
            
            # Calculate statistics
            stats = {
                'n_samples': int(len(embeddings)),
                'n_categories': int(len(set(labels))),
                'embedding_dim': int(embeddings.shape[1]),
                'cluster_centers': convert_numpy_types(cluster_centers)
            }
            
            result = {
                'data': convert_numpy_types(df.to_dict('records')),
                'stats': stats
            }
            
            logger.info("UMAP transformation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in UMAP transformation: {str(e)}")
            raise
    
    def transform_new_data(self, new_embeddings: np.ndarray) -> np.ndarray:
        """Transform new embeddings using fitted UMAP"""
        if not hasattr(self.reducer, 'embedding_'):
            raise ValueError("UMAP reducer has not been fitted yet")
        
        try:
            return self.reducer.transform(new_embeddings)
        except Exception as e:
            logger.error(f"Error transforming new data: {str(e)}")
            raise

def create_umap_visualization(embeddings: np.ndarray, labels: List[str]) -> Dict[str, any]:
    """Create UMAP visualization data"""
    visualizer = UMAPVisualizer()
    return visualizer.fit_transform(embeddings, labels) 