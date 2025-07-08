import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class ChromaManager:
    """Manages ChromaDB collections and operations"""
    
    def __init__(self):
        self.api_key = os.getenv("CHROMA_API_KEY")
        self.tenant = os.getenv("CHROMA_TENANT")
        self.database = os.getenv("CHROMA_DATABASE")
        
        # Try ChromaDB Cloud first, fallback to local
        try:
            if all([self.api_key, self.tenant, self.database]):
                # Initialize ChromaDB Cloud client
                self.client = chromadb.CloudClient(
                    api_key=self.api_key,
                    tenant=self.tenant,
                    database=self.database
                )
                logger.info("Connected to ChromaDB Cloud")
            else:
                raise ValueError("ChromaDB Cloud credentials not found")
        except Exception as e:
            logger.warning(f"ChromaDB Cloud connection failed: {e}")
            logger.info("Falling back to local ChromaDB")
            # Fallback to local ChromaDB
            self.client = chromadb.PersistentClient(path="./chroma_db")
            logger.info("Connected to local ChromaDB")
        
        # Embedding function
        self.embedding_function = DefaultEmbeddingFunction()
        
        # Collection names
        self.collections = {
            "cosine": "legal_docs_cosine",
            "euclidean": "legal_docs_euclidean",
            "mmr": "legal_docs_mmr",
            "hybrid": "legal_docs_hybrid"
        }
        
        self._collections_cache = {}
    
    async def initialize_collections(self):
        """Initialize all ChromaDB collections"""
        try:
            # Cosine similarity collection (default)
            self._collections_cache["cosine"] = self.client.get_or_create_collection(
                name=self.collections["cosine"],
                embedding_function=self.embedding_function
            )
            
            # Euclidean distance collection
            self._collections_cache["euclidean"] = self.client.get_or_create_collection(
                name=self.collections["euclidean"],
                embedding_function=self.embedding_function
            )
            
            # MMR collection (same as cosine for storage)
            self._collections_cache["mmr"] = self.client.get_or_create_collection(
                name=self.collections["mmr"],
                embedding_function=self.embedding_function
            )
            
            # Hybrid collection (same as cosine for storage)
            self._collections_cache["hybrid"] = self.client.get_or_create_collection(
                name=self.collections["hybrid"],
                embedding_function=self.embedding_function
            )
            
            logger.info("All ChromaDB collections initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB collections: {str(e)}")
            raise
    
    def get_collection(self, collection_type: str):
        """Get a specific collection"""
        if collection_type not in self._collections_cache:
            raise ValueError(f"Collection type '{collection_type}' not found")
        return self._collections_cache[collection_type]
    
    async def add_documents(
        self,
        collection_type: str,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ):
        """Add documents to a collection"""
        try:
            collection = self.get_collection(collection_type)
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to {collection_type} collection")
            
        except Exception as e:
            logger.error(f"Error adding documents to {collection_type}: {str(e)}")
            raise
    
    async def query_collection(
        self,
        collection_type: str,
        query_texts: List[str],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query a collection"""
        try:
            collection = self.get_collection(collection_type)
            results = collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            return results
            
        except Exception as e:
            logger.error(f"Error querying {collection_type} collection: {str(e)}")
            raise
    
    async def get_collections_info(self) -> Dict[str, Any]:
        """Get information about all collections"""
        try:
            info = {}
            for collection_type, collection_name in self.collections.items():
                if collection_type in self._collections_cache:
                    collection = self._collections_cache[collection_type]
                    count = collection.count()
                    info[collection_type] = {
                        "name": collection_name,
                        "document_count": count,
                        "status": "active"
                    }
                else:
                    info[collection_type] = {
                        "name": collection_name,
                        "document_count": 0,
                        "status": "not_initialized"
                    }
            return info
            
        except Exception as e:
            logger.error(f"Error getting collections info: {str(e)}")
            raise
    
    async def test_connection(self) -> bool:
        """Test ChromaDB connection"""
        try:
            # Try to list collections
            collections = self.client.list_collections()
            logger.info(f"ChromaDB connection successful. Found {len(collections)} collections.")
            return True
        except Exception as e:
            logger.error(f"ChromaDB connection test failed: {str(e)}")
            return False
    
    async def delete_collection(self, collection_type: str):
        """Delete a collection"""
        try:
            if collection_type in self.collections:
                self.client.delete_collection(name=self.collections[collection_type])
                if collection_type in self._collections_cache:
                    del self._collections_cache[collection_type]
                logger.info(f"Deleted {collection_type} collection")
        except Exception as e:
            logger.error(f"Error deleting {collection_type} collection: {str(e)}")
            raise 