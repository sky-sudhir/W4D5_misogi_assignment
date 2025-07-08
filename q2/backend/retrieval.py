from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import re

from chroma_utils import ChromaManager

logger = logging.getLogger(__name__)

class RetrievalEngine:
    """Handles all retrieval methods for legal document search"""
    
    def __init__(self, chroma_manager: ChromaManager):
        self.chroma_manager = chroma_manager
        self.legal_terms = self._load_legal_terms()
        
    def _load_legal_terms(self) -> List[str]:
        """Load legal terms for entity matching"""
        return [
            # Income Tax Act terms
            "section 80c", "section 80d", "section 24", "section 54",
            "capital gains", "house property", "salary income", "business income",
            "deduction", "exemption", "assessment year", "financial year",
            
            # GST Act terms
            "input tax credit", "output tax", "reverse charge", "composition scheme",
            "gst registration", "place of supply", "taxable supply", "exempt supply",
            "igst", "cgst", "sgst", "utgst",
            
            # General legal terms
            "plaintiff", "defendant", "appellant", "respondent", "writ petition",
            "high court", "supreme court", "civil procedure code", "criminal procedure code",
            "constitution", "fundamental rights", "directive principles",
            
            # Property law terms
            "sale deed", "lease deed", "mortgage", "easement", "title deed",
            "registration", "stamp duty", "property tax", "land revenue"
        ]
    
    async def cosine_similarity_search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform cosine similarity search"""
        try:
            results = await self.chroma_manager.query_collection(
                collection_type="cosine",
                query_texts=[query],
                n_results=top_k,
                where=where
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i],
                    "similarity_score": 1 - results['distances'][0][i],  # Convert distance to similarity
                    "method": "cosine"
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in cosine similarity search: {str(e)}")
            raise
    
    async def euclidean_distance_search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform Euclidean distance search"""
        try:
            results = await self.chroma_manager.query_collection(
                collection_type="euclidean",
                query_texts=[query],
                n_results=top_k,
                where=where
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]
                # Convert Euclidean distance to similarity score (0-1 range)
                similarity_score = 1 / (1 + distance)
                
                formatted_results.append({
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": distance,
                    "similarity_score": similarity_score,
                    "method": "euclidean"
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in Euclidean distance search: {str(e)}")
            raise
    
    async def mmr_search(
        self,
        query: str,
        top_k: int = 5,
        lambda_mult: float = 0.5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform MMR (Maximal Marginal Relevance) search"""
        try:
            # First, get more results than needed for MMR calculation
            initial_k = min(top_k * 3, 50)  # Get 3x more results for MMR selection
            
            results = await self.chroma_manager.query_collection(
                collection_type="mmr",
                query_texts=[query],
                n_results=initial_k,
                where=where
            )
            
            if not results['documents'][0]:
                return []
            
            # Implement MMR algorithm
            documents = results['documents'][0]
            distances = results['distances'][0]
            metadatas = results['metadatas'][0]
            
            # Convert distances to similarities
            similarities = [1 - d for d in distances]
            
            # MMR selection
            selected_indices = []
            remaining_indices = list(range(len(documents)))
            
            # Select first document (highest similarity)
            if remaining_indices:
                best_idx = remaining_indices[0]  # Already sorted by similarity
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            
            # Select remaining documents using MMR
            while len(selected_indices) < top_k and remaining_indices:
                best_mmr_score = -float('inf')
                best_idx = None
                
                for idx in remaining_indices:
                    # Calculate relevance score
                    relevance = similarities[idx]
                    
                    # Calculate maximum similarity to already selected documents
                    max_sim_to_selected = 0
                    if selected_indices:
                        for selected_idx in selected_indices:
                            # Simple text similarity for diversity calculation
                            sim = self._calculate_text_similarity(
                                documents[idx], 
                                documents[selected_idx]
                            )
                            max_sim_to_selected = max(max_sim_to_selected, sim)
                    
                    # MMR score
                    mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim_to_selected
                    
                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_idx = idx
                
                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
                else:
                    break
            
            # Format results
            formatted_results = []
            for idx in selected_indices:
                formatted_results.append({
                    "document": documents[idx],
                    "metadata": metadatas[idx],
                    "distance": distances[idx],
                    "similarity_score": similarities[idx],
                    "method": "mmr"
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in MMR search: {str(e)}")
            raise
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using TF-IDF and cosine similarity"""
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0
    
    def _calculate_legal_entity_score(self, query: str, document: str) -> float:
        """Calculate legal entity matching score"""
        query_lower = query.lower()
        document_lower = document.lower()
        
        # Find legal terms in query
        query_terms = []
        for term in self.legal_terms:
            if term.lower() in query_lower:
                query_terms.append(term.lower())
        
        if not query_terms:
            return 0.0
        
        # Calculate matching score
        matches = 0
        for term in query_terms:
            if term in document_lower:
                matches += 1
        
        # Additional scoring for section numbers
        section_pattern = r'section\s+(\d+[a-z]*)'
        query_sections = set(re.findall(section_pattern, query_lower))
        doc_sections = set(re.findall(section_pattern, document_lower))
        
        if query_sections and doc_sections:
            section_matches = len(query_sections.intersection(doc_sections))
            matches += section_matches * 2  # Weight section matches higher
        
        # Normalize score
        max_possible_matches = len(query_terms) + len(query_sections) * 2
        if max_possible_matches == 0:
            return 0.0
        
        return min(matches / max_possible_matches, 1.0)
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        cosine_weight: float = 0.6,
        entity_weight: float = 0.4,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search (cosine + legal entity matching)"""
        try:
            # Get cosine similarity results
            cosine_results = await self.cosine_similarity_search(
                query=query,
                top_k=top_k * 2,  # Get more results for reranking
                where=where
            )
            
            # Calculate hybrid scores
            hybrid_results = []
            for result in cosine_results:
                cosine_score = result['similarity_score']
                entity_score = self._calculate_legal_entity_score(
                    query, 
                    result['document']
                )
                
                # Hybrid score calculation
                hybrid_score = (cosine_weight * cosine_score) + (entity_weight * entity_score)
                
                hybrid_results.append({
                    "document": result['document'],
                    "metadata": result['metadata'],
                    "distance": result['distance'],
                    "similarity_score": hybrid_score,
                    "cosine_score": cosine_score,
                    "entity_score": entity_score,
                    "method": "hybrid"
                })
            
            # Sort by hybrid score and return top_k
            hybrid_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return hybrid_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise
    
    async def search_all_methods(
        self,
        query: str,
        top_k: int = 5,
        methods: List[str] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search using all specified methods"""
        if methods is None:
            methods = ["cosine", "euclidean", "mmr", "hybrid"]
        
        results = {}
        
        try:
            # Execute all searches in parallel (conceptually)
            if "cosine" in methods:
                results["cosine"] = await self.cosine_similarity_search(
                    query=query, top_k=top_k, where=where
                )
            
            if "euclidean" in methods:
                results["euclidean"] = await self.euclidean_distance_search(
                    query=query, top_k=top_k, where=where
                )
            
            if "mmr" in methods:
                results["mmr"] = await self.mmr_search(
                    query=query, top_k=top_k, where=where
                )
            
            if "hybrid" in methods:
                results["hybrid"] = await self.hybrid_search(
                    query=query, top_k=top_k, where=where
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search_all_methods: {str(e)}")
            raise
    
    def calculate_metrics(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        relevant_docs: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Calculate precision, recall, and diversity metrics"""
        metrics = {}
        
        for method, method_results in results.items():
            method_metrics = {
                "precision_at_5": 0.0,
                "recall_at_5": 0.0,
                "diversity_score": 0.0,
                "average_similarity": 0.0
            }
            
            if method_results:
                # Calculate average similarity
                similarities = [r['similarity_score'] for r in method_results]
                method_metrics["average_similarity"] = np.mean(similarities)
                
                # Calculate diversity (average pairwise dissimilarity)
                if len(method_results) > 1:
                    diversity_scores = []
                    for i in range(len(method_results)):
                        for j in range(i + 1, len(method_results)):
                            sim = self._calculate_text_similarity(
                                method_results[i]['document'],
                                method_results[j]['document']
                            )
                            diversity_scores.append(1 - sim)  # Dissimilarity
                    
                    method_metrics["diversity_score"] = np.mean(diversity_scores)
                
                # Calculate precision and recall if relevant docs provided
                if relevant_docs:
                    retrieved_docs = [r['document'] for r in method_results]
                    relevant_retrieved = len(set(retrieved_docs) & set(relevant_docs))
                    
                    method_metrics["precision_at_5"] = relevant_retrieved / len(retrieved_docs)
                    method_metrics["recall_at_5"] = relevant_retrieved / len(relevant_docs)
            
            metrics[method] = method_metrics
        
        return metrics 