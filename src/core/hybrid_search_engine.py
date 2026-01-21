"""
Advanced Hybrid Search Engine
Combines BM25 (lexical) + FAISS (semantic) with Reciprocal Rank Fusion
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer
import torch
from collections import defaultdict
from loguru import logger


@dataclass
class SearchResult:
    """Unified search result structure"""
    chunk_id: str
    content: str
    score: float
    source: str  # 'bm25', 'semantic', or 'hybrid'
    metadata: Dict
    rank: int


class HybridSearchEngine:
    """
    Production-grade hybrid search combining lexical and semantic retrieval
    with intelligent fusion and adaptive weighting
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_embeddings: bool = True,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        if self.device == "cuda":
            self.embedder.half()  # FP16 for memory efficiency
        
        # Storage
        self.documents: List[Dict] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.cache_embeddings = cache_embeddings
        
        # Adaptive weights (will be tuned based on query type)
        self.default_weights = {
            'bm25': 0.4,
            'semantic': 0.6
        }
        
        logger.info(f"HybridSearchEngine initialized on {self.device}")
    
    def index_documents(self, documents: List[Dict]) -> None:
        """
        Index documents for both BM25 and semantic search
        
        Args:
            documents: List of dicts with keys: 'id', 'content', 'metadata'
        """
        if not documents:
            logger.warning("No documents to index")
            return
            
        self.documents = documents
        
        # Extract text content
        texts = [doc['content'] for doc in documents]
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # Build FAISS index
        logger.info("Building semantic index...")
        embeddings = self._batch_encode(texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index (Inner Product = cosine similarity after normalization)
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)
        
        # Cache embeddings if enabled
        if self.cache_embeddings:
            for doc, emb in zip(documents, embeddings):
                self.embeddings_cache[doc['id']] = emb
        
        logger.success(f"Indexed {len(documents)} documents")
    
    def _batch_encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts in batches for memory efficiency"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                embeddings = self.embedder.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=False  # We'll normalize in FAISS
                )
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings).astype('float32')
    
    def _detect_query_type(self, query: str) -> Dict[str, float]:
        """
        Analyze query to determine optimal search weights
        """
        query_lower = query.lower()
        
        # Keywords indicating specific fact-finding
        specific_indicators = ['who', 'when', 'where', 'what year', 'how many', 
                              'list', 'name', 'define', 'specific']
        
        # Keywords indicating conceptual queries
        conceptual_indicators = ['why', 'how does', 'explain', 'relationship', 
                                'compare', 'difference', 'impact', 'effect']
        
        specific_score = sum(1 for ind in specific_indicators if ind in query_lower)
        conceptual_score = sum(1 for ind in conceptual_indicators if ind in query_lower)
        
        # Adjust weights based on query type
        if specific_score > conceptual_score:
            return {'bm25': 0.65, 'semantic': 0.35}
        elif conceptual_score > specific_score:
            return {'bm25': 0.30, 'semantic': 0.70}
        else:
            return self.default_weights
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        use_adaptive_weights: bool = True,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> List[SearchResult]:
        """
        Hybrid search with Reciprocal Rank Fusion (RRF)
        """
        if not self.documents:
            return []
        
        # Determine weights
        if custom_weights:
            weights = custom_weights
        elif use_adaptive_weights:
            weights = self._detect_query_type(query)
        else:
            weights = self.default_weights
        
        # BM25 Search
        bm25_scores = self._bm25_search(query, top_k * 2)
        
        # Semantic Search
        semantic_scores = self._semantic_search(query, top_k * 2)
        
        # Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            bm25_results=bm25_scores,
            semantic_results=semantic_scores,
            weights=weights,
            k=60
        )
        
        # Convert to SearchResult objects
        results = []
        for rank, (doc_id, score) in enumerate(fused_results[:top_k], 1):
            doc = next((d for d in self.documents if d['id'] == doc_id), None)
            if doc:
                results.append(SearchResult(
                    chunk_id=doc_id,
                    content=doc['content'],
                    score=score,
                    source='hybrid',
                    metadata=doc.get('metadata', {}),
                    rank=rank
                ))
        
        return results
    
    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Execute BM25 lexical search"""
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(self.documents[i]['id'], float(scores[i])) for i in top_indices]
    
    def _semantic_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Execute FAISS semantic search"""
        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False
        ).astype('float32')
        
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.documents)))
        
        return [(self.documents[i]['id'], float(distances[0][idx])) 
                for idx, i in enumerate(indices[0]) if i < len(self.documents)]
    
    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[str, float]],
        semantic_results: List[Tuple[str, float]],
        weights: Dict[str, float],
        k: int = 60
    ) -> List[Tuple[str, float]]:
        """
        Fuse rankings using Reciprocal Rank Fusion
        RRF Score = Σ (weight / (k + rank))
        """
        fused_scores = defaultdict(float)
        
        for rank, (doc_id, _) in enumerate(bm25_results, 1):
            fused_scores[doc_id] += weights['bm25'] / (k + rank)
        
        for rank, (doc_id, _) in enumerate(semantic_results, 1):
            fused_scores[doc_id] += weights['semantic'] / (k + rank)
        
        sorted_results = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results
    
    def get_all_embeddings(self) -> np.ndarray:
        """Get all cached embeddings as array"""
        if not self.embeddings_cache:
            return np.array([])
        return np.vstack(list(self.embeddings_cache.values()))
    
    def get_all_texts(self) -> List[str]:
        """Get all indexed texts"""
        return [doc['content'] for doc in self.documents]
    
    def get_all_sources(self) -> List[str]:
        """Get all source identifiers"""
        return [doc.get('metadata', {}).get('source', doc['id']) for doc in self.documents]
    
    def search_with_persona(
        self,
        query: str,
        persona_weights,
        query_transformation,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Persona-aware hybrid search using persona's search weight profile
        
        Args:
            query: Original query
            persona_weights: SearchWeightProfile from persona
            query_transformation: QueryTransformation with expanded query and sub-queries
            top_k: Number of results
        
        Returns:
            Persona-optimized search results
        """
        if not self.documents:
            return []
        
        # Use persona's preferred BM25/semantic balance
        weights = {
            'bm25': persona_weights.bm25_weight,
            'semantic': persona_weights.semantic_weight
        }
        
        # Search main query
        main_results = self._execute_weighted_search(
            query_transformation.expanded_query,
            weights,
            top_k * 2
        )
        
        # Merge results dict: doc_id -> score
        all_results = {doc_id: score for doc_id, score in main_results}
        
        # Search sub-queries for comprehensive strategy
        if query_transformation.search_strategy == 'comprehensive':
            for sub_query in query_transformation.sub_queries:
                sub_results = self._execute_weighted_search(sub_query, weights, top_k)
                
                # Merge results with decay factor
                for doc_id, score in sub_results:
                    if doc_id in all_results:
                        all_results[doc_id] = max(all_results[doc_id], score * 0.7)
                    else:
                        all_results[doc_id] = score * 0.7
        
        # Apply persona-specific adjustments
        adjusted_results = self._apply_persona_adjustments(all_results, persona_weights)
        
        # Convert to SearchResult objects
        sorted_results = sorted(adjusted_results.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (doc_id, score) in enumerate(sorted_results[:top_k], 1):
            doc = next((d for d in self.documents if d['id'] == doc_id), None)
            if doc:
                results.append(SearchResult(
                    chunk_id=doc_id,
                    content=doc['content'],
                    score=score,
                    source='persona_hybrid',
                    metadata=doc.get('metadata', {}),
                    rank=rank
                ))
        
        return results
    
    def _execute_weighted_search(
        self,
        query: str,
        weights: Dict[str, float],
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Execute single search with given weights"""
        bm25_results = self._bm25_search(query, top_k)
        semantic_results = self._semantic_search(query, top_k)
        
        return self._reciprocal_rank_fusion(
            bm25_results=bm25_results,
            semantic_results=semantic_results,
            weights=weights,
            k=60
        )
    
    def _apply_persona_adjustments(
        self,
        results: Dict[str, float],
        persona_weights
    ) -> Dict[str, float]:
        """
        Apply persona-specific result adjustments:
        - Recency boost
        - Diversity preference
        - Citation importance
        """
        adjusted = {}
        
        for doc_id, score in results.items():
            doc = next((d for d in self.documents if d['id'] == doc_id), None)
            if not doc:
                continue
                
            metadata = doc.get('metadata', {})
            adjusted_score = score
            
            # Recency boost (if document has timestamp)
            if 'timestamp' in metadata and persona_weights.recency_boost > 0:
                age_factor = metadata.get('age_days', 365) / 365
                recency_multiplier = 1 + (persona_weights.recency_boost * (1 - min(age_factor, 1)))
                adjusted_score *= recency_multiplier
            
            # Citation importance (if document has citation count)
            if 'citations' in metadata and persona_weights.citation_importance > 0:
                citation_count = metadata['citations']
                citation_multiplier = 1 + (persona_weights.citation_importance * (citation_count / 100))
                adjusted_score *= min(citation_multiplier, 2.0)
            
            adjusted[doc_id] = adjusted_score
        
        return adjusted

