"""
Hybrid Search Engine

Combines BM25 lexical search with FAISS semantic search
using Reciprocal Rank Fusion (RRF).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from loguru import logger

from src.core.embeddings import EmbeddingManager


@dataclass
class SearchResult:
    """Container for search results."""
    content: str
    score: float
    metadata: Dict
    source: str  # "semantic", "lexical", or "hybrid"


class HybridSearchEngine:
    """
    Hybrid search combining BM25 and semantic search.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results
    from both search methods.
    """
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        rrf_k: int = 60,
        default_semantic_weight: float = 0.5,
        default_lexical_weight: float = 0.5
    ):
        """
        Initialize hybrid search engine.
        
        Args:
            embedding_manager: Embedding manager for semantic search
            rrf_k: RRF constant (higher = less weight to top ranks)
            default_semantic_weight: Default weight for semantic results
            default_lexical_weight: Default weight for lexical results
        """
        self.embedding_manager = embedding_manager
        self.rrf_k = rrf_k
        self.default_semantic_weight = default_semantic_weight
        self.default_lexical_weight = default_lexical_weight
        
        # Indices
        self.documents: List[Document] = []
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_store: Optional[FAISS] = None
        
        self._indexed = False
        
    def build_indices(self, documents: List[Document]):
        """
        Build both BM25 and FAISS indices.
        
        Args:
            documents: List of LangChain documents to index
        """
        self.documents = documents
        logger.info(f"Building indices for {len(documents)} documents...")
        
        # Build BM25
        tokenized = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built")
        
        # Build FAISS
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_manager.encode(texts, show_progress=True)
        
        # Create FAISS store with embeddings
        text_embedding_pairs = list(zip(texts, embeddings.tolist()))
        metadatas = [doc.metadata for doc in documents]
        
        self.faiss_store = FAISS.from_embeddings(
            text_embedding_pairs,
            self.embedding_manager.as_langchain,
            metadatas=metadatas
        )
        logger.info("FAISS index built")
        
        self._indexed = True
        
    def search_semantic(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        Semantic search using FAISS.
        
        Returns list of (document, score) tuples.
        """
        if not self.faiss_store:
            raise RuntimeError("FAISS index not built. Call build_indices first.")
            
        results = self.faiss_store.similarity_search_with_score(query, k=k)
        return results
    
    def search_lexical(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        Lexical search using BM25.
        
        Returns list of (document, score) tuples.
        """
        if not self.bm25:
            raise RuntimeError("BM25 index not built. Call build_indices first.")
            
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include matches
                results.append((self.documents[idx], float(scores[idx])))
                
        return results
    
    def search_hybrid(
        self,
        query: str,
        k: int = 5,
        semantic_weight: Optional[float] = None,
        lexical_weight: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Hybrid search using RRF fusion.
        
        Args:
            query: Search query
            k: Number of results to return
            semantic_weight: Weight for semantic results (0-1)
            lexical_weight: Weight for lexical results (0-1)
            
        Returns:
            List of SearchResult objects
        """
        if not self._indexed:
            raise RuntimeError("Indices not built. Call build_indices first.")
            
        sem_w = semantic_weight or self.default_semantic_weight
        lex_w = lexical_weight or self.default_lexical_weight
        
        # Normalize weights
        total = sem_w + lex_w
        sem_w /= total
        lex_w /= total
        
        # Get results from both methods
        fetch_k = k * 3  # Fetch more for fusion
        
        semantic_results = self.search_semantic(query, k=fetch_k)
        lexical_results = self.search_lexical(query, k=fetch_k)
        
        # RRF fusion
        combined_scores: Dict[str, float] = {}
        doc_map: Dict[str, Tuple[str, Dict]] = {}  # content_key -> (full_content, metadata)
        
        # Process semantic results
        for rank, (doc, score) in enumerate(semantic_results):
            key = doc.page_content[:100]  # Use prefix as key
            rrf_score = sem_w / (self.rrf_k + rank + 1)
            combined_scores[key] = combined_scores.get(key, 0) + rrf_score
            doc_map[key] = (doc.page_content, doc.metadata)
            
        # Process lexical results
        for rank, (doc, score) in enumerate(lexical_results):
            key = doc.page_content[:100]
            rrf_score = lex_w / (self.rrf_k + rank + 1)
            combined_scores[key] = combined_scores.get(key, 0) + rrf_score
            if key not in doc_map:
                doc_map[key] = (doc.page_content, doc.metadata)
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        # Build result objects
        results = []
        for key, score in sorted_results:
            content, metadata = doc_map[key]
            results.append(SearchResult(
                content=content,
                score=score,
                metadata=metadata,
                source="hybrid"
            ))
            
        logger.debug(f"Hybrid search returned {len(results)} results")
        return results
    
    def save_faiss(self, path: str):
        """Save FAISS index to disk."""
        if self.faiss_store:
            self.faiss_store.save_local(path)
            logger.info(f"FAISS index saved to {path}")
            
    def load_faiss(self, path: str):
        """Load FAISS index from disk."""
        self.faiss_store = FAISS.load_local(
            path,
            self.embedding_manager.as_langchain,
            allow_dangerous_deserialization=True
        )
        logger.info(f"FAISS index loaded from {path}")


