"""
Semantic Cache

Caches LLM responses based on query similarity
to avoid redundant API calls.
"""

import time
from typing import Optional, Dict, Any
from collections import OrderedDict
import numpy as np
from loguru import logger


class SemanticCache:
    """
    Similarity-based cache for RAG responses.
    
    Uses embedding similarity to find cached responses
    for semantically similar queries.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        max_items: int = 100,
        ttl_seconds: int = 3600
    ):
        """
        Initialize semantic cache.
        
        Args:
            similarity_threshold: Min cosine similarity to match (0-1)
            max_items: Maximum cache entries (LRU eviction)
            ttl_seconds: Time-to-live for entries
        """
        self.similarity_threshold = similarity_threshold
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        
        # Cache: query -> (embedding, response, timestamp)
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        
        # Stats
        self._hits = 0
        self._misses = 0
        
    def get(
        self,
        query: str,
        query_embedding: np.ndarray
    ) -> Optional[str]:
        """
        Get cached response if similar query exists.
        
        Args:
            query: The query string
            query_embedding: Query embedding vector
            
        Returns:
            Cached response if found, None otherwise
        """
        current_time = time.time()
        
        # Check for expired entries and similar queries
        keys_to_remove = []
        best_match = None
        best_similarity = 0.0
        
        for cached_query, entry in self._cache.items():
            # Check TTL
            if current_time - entry["timestamp"] > self.ttl_seconds:
                keys_to_remove.append(cached_query)
                continue
                
            # Calculate similarity
            cached_embedding = entry["embedding"]
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            
            if similarity >= self.similarity_threshold and similarity > best_similarity:
                best_match = entry["response"]
                best_similarity = similarity
                
                # Move to end (most recently used)
                self._cache.move_to_end(cached_query)
        
        # Remove expired entries
        for key in keys_to_remove:
            del self._cache[key]
        
        if best_match:
            self._hits += 1
            logger.debug(f"Cache hit (similarity={best_similarity:.3f})")
            return best_match
        
        self._misses += 1
        return None
    
    def set(
        self,
        query: str,
        query_embedding: np.ndarray,
        response: str
    ):
        """
        Cache a query-response pair.
        
        Args:
            query: The query string
            query_embedding: Query embedding vector
            response: The response to cache
        """
        # LRU eviction if at capacity
        while len(self._cache) >= self.max_items:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
            logger.debug(f"Cache evicted oldest entry")
        
        self._cache[query] = {
            "embedding": query_embedding,
            "response": response,
            "timestamp": time.time()
        }
        
        logger.debug(f"Cached response (total={len(self._cache)})")
    
    def _cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_items,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate
        }
    
    @property
    def hit_rate(self) -> float:
        """Get current cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

