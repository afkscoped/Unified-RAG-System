"""
Semantic Query Cache - Reduces redundant LLM calls
Uses cosine similarity to match similar queries
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from datetime import datetime
import hashlib
from sentence_transformers import SentenceTransformer
import torch
from diskcache import Cache
from loguru import logger


@dataclass
class CacheEntry:
    """Single cache entry"""
    query: str
    query_embedding: np.ndarray
    response: str
    sources: List[Dict]
    timestamp: datetime
    hit_count: int = 0


class SemanticCache:
    """
    Intelligent semantic cache with similarity-based retrieval
    Persists to disk for cross-session caching
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.95,
        cache_dir: str = "./cache/semantic_cache",
        max_entries: int = 1000
    ):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        
        # Initialize embedder (lazy load to avoid memory issues)
        self._embedder = None
        self._embedding_model = embedding_model
        
        # Disk cache for persistence
        try:
            self.disk_cache = Cache(cache_dir, size_limit=int(1e9))  # 1GB limit
        except Exception as e:
            logger.warning(f"Could not create disk cache: {e}")
            self.disk_cache = None
        
        # In-memory cache for fast access
        self.memory_cache: List[CacheEntry] = []
        
        # Load from disk
        self._load_from_disk()
    
    @property
    def embedder(self):
        """Lazy load embedder"""
        if self._embedder is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._embedder = SentenceTransformer(self._embedding_model, device=device)
        return self._embedder
    
    def _load_from_disk(self) -> None:
        """Load cached entries from disk"""
        if not self.disk_cache:
            return
            
        try:
            count = 0
            for key in list(self.disk_cache):
                if key.startswith("cache_entry_"):
                    data = self.disk_cache.get(key)
                    if data:
                        entry = CacheEntry(
                            query=data['query'],
                            query_embedding=np.array(data['query_embedding']),
                            response=data['response'],
                            sources=data['sources'],
                            timestamp=datetime.fromisoformat(data['timestamp']),
                            hit_count=data.get('hit_count', 0)
                        )
                        self.memory_cache.append(entry)
                        count += 1
            
            if count > 0:
                logger.info(f"Loaded {count} cached entries from disk")
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
    
    def _save_to_disk(self, entry: CacheEntry) -> None:
        """Persist entry to disk"""
        if not self.disk_cache:
            return
            
        entry_id = hashlib.md5(entry.query.encode()).hexdigest()
        key = f"cache_entry_{entry_id}"
        
        data = {
            'query': entry.query,
            'query_embedding': entry.query_embedding.tolist(),
            'response': entry.response,
            'sources': entry.sources,
            'timestamp': entry.timestamp.isoformat(),
            'hit_count': entry.hit_count
        }
        
        try:
            self.disk_cache[key] = data
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    def get(self, query: str) -> Optional[Tuple[str, List[Dict], float]]:
        """
        Retrieve cached response if similar query exists
        
        Returns:
            (response, sources, similarity_score) or None
        """
        if not self.memory_cache:
            return None
        
        # Encode query
        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        # Find most similar cached query
        best_match = None
        best_similarity = -1.0
        
        for entry in self.memory_cache:
            similarity = self._cosine_similarity(
                query_embedding,
                entry.query_embedding
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry
        
        # Return if above threshold
        if best_similarity >= self.similarity_threshold and best_match:
            best_match.hit_count += 1
            self._save_to_disk(best_match)
            
            return (best_match.response, best_match.sources, best_similarity)
        
        return None
    
    def set(
        self,
        query: str,
        response: str,
        sources: List[Dict]
    ) -> None:
        """Add new entry to cache"""
        # Encode query
        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        # Create entry
        entry = CacheEntry(
            query=query,
            query_embedding=query_embedding,
            response=response,
            sources=sources,
            timestamp=datetime.now()
        )
        
        # Add to memory cache
        self.memory_cache.append(entry)
        
        # Enforce max entries (LRU eviction)
        if len(self.memory_cache) > self.max_entries:
            self.memory_cache.sort(key=lambda x: x.timestamp)
            removed = self.memory_cache.pop(0)
            
            if self.disk_cache:
                entry_id = hashlib.md5(removed.query.encode()).hexdigest()
                self.disk_cache.pop(f"cache_entry_{entry_id}", None)
        
        # Save to disk
        self._save_to_disk(entry)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    
    def clear(self) -> None:
        """Clear entire cache"""
        self.memory_cache.clear()
        if self.disk_cache:
            self.disk_cache.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_hits = sum(entry.hit_count for entry in self.memory_cache)
        
        disk_size = 0
        if self.disk_cache:
            try:
                disk_size = self.disk_cache.volume() / 1e6
            except:
                pass
        
        return {
            'total_entries': len(self.memory_cache),
            'total_hits': total_hits,
            'cache_hit_rate': total_hits / max(len(self.memory_cache), 1),
            'disk_size_mb': disk_size
        }
