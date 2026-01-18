"""
Core RAG System Tests

Tests for the main RAG components:
- Embedding manager
- Hybrid search
- Weight manager
- Semantic cache
- Full RAG system
"""

import pytest
import numpy as np


class TestEmbeddingManager:
    """Tests for EmbeddingManager."""
    
    def test_import(self):
        """Test that EmbeddingManager can be imported."""
        from src.core.embeddings import EmbeddingManager
        assert EmbeddingManager is not None
    
    def test_initialization_cpu(self):
        """Test initialization on CPU."""
        from src.core.embeddings import EmbeddingManager
        
        manager = EmbeddingManager(
            model_name="BAAI/bge-small-en-v1.5",
            device="cpu",
            use_fp16=False,
            batch_size=8
        )
        
        assert manager is not None
        assert manager.device == "cpu"
        assert manager.embedding_dim > 0
    
    def test_encode_single(self):
        """Test encoding a single text."""
        from src.core.embeddings import EmbeddingManager
        
        manager = EmbeddingManager(device="cpu", use_fp16=False, batch_size=8)
        
        embedding = manager.encode_single("This is a test sentence.")
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == manager.embedding_dim
    
    def test_encode_batch(self):
        """Test batch encoding."""
        from src.core.embeddings import EmbeddingManager
        
        manager = EmbeddingManager(device="cpu", use_fp16=False, batch_size=8)
        
        texts = [
            "First sentence",
            "Second sentence",
            "Third sentence"
        ]
        
        embeddings = manager.encode(texts)
        
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == manager.embedding_dim


class TestWeightManager:
    """Tests for AdaptiveWeightManager."""
    
    def test_import(self):
        """Test import."""
        from src.core.weight_manager import AdaptiveWeightManager
        assert AdaptiveWeightManager is not None
    
    def test_classify_keyword(self):
        """Test keyword query classification."""
        from src.core.weight_manager import AdaptiveWeightManager
        
        manager = AdaptiveWeightManager()
        
        assert manager.classify_query("python API") == "keyword"
        assert manager.classify_query("best LLM") == "keyword"
    
    def test_classify_conceptual(self):
        """Test conceptual query classification."""
        from src.core.weight_manager import AdaptiveWeightManager
        
        manager = AdaptiveWeightManager()
        
        assert manager.classify_query("explain how neural networks work") == "conceptual"
        assert manager.classify_query("what is the meaning of life") == "conceptual"
    
    def test_get_weights(self):
        """Test weight retrieval."""
        from src.core.weight_manager import AdaptiveWeightManager
        
        manager = AdaptiveWeightManager()
        
        sem, lex = manager.get_weights("python API")
        
        assert 0 <= sem <= 1
        assert 0 <= lex <= 1
        assert abs(sem + lex - 1.0) < 0.01  # Should sum to ~1


class TestSemanticCache:
    """Tests for SemanticCache."""
    
    def test_import(self):
        """Test import."""
        from src.cache.semantic_cache import SemanticCache
        assert SemanticCache is not None
    
    def test_set_get_exact(self):
        """Test exact cache hit."""
        from src.cache.semantic_cache import SemanticCache
        
        cache = SemanticCache(similarity_threshold=0.95)
        
        query = "What is Python?"
        embedding = np.random.randn(384).astype(np.float32)
        response = "Python is a programming language."
        
        cache.set(query, embedding, response)
        
        # Same embedding should hit
        result = cache.get(query, embedding)
        assert result == response
    
    def test_cache_miss(self):
        """Test cache miss with different embedding."""
        from src.cache.semantic_cache import SemanticCache
        
        cache = SemanticCache(similarity_threshold=0.99)
        
        embedding1 = np.random.randn(384).astype(np.float32)
        embedding2 = np.random.randn(384).astype(np.float32)
        
        cache.set("query1", embedding1, "response1")
        
        # Very different embedding should miss
        result = cache.get("query2", embedding2)
        assert result is None
    
    def test_hit_rate(self):
        """Test hit rate calculation."""
        from src.cache.semantic_cache import SemanticCache
        
        cache = SemanticCache()
        
        embedding = np.random.randn(384).astype(np.float32)
        cache.set("query", embedding, "response")
        
        # 1 hit
        cache.get("query", embedding)
        
        # 1 miss
        cache.get("other", np.random.randn(384).astype(np.float32))
        
        assert cache.hit_rate == 0.5


class TestHybridSearch:
    """Tests for HybridSearchEngine."""
    
    def test_import(self):
        """Test import."""
        from src.core.hybrid_search import HybridSearchEngine, SearchResult
        assert HybridSearchEngine is not None
        assert SearchResult is not None


class TestMemoryMonitor:
    """Tests for MemoryMonitor."""
    
    def test_import(self):
        """Test import."""
        from src.utils.memory_monitor import MemoryMonitor
        assert MemoryMonitor is not None
    
    def test_ram_usage(self):
        """Test RAM usage retrieval."""
        from src.utils.memory_monitor import MemoryMonitor
        
        monitor = MemoryMonitor()
        ram = monitor.get_ram_usage()
        
        assert "total_gb" in ram
        assert "used_gb" in ram
        assert "percent" in ram
        assert ram["total_gb"] > 0
    
    def test_status(self):
        """Test full status."""
        from src.utils.memory_monitor import MemoryMonitor
        
        monitor = MemoryMonitor()
        status = monitor.get_status()
        
        assert "ram" in status
        assert "cleanup_count" in status

