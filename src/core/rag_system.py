"""
Unified RAG System

Main orchestrator that combines all components:
- Document ingestion and chunking
- Hybrid search (BM25 + Semantic)
- Semantic caching
- Adaptive weight management
- LLM response generation
"""

import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import yaml
from dataclasses import dataclass, field
from loguru import logger

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    UnstructuredWordDocumentLoader
)

from src.core.embeddings import EmbeddingManager
from src.core.hybrid_search import HybridSearchEngine, SearchResult
from src.core.weight_manager import AdaptiveWeightManager
from src.cache.semantic_cache import SemanticCache
from src.llm.llm_router import LLMRouter
from src.utils.memory_monitor import MemoryMonitor


@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    # Model settings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm_provider: str = "groq"
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.2
    
    # Device settings
    device: str = "cuda"
    use_fp16: bool = True
    
    # Search settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    rrf_k: int = 60
    
    # Cache settings
    cache_enabled: bool = True
    cache_threshold: float = 0.95
    cache_max_items: int = 100
    
    # Memory settings
    max_ram_gb: float = 12.0
    max_vram_gb: float = 4.8
    embedding_batch_size: int = 32
    
    @classmethod
    def from_yaml(cls, path: str) -> "RAGConfig":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            embedding_model=data.get('model', {}).get('embedding', cls.embedding_model),
            llm_provider=data.get('model', {}).get('llm_provider', cls.llm_provider),
            llm_model=data.get('model', {}).get('llm_model', cls.llm_model),
            llm_temperature=data.get('model', {}).get('llm_temperature', cls.llm_temperature),
            device=data.get('model', {}).get('device', cls.device),
            use_fp16=data.get('model', {}).get('use_fp16', cls.use_fp16),
            chunk_size=data.get('search', {}).get('chunk_size', cls.chunk_size),
            chunk_overlap=data.get('search', {}).get('chunk_overlap', cls.chunk_overlap),
            top_k=data.get('search', {}).get('top_k', cls.top_k),
            rrf_k=data.get('search', {}).get('rrf_k', cls.rrf_k),
            cache_enabled=data.get('cache', {}).get('enabled', cls.cache_enabled),
            cache_threshold=data.get('cache', {}).get('similarity_threshold', cls.cache_threshold),
            cache_max_items=data.get('cache', {}).get('max_items', cls.cache_max_items),
            max_ram_gb=data.get('memory', {}).get('max_ram_gb', cls.max_ram_gb),
            max_vram_gb=data.get('memory', {}).get('max_vram_gb', cls.max_vram_gb),
            embedding_batch_size=data.get('memory', {}).get('embedding_batch_size', cls.embedding_batch_size),
        )


@dataclass
class QueryResult:
    """Result from a RAG query."""
    answer: str
    sources: List[SearchResult] = field(default_factory=list)
    cached: bool = False
    query_type: str = "mixed"
    semantic_weight: float = 0.5
    lexical_weight: float = 0.5


class UnifiedRAGSystem:
    """
    Main RAG system orchestrator.
    
    Combines document ingestion, hybrid search, caching,
    and LLM generation into a unified interface.
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer questions based on the provided context.
If the context doesn't contain relevant information, say so honestly.
Be concise and accurate in your responses."""
    
    def __init__(self, config: Union[str, RAGConfig, None] = None):
        """
        Initialize RAG system.
        
        Args:
            config: Path to config.yaml, RAGConfig object, or None for defaults
        """
        # Load config
        if isinstance(config, str):
            self.config = RAGConfig.from_yaml(config)
        elif isinstance(config, RAGConfig):
            self.config = config
        else:
            self.config = RAGConfig()
            
        logger.info("Initializing Unified RAG System...")
        
        # Initialize components
        self.memory_monitor = MemoryMonitor(
            max_ram_gb=self.config.max_ram_gb,
            max_vram_gb=self.config.max_vram_gb
        )
        
        self.embedding_manager = EmbeddingManager(
            model_name=self.config.embedding_model,
            device=self.config.device,
            use_fp16=self.config.use_fp16,
            batch_size=self.config.embedding_batch_size,
            max_vram_gb=self.config.max_vram_gb
        )
        
        self.search_engine = HybridSearchEngine(
            embedding_manager=self.embedding_manager,
            rrf_k=self.config.rrf_k
        )
        
        self.weight_manager = AdaptiveWeightManager()
        
        self.semantic_cache = SemanticCache(
            similarity_threshold=self.config.cache_threshold,
            max_items=self.config.cache_max_items
        ) if self.config.cache_enabled else None
        
        self.llm_router = LLMRouter(
            primary_provider=self.config.llm_provider,
            groq_model=self.config.llm_model,
            temperature=self.config.llm_temperature
        )
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # State
        self.documents: List[Document] = []
        self._indexed = False
        
        # Metrics
        self.metrics = {
            "queries": 0,
            "cache_hits": 0,
            "documents_ingested": 0,
            "chunks_created": 0
        }
        
        logger.info("RAG System initialized")
        self.memory_monitor.log_status()
        
    def ingest_file(self, file_path: str) -> int:
        """
        Ingest a single file.
        
        Args:
            file_path: Path to file (PDF, TXT, DOCX)
            
        Returns:
            Number of chunks created
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        
        logger.info(f"Ingesting: {path.name}")
        
        # Select loader
        if ext == ".pdf":
            loader = PyPDFLoader(str(path))
        elif ext == ".txt":
            loader = TextLoader(str(path))
        elif ext in [".docx", ".doc"]:
            loader = UnstructuredWordDocumentLoader(str(path))
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Load and split
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        
        # Add metadata
        for chunk in chunks:
            chunk.metadata["source_file"] = path.name
            
        self.documents.extend(chunks)
        self.metrics["documents_ingested"] += 1
        self.metrics["chunks_created"] += len(chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {path.name}")
        return len(chunks)
    
    def ingest_directory(
        self,
        dir_path: str,
        glob_pattern: str = "**/*.*",
        extensions: Optional[List[str]] = None
    ) -> int:
        """
        Ingest all files from a directory.
        
        Args:
            dir_path: Path to directory
            glob_pattern: Glob pattern for files
            extensions: List of extensions to include (e.g., [".pdf", ".txt"])
            
        Returns:
            Total chunks created
        """
        extensions = extensions or [".pdf", ".txt", ".docx"]
        path = Path(dir_path)
        total_chunks = 0
        
        for file_path in path.glob(glob_pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    chunks = self.ingest_file(str(file_path))
                    total_chunks += chunks
                except Exception as e:
                    logger.error(f"Failed to ingest {file_path.name}: {e}")
                    
        return total_chunks
    
    async def ingest_files_async(
        self,
        file_paths: List[str]
    ) -> int:
        """Async batch ingestion."""
        total_chunks = 0
        for path in file_paths:
            chunks = await asyncio.to_thread(self.ingest_file, path)
            total_chunks += chunks
            self.memory_monitor.check_and_cleanup()
        return total_chunks
    
    def build_index(self):
        """Build search indices from ingested documents."""
        if not self.documents:
            raise RuntimeError("No documents ingested. Call ingest_file first.")
            
        logger.info(f"Building index for {len(self.documents)} chunks...")
        self.search_engine.build_indices(self.documents)
        self._indexed = True
        
        self.memory_monitor.check_and_cleanup()
        logger.info("Index built successfully")
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_cache: bool = True
    ) -> QueryResult:
        """
        Query the RAG system.
        
        Args:
            question: User question
            top_k: Number of sources to retrieve
            use_cache: Whether to use semantic cache
            
        Returns:
            QueryResult with answer and sources
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call build_index first.")
            
        self.metrics["queries"] += 1
        k = top_k or self.config.top_k
        
        # Encode query
        query_embedding = self.embedding_manager.encode_single(question)
        
        # Check cache
        if use_cache and self.semantic_cache:
            cached_response = self.semantic_cache.get(question, query_embedding)
            if cached_response:
                self.metrics["cache_hits"] += 1
                logger.debug("Cache hit")
                return QueryResult(
                    answer=cached_response,
                    cached=True
                )
        
        # Get adaptive weights
        sem_weight, lex_weight = self.weight_manager.get_weights(question)
        query_type = self.weight_manager.classify_query(question)
        
        # Hybrid search
        sources = self.search_engine.search_hybrid(
            query=question,
            k=k,
            semantic_weight=sem_weight,
            lexical_weight=lex_weight
        )
        
        # Build context
        context = "\n\n---\n\n".join([
            f"Source {i+1}:\n{src.content}"
            for i, src in enumerate(sources)
        ])
        
        # Generate response
        prompt = f"""Context:
{context}

---

Question: {question}

Please provide a clear and accurate answer based on the context above."""

        answer = self.llm_router.generate(
            prompt=prompt,
            system_prompt=self.DEFAULT_SYSTEM_PROMPT
        )
        
        # Cache response
        if self.semantic_cache:
            self.semantic_cache.set(question, query_embedding, answer)
        
        # Periodic memory cleanup
        if self.metrics["queries"] % 10 == 0:
            self.memory_monitor.check_and_cleanup()
        
        return QueryResult(
            answer=answer,
            sources=sources,
            cached=False,
            query_type=query_type,
            semantic_weight=sem_weight,
            lexical_weight=lex_weight
        )
    
    async def aquery(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_cache: bool = True
    ) -> QueryResult:
        """Async version of query."""
        return await asyncio.to_thread(
            self.query, question, top_k, use_cache
        )
    
    def submit_feedback(self, question: str, rating: int):
        """
        Submit feedback for a query.
        
        Args:
            question: The original question
            rating: Rating 1-5
        """
        sem_weight, _ = self.weight_manager.get_weights(question)
        self.weight_manager.record_feedback(question, float(rating), sem_weight)
        logger.debug(f"Feedback recorded: rating={rating}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        cache_stats = self.semantic_cache.get_stats() if self.semantic_cache else {}
        
        return {
            **self.metrics,
            "cache_hit_rate": cache_stats.get("hit_rate", 0),
            "cache_size": cache_stats.get("size", 0),
            "weight_stats": self.weight_manager.get_stats(),
            "llm_stats": self.llm_router.get_stats(),
            "memory": self.memory_monitor.get_status()
        }
    
    def save_index(self, path: str):
        """Save FAISS index to disk."""
        self.search_engine.save_faiss(path)
        
    def load_index(self, path: str):
        """Load FAISS index from disk."""
        self.search_engine.load_faiss(path)
        self._indexed = True

