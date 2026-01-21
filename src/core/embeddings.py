"""
GPU-Optimized Embedding Manager

Handles embedding generation with:
- fp16 precision for memory efficiency
- Batch processing with VRAM limits
- Auto garbage collection
"""

import torch
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from loguru import logger

from src.utils.memory_monitor import MemoryMonitor


class LangChainEmbeddingsWrapper(Embeddings):
    """Wrapper to make EmbeddingManager compatible with LangChain."""
    
    def __init__(self, embedding_manager: "EmbeddingManager"):
        self.embedding_manager = embedding_manager
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.embedding_manager.encode(texts, normalize=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.embedding_manager.encode_single(text, normalize=True)
        return embedding.tolist()


class EmbeddingManager:
    """Manages GPU-optimized embeddings with memory safety."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None,
        use_fp16: bool = True,
        batch_size: int = 32,
        max_vram_gb: float = 4.8
    ):
        """
        Initialize embedding manager.
        
        Args:
            model_name: HuggingFace model identifier
            device: cuda or cpu (auto-detect if None)
            use_fp16: Use half precision for GPU
            batch_size: Max batch size for encoding
            max_vram_gb: Maximum VRAM to use
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        
        # Auto-detect device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available or Torch not compiled with CUDA. Falling back to CPU.")
            self.device = "cpu"
        elif device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Memory monitor
        self.memory_monitor = MemoryMonitor(max_vram_gb=max_vram_gb)
        
        # Load model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        if self.device == "cuda":
            self.model = self.model.to(self.device)
            if use_fp16:
                self.model = self.model.half()
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(max_vram_gb / 6.0)  # Assuming 6GB GPU
            
        logger.info(f"Embedding model loaded on {self.device} (fp16={use_fp16})")
        
    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings with batching.
        
        Args:
            texts: List of texts to encode
            normalize: Normalize embeddings for cosine similarity
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        embeddings = self.model.encode(
                            batch,
                            convert_to_numpy=True,
                            normalize_embeddings=normalize,
                            show_progress_bar=show_progress and i == 0
                        )
                else:
                    embeddings = self.model.encode(
                        batch,
                        convert_to_numpy=True,
                        normalize_embeddings=normalize,
                        show_progress_bar=show_progress and i == 0
                    )
                    
            all_embeddings.append(embeddings)
            
            # Periodic cleanup
            if i % (self.batch_size * 4) == 0 and i > 0:
                self.memory_monitor.check_and_cleanup()
                
        return np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text."""
        return self.encode([text], normalize=normalize)[0]
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts (alias for encode).
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        return self.encode(texts, normalize=True)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text (alias for encode_single)."""
        return self.encode_single(text, normalize=True)

    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    @property
    def as_langchain(self) -> LangChainEmbeddingsWrapper:
        """Get LangChain-compatible embeddings wrapper."""
        return LangChainEmbeddingsWrapper(self)
    
    def cleanup(self):
        """Force GPU memory cleanup."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")


