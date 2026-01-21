"""
[ELITE ARCHITECTURE] embedding_manager.py
Manages Sentence-Transformer models with VRAM safety.
"""

from sentence_transformers import SentenceTransformer
import torch
from loguru import logger
import numpy as np

class EmbeddingManager:
    """
    Innovation: Precision Control.
    Handles embedding generation with support for GPU/CPU toggling 
    to free VRAM when the LLM is active.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = None):
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU for Elite Embeddings.")
            self.device = "cpu"
        elif device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Loading Embedding Model: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def get_embeddings(self, texts: list) -> np.ndarray:
        """Generates vectors for a list of strings."""
        return self.model.encode(texts, convert_to_numpy=True)

    def to_cpu(self):
        """Moves model to RAM to free VRAM."""
        self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to_gpu(self):
        """Moves model back to GPU."""
        if torch.cuda.is_available():
            self.model.to("cuda")

if __name__ == "__main__":
    em = EmbeddingManager()
    print("Embedding Manager Standby.")
