"""
[ELITE ARCHITECTURE] cross_encoder_reranker.py
Surgical precision reranking for the final candidate pool.
"""

import torch
from sentence_transformers import CrossEncoder
from typing import List, Any
from loguru import logger

class CrossEncoderReranker:
    """
    Innovation: Cross-Encoders are significantly more accurate than bi-encoders (vectors)
    because they process the query and document simultaneously with cross-attention.
    Optimized: Moves model back and forth to CPU to preserve VRAM for LLM.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        if self.model is None:
            logger.info(f"Dynamically Loading Reranker: {self.model_name}")
            self.model = CrossEncoder(self.model_name, device=self.device)

    def _unload_model(self):
        """Releases VRAM/RAM when not in active use."""
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def rerank(self, query: str, candidates: List[Any], top_n: int = 5) -> List[Any]:
        """
        Re-scores and sorts candidates.
        """
        if not candidates: return []
        
        self._load_model()
        
        # Prepare pairs [Query, Content]
        pairs = [[query, c.content] for c in candidates]
        
        logger.debug(f"Cross-Attention Reranking {len(pairs)} pairs...")
        scores = self.model.predict(pairs)
        
        # Merge scores and sort
        for i, candidate in enumerate(candidates):
            candidate.score = float(scores[i])
            
        ranked = sorted(candidates, key=lambda x: x.score, reverse=True)[:top_n]
        
        # Optimization: Cleanup immediately
        self._unload_model()
        
        return ranked

if __name__ == "__main__":
    print("Reranker Engine Ready.")
