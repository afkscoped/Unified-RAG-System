"""
[ELITE ARCHITECTURE] semantic_chunker.py
Embedding-drift splitting for research-grade context windows.
"""

import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

class SemanticChunker:
    """
    Splits text based on semantic transitions.
    Innovation: Instead of fixed sizes, we measure the cosine distance between 
    sequential sentences. A new chunk is created when the 'semantic drift' 
    exceeds a percentile threshold.
    """
    
    def __init__(self, embedding_manager, threshold: float = 0.85):
        self.encoder = embedding_manager
        self.threshold = threshold

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into semantically unique segments.
        """
        if not text.strip(): return []
        
        # 1. Decompose into sentences (using simple heuristic for speed/low-VRAM)
        sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
        if len(sentences) <= 1: return [text]
        
        # 2. Vectorize all sentences
        embeddings = self.encoder.encode(sentences)
        
        # 3. Compute semantic transitions
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            similarities.append(sim)
        
        # 4. Identification of Breakpoints
        chunks = []
        current_chunk = [sentences[0]]
        
        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                # Boundary detected
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk = [sentences[i+1]]
            else:
                current_chunk.append(sentences[i+1])
                
        # Final append
        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")
            
        logger.debug(f"Semantic Chunking: {len(sentences)} sentences -> {len(chunks)} chunks.")
        return chunks

if __name__ == "__main__":
    print("Semantic Chunker Ready.")
