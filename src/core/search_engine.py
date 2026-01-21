"""
[ELITE ARCHITECTURE] search_engine.py
FAISS-based Hybrid Search Engine.
"""

import faiss
import numpy as np
from typing import List, Dict, Any
from loguru import logger

class SearchEngine:
    """
    Innovation: Low-Latency Persistence.
    Implements FAISS for O(1) retrieval and metadata persistence.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = [] # Metadata + Content storage
        self.embeddings = [] # Raw vectors for visualization

    def add_document(self, content: str, metadata: Dict[str, Any], embedding: np.ndarray):
        """Adds a chunk to the vector database."""
        if embedding.shape[0] != self.dimension:
            logger.error(f"Dimension mismatch: expected {self.dimension}, got {embedding.shape[0]}")
            return
            
        self.index.add(np.array([embedding]).astype('float32'))
        self.documents.append({"content": content, "metadata": metadata})
        self.embeddings.append(embedding)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Any]:
        """Performs vector search."""
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), k)
        
        results = []
        for idx in I[0]:
            if idx != -1 and idx < len(self.documents):
                # Format to match expectations of RAG agents
                doc = type('Document', (), self.documents[idx])
                results.append(doc)
        return results

    def save(self, directory: str):
        """Saves index and metadata to directory."""
        import pickle
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        
        # Save metadata
        metadata = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "dimension": self.dimension
        }
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"Search index saved to {directory}")

    def load(self, directory: str):
        """Loads index and metadata from directory."""
        import pickle
        import os
        
        index_path = os.path.join(directory, "index.faiss")
        meta_path = os.path.join(directory, "metadata.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            logger.warning(f"Persistence files not found in {directory}")
            return False
            
        # Load FAISS
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
            self.documents = metadata.get("documents", [])
            self.embeddings = metadata.get("embeddings", [])
            self.dimension = metadata.get("dimension", 384)
            
        logger.info(f"Search index loaded from {directory} (Docs: {len(self.documents)})")
        return True

    def get_all_embeddings(self) -> np.ndarray:
        return np.array(self.embeddings)

    def get_all_texts(self) -> List[str]:
        return [d['content'] for d in self.documents]

    def get_all_sources(self) -> List[str]:
        return [d['metadata'].get('source_file', 'unknown') for d in self.documents]

if __name__ == "__main__":
    se = SearchEngine()
    print("Search Engine Standby.")
