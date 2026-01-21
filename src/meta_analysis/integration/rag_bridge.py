"""
RAG Bridge

Connects meta-analysis engine to existing UnifiedRAGSystem.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger


class MetaAnalysisRAGBridge:
    """
    Bridges meta-analysis to the existing RAG infrastructure.
    
    Allows meta-analysis to leverage existing embeddings and retrieval
    while maintaining specialized collections for methodology docs.
    """
    
    def __init__(
        self,
        rag_system=None,
        methodology_docs_path: str = "./data/methodology_papers",
        experiment_archive_path: str = "./data/experiment_archive"
    ):
        """
        Initialize RAG bridge.
        
        Args:
            rag_system: Existing UnifiedRAGSystem instance (optional)
            methodology_docs_path: Path to methodology documents
            experiment_archive_path: Path to experiment archive
        """
        self.rag_system = rag_system
        self.methodology_docs_path = Path(methodology_docs_path)
        self.experiment_archive_path = Path(experiment_archive_path)
        self._initialized = False
        
        # Ensure directories exist
        self.methodology_docs_path.mkdir(parents=True, exist_ok=True)
        self.experiment_archive_path.mkdir(parents=True, exist_ok=True)
    
    def initialize(self):
        """Initialize the bridge with RAG system."""
        if self.rag_system is None:
            logger.warning("No RAG system provided - operating in standalone mode")
            self._initialized = True
            return
        
        # Index methodology papers if available
        self._index_methodology_docs()
        self._initialized = True
        logger.info("MetaAnalysisRAGBridge initialized")
    
    def _index_methodology_docs(self):
        """Index methodology documents into RAG system."""
        if not self.methodology_docs_path.exists():
            logger.info("No methodology docs directory found")
            return
        
        # Look for supported file types
        doc_files = list(self.methodology_docs_path.glob("*.pdf"))
        doc_files.extend(self.methodology_docs_path.glob("*.txt"))
        doc_files.extend(self.methodology_docs_path.glob("*.md"))
        
        if doc_files and self.rag_system:
            try:
                for doc_file in doc_files:
                    self.rag_system.ingest_file(str(doc_file))
                logger.info(f"Indexed {len(doc_files)} methodology documents")
            except Exception as e:
                logger.warning(f"Could not index methodology docs: {e}")
    
    def query_methodology(
        self,
        question: str,
        top_k: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Query methodology knowledge base.
        
        Args:
            question: Statistical/methodological question
            top_k: Number of sources to retrieve
            
        Returns:
            Query result with answer and sources
        """
        if not self.rag_system:
            logger.debug("No RAG system available for query")
            return None
        
        try:
            result = self.rag_system.query(
                question=question,
                top_k=top_k,
                use_cache=True
            )
            return {
                "answer": result.answer,
                "sources": [
                    {"content": s.content, "score": s.score}
                    for s in result.sources
                ],
                "cached": result.cached
            }
        except Exception as e:
            logger.error(f"Methodology query failed: {e}")
            return None
    
    def get_embedding_model(self):
        """Get the embedding model from RAG system."""
        if self.rag_system:
            return getattr(self.rag_system, 'embeddings', None)
        return None
    
    def is_available(self) -> bool:
        """Check if RAG bridge is available."""
        return self._initialized and self.rag_system is not None
