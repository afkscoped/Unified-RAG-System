"""
Integration Layer

Bridges meta-analysis engine to existing system components:
- RAGBridge: Connect to UnifiedRAGSystem
- LLMBridge: Connect to LLMRouter
"""

from src.meta_analysis.integration.rag_bridge import MetaAnalysisRAGBridge
from src.meta_analysis.integration.llm_bridge import MetaAnalysisLLMBridge

__all__ = [
    "MetaAnalysisRAGBridge",
    "MetaAnalysisLLMBridge"
]
