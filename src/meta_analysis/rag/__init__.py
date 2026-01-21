"""
RAG Integration for Meta-Analysis

Provides RAG-powered insights and recommendations:
- MetaAnalysisRAG: Specialized RAG for statistical interpretation
- MethodologyIndexer: Index academic papers on meta-analysis
- ExperimentArchive: Searchable experiment database
- BenchmarkRetriever: Industry comparisons
"""

from src.meta_analysis.rag.meta_analysis_rag import MetaAnalysisRAG

__all__ = [
    "MetaAnalysisRAG"
]
