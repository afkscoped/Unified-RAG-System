"""
A/B Test Meta-Analysis Engine

Feature 3: Comprehensive meta-analysis system that aggregates A/B test data,
performs statistical analysis, and provides RAG-powered insights.

Components:
- mcp_servers: Data ingestion from CSV/Excel and experiment platforms
- statistical: Core statistical engine (meta-analysis, bias detection)
- rag: RAG integration for insights and recommendations
- orchestration: LangGraph workflow for analysis pipeline
- integration: Bridges to existing RAG system and LLM router
"""

from src.meta_analysis.mcp_servers.base_experiment_mcp import (
    StandardizedStudy,
    ExperimentMCPBase,
    ValidationReport
)

__all__ = [
    "StandardizedStudy",
    "ExperimentMCPBase", 
    "ValidationReport"
]
