"""
MCP Server Layer for Experiment Data Ingestion

Provides data ingestion capabilities for A/B test experiments from:
- CSV/Excel files
- Future: Optimizely, LaunchDarkly, GrowthBook APIs
"""

from src.meta_analysis.mcp_servers.base_experiment_mcp import (
    StandardizedStudy,
    ExperimentMCPBase,
    ValidationReport
)
from src.meta_analysis.mcp_servers.csv_experiment_server import CSVExperimentMCP

__all__ = [
    "StandardizedStudy",
    "ExperimentMCPBase",
    "ValidationReport",
    "CSVExperimentMCP"
]
