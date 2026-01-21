"""
Data Collector Node

Collects experiments from various MCP sources.
"""

from typing import Dict, List, Any, Optional
from loguru import logger

from src.meta_analysis.mcp_servers.base_experiment_mcp import (
    StandardizedStudy,
    ExperimentFilter
)


def collect_data_node(
    sources: List[str],
    filters: Optional[Dict] = None,
    csv_mcp=None,
    platform_mcps: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Collect experiment data from MCP sources.
    
    Args:
        sources: List of source types to query (e.g., ["csv", "optimizely"])
        filters: Filters to apply
        csv_mcp: CSV MCP server instance
        platform_mcps: Dict of platform MCP instances
        
    Returns:
        Collection results with studies and metadata
    """
    result = {
        "studies": [],
        "sources_queried": [],
        "errors": [],
        "metadata": {}
    }
    
    # Build filter object
    exp_filter = None
    if filters:
        from datetime import datetime
        exp_filter = ExperimentFilter(
            min_sample_size=filters.get("min_sample_size"),
            search_query=filters.get("search_query")
        )
        if filters.get("year_from"):
            exp_filter.date_from = datetime(filters["year_from"], 1, 1)
    
    # Collect from CSV source
    if "csv" in sources and csv_mcp:
        try:
            studies = csv_mcp.get_experiments(filters=exp_filter)
            result["studies"].extend(studies)
            result["sources_queried"].append("csv")
            result["metadata"]["csv_count"] = len(studies)
            logger.info(f"Collected {len(studies)} studies from CSV")
        except Exception as e:
            result["errors"].append(f"CSV collection failed: {e}")
            logger.warning(f"CSV collection failed: {e}")
    
    # Collect from platform MCPs (future)
    if platform_mcps:
        for platform, mcp in platform_mcps.items():
            if platform in sources:
                try:
                    studies = mcp.get_experiments(filters=exp_filter)
                    result["studies"].extend(studies)
                    result["sources_queried"].append(platform)
                    result["metadata"][f"{platform}_count"] = len(studies)
                except Exception as e:
                    result["errors"].append(f"{platform} collection failed: {e}")
    
    result["total_studies"] = len(result["studies"])
    return result
