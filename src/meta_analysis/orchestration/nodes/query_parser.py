"""
Query Parser Node

Parses user queries into structured analysis requests.
"""

from typing import Dict, Any, Optional
import re


def parse_query_node(query: str, llm_bridge=None) -> Dict[str, Any]:
    """
    Parse user query into structured format.
    
    Args:
        query: User's natural language query
        llm_bridge: Optional LLM bridge for enhanced parsing
        
    Returns:
        Parsed intent dictionary
    """
    query_lower = query.lower()
    
    result = {
        "raw_query": query,
        "analysis_types": [],
        "model_preference": "auto",
        "filters": {},
        "specific_requests": []
    }
    
    # Detect analysis types
    if any(term in query_lower for term in ["meta-analysis", "meta analysis", "pool", "combine"]):
        result["analysis_types"].append("meta_analysis")
    
    if any(term in query_lower for term in ["bias", "funnel", "egger"]):
        result["analysis_types"].append("publication_bias")
    
    if any(term in query_lower for term in ["sensitivity", "robust", "leave-one-out"]):
        result["analysis_types"].append("sensitivity")
    
    if any(term in query_lower for term in ["subgroup", "segment", "stratif", "simpson"]):
        result["analysis_types"].append("subgroup")
    
    # Default to full analysis if nothing specific
    if not result["analysis_types"]:
        result["analysis_types"] = ["meta_analysis", "publication_bias", "sensitivity"]
    
    # Detect model preference
    if "fixed effect" in query_lower:
        result["model_preference"] = "fixed"
    elif "random effect" in query_lower:
        result["model_preference"] = "random"
    
    # Extract filters
    # Sample size
    sample_match = re.search(r'(?:sample|n)\s*(?:size)?[:\s>]+(\d+)', query_lower)
    if sample_match:
        result["filters"]["min_sample_size"] = int(sample_match.group(1))
    
    # Date filters
    year_match = re.search(r'(?:from|since|after)\s+(\d{4})', query_lower)
    if year_match:
        result["filters"]["year_from"] = int(year_match.group(1))
    
    # Metric type
    if "conversion" in query_lower:
        result["filters"]["metric_type"] = "conversion"
    elif "revenue" in query_lower:
        result["filters"]["metric_type"] = "revenue"
    elif "retention" in query_lower:
        result["filters"]["metric_type"] = "retention"
    
    # Specific requests
    if "explain" in query_lower or "interpret" in query_lower:
        result["specific_requests"].append("interpretation")
    if "recommend" in query_lower:
        result["specific_requests"].append("recommendations")
    if "compare" in query_lower or "benchmark" in query_lower:
        result["specific_requests"].append("benchmarking")
    
    return result
