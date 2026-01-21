"""
Analyzer Node

Runs meta-analysis on collected studies.
"""

from typing import Dict, List, Any, Optional
from loguru import logger

from src.meta_analysis.mcp_servers.base_experiment_mcp import StandardizedStudy
from src.meta_analysis.statistical.meta_analyzer import MetaAnalyzer


def analyze_node(
    studies: List[StandardizedStudy],
    model: str = "auto",
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Run meta-analysis on studies.
    
    Args:
        studies: List of standardized studies
        model: Model type ("fixed", "random", "auto")
        confidence_level: Confidence level for intervals
        
    Returns:
        Analysis results dictionary
    """
    result = {
        "success": False,
        "meta_result": None,
        "comparison": None,
        "error": None
    }
    
    if len(studies) < 2:
        result["error"] = "Need at least 2 studies for meta-analysis"
        return result
    
    try:
        analyzer = MetaAnalyzer(confidence_level=confidence_level)
        
        # Run main analysis
        meta_result = analyzer.analyze(studies, model=model)
        result["meta_result"] = meta_result.to_dict()
        result["success"] = True
        
        # Also run model comparison
        comparison = analyzer.compare_models(studies)
        result["comparison"] = comparison
        
        # Calculate prediction interval
        pred_interval = analyzer.prediction_interval(meta_result)
        result["prediction_interval"] = {
            "lower": pred_interval[0],
            "upper": pred_interval[1]
        }
        
        logger.info(
            f"Analysis complete: effect={meta_result.pooled_effect:.4f}, "
            f"I²={meta_result.heterogeneity['I2']:.1f}%"
        )
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Analysis failed: {e}")
    
    return result
