"""
Bias Detector Node

Detects publication bias in meta-analysis.
"""

from typing import Dict, List, Any, Optional
from loguru import logger

from src.meta_analysis.mcp_servers.base_experiment_mcp import StandardizedStudy
from src.meta_analysis.statistical.publication_bias import PublicationBiasDetector
from src.meta_analysis.statistical.simpsons_detector import SimpsonsParadoxDetector
from src.meta_analysis.statistical.meta_analyzer import MetaAnalysisResult


def detect_bias_node(
    studies: List[StandardizedStudy],
    meta_result: Optional[MetaAnalysisResult] = None,
    check_simpsons: bool = True,
    segment_key: str = "segment"
) -> Dict[str, Any]:
    """
    Detect publication bias and Simpson's paradox.
    
    Args:
        studies: List of standardized studies
        meta_result: Pre-computed meta-analysis result
        check_simpsons: Whether to check for Simpson's paradox
        segment_key: Key for subgroup segmentation
        
    Returns:
        Bias detection results
    """
    result = {
        "publication_bias": None,
        "simpsons_paradox": None,
        "errors": []
    }
    
    # Publication bias detection
    if len(studies) >= 3:
        try:
            detector = PublicationBiasDetector()
            bias_result = detector.analyze(studies, meta_result)
            result["publication_bias"] = bias_result.to_dict()
            logger.info(f"Publication bias: {bias_result.bias_severity}")
        except Exception as e:
            result["errors"].append(f"Publication bias detection failed: {e}")
            logger.warning(f"Publication bias detection failed: {e}")
    else:
        result["errors"].append("Too few studies for publication bias analysis")
    
    # Simpson's paradox detection
    if check_simpsons and len(studies) >= 4:
        try:
            simpsons_detector = SimpsonsParadoxDetector()
            simpsons_result = simpsons_detector.detect_simpsons_paradox(
                studies, 
                segment_key=segment_key
            )
            result["simpsons_paradox"] = simpsons_result.to_dict()
            logger.info(f"Simpson's paradox: {simpsons_result.paradox_detected}")
        except Exception as e:
            result["errors"].append(f"Simpson's paradox detection failed: {e}")
            logger.warning(f"Simpson's paradox detection failed: {e}")
    
    return result
