"""
Statistical Engine for Meta-Analysis

Core statistical components:
- DataHarmonizer: Effect size conversions and standardization
- MetaAnalyzer: Fixed and random effects models
- PublicationBiasDetector: Egger's test, trim-and-fill
- SimpsonsParadoxDetector: Paradox detection and stratified analysis
- SensitivityAnalyzer: Leave-one-out, influence diagnostics
- Visualization: Forest plots, funnel plots
"""

from src.meta_analysis.statistical.data_harmonizer import DataHarmonizer
from src.meta_analysis.statistical.meta_analyzer import MetaAnalyzer
from src.meta_analysis.statistical.publication_bias import PublicationBiasDetector
from src.meta_analysis.statistical.simpsons_detector import SimpsonsParadoxDetector
from src.meta_analysis.statistical.sensitivity_analysis import SensitivityAnalyzer
from src.meta_analysis.statistical.visualization import MetaAnalysisVisualizer

__all__ = [
    "DataHarmonizer",
    "MetaAnalyzer",
    "PublicationBiasDetector",
    "SimpsonsParadoxDetector",
    "SensitivityAnalyzer",
    "MetaAnalysisVisualizer"
]
