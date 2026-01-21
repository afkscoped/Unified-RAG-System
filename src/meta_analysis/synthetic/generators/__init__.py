"""
Synthetic Data Generators

CTGAN and CopulaGAN implementations for generating realistic A/B test data.
"""

from src.meta_analysis.synthetic.generators.base_generator import (
    BaseGenerator,
    SyntheticConfig,
    ValidationReport
)
from src.meta_analysis.synthetic.generators.ctgan_generator import CTGANGenerator
from src.meta_analysis.synthetic.generators.copulagan_generator import CopulaGANGenerator

__all__ = [
    'BaseGenerator',
    'SyntheticConfig',
    'ValidationReport',
    'CTGANGenerator',
    'CopulaGANGenerator'
]
