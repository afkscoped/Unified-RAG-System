"""
Synthetic A/B Test Data Generator Module

Provides CTGAN and CopulaGAN based synthetic data generation
across 5 domains: E-Commerce, SaaS, Healthcare, Marketing, EdTech.
"""

from src.meta_analysis.synthetic.generators.base_generator import (
    BaseGenerator,
    SyntheticConfig,
    ValidationReport
)
from src.meta_analysis.synthetic.domains.domain_constraints import (
    DomainConstraints,
    DOMAIN_CONFIGS
)

__all__ = [
    'BaseGenerator',
    'SyntheticConfig', 
    'ValidationReport',
    'DomainConstraints',
    'DOMAIN_CONFIGS'
]
