"""
Utility Functions for Meta-Analysis

Helper functions for:
- Effect size conversions
- Data validation
- Synthetic data generation
- Kaggle dataset import
"""

from src.meta_analysis.utils.effect_size_converters import (
    cohens_d_to_log_odds,
    log_odds_to_cohens_d,
    risk_ratio_to_log_odds,
    calculate_standard_error
)
from src.meta_analysis.utils.validation import (
    validate_effect_size,
    validate_sample_size,
    validate_study_data
)
from src.meta_analysis.utils.synthetic_generator import (
    SyntheticABTestGenerator,
    DOMAIN_CONFIGS
)
from src.meta_analysis.utils.kaggle_importer import (
    KaggleABTestImporter,
    AB_COLUMN_PATTERNS
)

__all__ = [
    "cohens_d_to_log_odds",
    "log_odds_to_cohens_d",
    "risk_ratio_to_log_odds",
    "calculate_standard_error",
    "validate_effect_size",
    "validate_sample_size",
    "validate_study_data",
    "SyntheticABTestGenerator",
    "DOMAIN_CONFIGS",
    "KaggleABTestImporter",
    "AB_COLUMN_PATTERNS"
]
