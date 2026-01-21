"""
Data Harmonizer

Standardizes effect sizes across different sources and formats.
Handles conversions between different effect size measures.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np
from loguru import logger

from src.meta_analysis.mcp_servers.base_experiment_mcp import (
    StandardizedStudy,
    EffectSizeType
)
from src.meta_analysis.utils.effect_size_converters import (
    cohens_d_to_log_odds,
    log_odds_to_cohens_d,
    risk_ratio_to_log_odds,
    calculate_standard_error
)


class DataHarmonizer:
    """
    Harmonizes A/B test data from different sources.
    
    Converts all effect sizes to a common measure (log odds ratio by default)
    and ensures consistent metadata structure.
    """
    
    def __init__(
        self,
        target_effect_type: EffectSizeType = EffectSizeType.LOG_ODDS_RATIO,
        default_baseline_risk: float = 0.1
    ):
        """
        Initialize the harmonizer.
        
        Args:
            target_effect_type: Effect size type to standardize to
            default_baseline_risk: Default baseline risk for RR conversions
        """
        self.target_effect_type = target_effect_type
        self.default_baseline_risk = default_baseline_risk
    
    def harmonize_studies(
        self,
        studies: List[StandardizedStudy],
        validate: bool = True
    ) -> List[StandardizedStudy]:
        """
        Harmonize a list of studies to a common effect size format.
        
        Args:
            studies: List of studies to harmonize
            validate: Whether to validate studies after harmonization
            
        Returns:
            List of harmonized studies
        """
        harmonized = []
        
        for study in studies:
            try:
                harmonized_study = self.harmonize_single(study)
                if harmonized_study:
                    harmonized.append(harmonized_study)
            except Exception as e:
                logger.warning(f"Failed to harmonize study {study.study_id}: {e}")
        
        logger.info(f"Harmonized {len(harmonized)}/{len(studies)} studies")
        return harmonized
    
    def harmonize_single(
        self,
        study: StandardizedStudy
    ) -> Optional[StandardizedStudy]:
        """
        Harmonize a single study.
        
        Args:
            study: Study to harmonize
            
        Returns:
            Harmonized study or None if conversion fails
        """
        if study.effect_size_type == self.target_effect_type:
            # Already in target format
            return study
        
        # Convert effect size
        new_effect_size, new_se = self._convert_effect_size(
            study.effect_size,
            study.standard_error,
            study.effect_size_type,
            study.total_sample_size
        )
        
        if new_effect_size is None:
            return None
        
        # Create new harmonized study
        return StandardizedStudy(
            study_id=study.study_id,
            study_name=study.study_name,
            effect_size=new_effect_size,
            standard_error=new_se,
            sample_size_control=study.sample_size_control,
            sample_size_treatment=study.sample_size_treatment,
            metric_name=study.metric_name,
            metric_type=study.metric_type,
            effect_size_type=self.target_effect_type,
            platform=study.platform,
            timestamp=study.timestamp,
            p_value=study.p_value,
            segments=study.segments,
            metadata={
                **(study.metadata or {}),
                "original_effect_type": study.effect_size_type.value,
                "original_effect_size": study.effect_size,
                "original_se": study.standard_error
            }
        )
    
    def _convert_effect_size(
        self,
        effect_size: float,
        standard_error: float,
        from_type: EffectSizeType,
        sample_size: int
    ) -> tuple:
        """Convert effect size to target type."""
        
        if self.target_effect_type == EffectSizeType.LOG_ODDS_RATIO:
            if from_type == EffectSizeType.COHENS_D:
                new_effect = cohens_d_to_log_odds(effect_size)
                # Approximate SE conversion
                new_se = standard_error * (np.pi / np.sqrt(3))
                return new_effect, new_se
            
            elif from_type == EffectSizeType.RISK_RATIO:
                try:
                    new_effect = risk_ratio_to_log_odds(
                        effect_size, 
                        self.default_baseline_risk
                    )
                    # Approximate SE
                    new_se = calculate_standard_error(new_effect, sample_size, "log_odds")
                    return new_effect, new_se
                except ValueError as e:
                    logger.warning(f"Could not convert risk ratio: {e}")
                    return None, None
        
        elif self.target_effect_type == EffectSizeType.COHENS_D:
            if from_type == EffectSizeType.LOG_ODDS_RATIO:
                new_effect = log_odds_to_cohens_d(effect_size)
                new_se = standard_error * (np.sqrt(3) / np.pi)
                return new_effect, new_se
        
        logger.warning(f"Unsupported conversion: {from_type} -> {self.target_effect_type}")
        return None, None
    
    def standardize_metadata(
        self,
        studies: List[StandardizedStudy]
    ) -> List[StandardizedStudy]:
        """
        Ensure all studies have consistent metadata structure.
        
        Args:
            studies: List of studies
            
        Returns:
            Studies with standardized metadata
        """
        standardized = []
        
        for study in studies:
            metadata = study.metadata or {}
            
            # Ensure common fields exist
            standardized_metadata = {
                "source_platform": metadata.get("source_platform", study.platform),
                "harmonized": True,
                "harmonization_date": str(np.datetime64('now')),
                **metadata
            }
            
            # Create new study with standardized metadata
            new_study = StandardizedStudy(
                study_id=study.study_id,
                study_name=study.study_name,
                effect_size=study.effect_size,
                standard_error=study.standard_error,
                sample_size_control=study.sample_size_control,
                sample_size_treatment=study.sample_size_treatment,
                metric_name=study.metric_name,
                metric_type=study.metric_type,
                effect_size_type=study.effect_size_type,
                platform=study.platform,
                timestamp=study.timestamp,
                confidence_interval_lower=study.confidence_interval_lower,
                confidence_interval_upper=study.confidence_interval_upper,
                p_value=study.p_value,
                segments=study.segments,
                metadata=standardized_metadata
            )
            standardized.append(new_study)
        
        return standardized
    
    def compute_summary_stats(
        self,
        studies: List[StandardizedStudy]
    ) -> Dict[str, Any]:
        """
        Compute summary statistics for a collection of studies.
        
        Args:
            studies: List of harmonized studies
            
        Returns:
            Dictionary of summary statistics
        """
        if not studies:
            return {}
        
        effect_sizes = np.array([s.effect_size for s in studies])
        standard_errors = np.array([s.standard_error for s in studies])
        sample_sizes = np.array([s.total_sample_size for s in studies])
        
        return {
            "n_studies": len(studies),
            "effect_size": {
                "mean": float(np.mean(effect_sizes)),
                "median": float(np.median(effect_sizes)),
                "std": float(np.std(effect_sizes)),
                "min": float(np.min(effect_sizes)),
                "max": float(np.max(effect_sizes)),
                "range": float(np.max(effect_sizes) - np.min(effect_sizes))
            },
            "standard_error": {
                "mean": float(np.mean(standard_errors)),
                "median": float(np.median(standard_errors)),
                "min": float(np.min(standard_errors)),
                "max": float(np.max(standard_errors))
            },
            "sample_size": {
                "total": int(np.sum(sample_sizes)),
                "mean": float(np.mean(sample_sizes)),
                "median": float(np.median(sample_sizes)),
                "min": int(np.min(sample_sizes)),
                "max": int(np.max(sample_sizes))
            },
            "effect_type": self.target_effect_type.value
        }
