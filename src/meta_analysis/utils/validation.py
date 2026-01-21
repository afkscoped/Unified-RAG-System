"""
Data Validation Utilities

Functions for validating A/B test experiment data quality.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from src.meta_analysis.mcp_servers.base_experiment_mcp import StandardizedStudy


def validate_effect_size(
    effect_size: float,
    max_abs_value: float = 5.0
) -> Tuple[bool, Optional[str]]:
    """
    Validate that effect size is within reasonable bounds.
    
    Args:
        effect_size: Effect size to validate
        max_abs_value: Maximum absolute value allowed
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not np.isfinite(effect_size):
        return False, "Effect size is not finite (NaN or Inf)"
    
    if abs(effect_size) > max_abs_value:
        return False, f"Effect size {effect_size:.3f} exceeds maximum |{max_abs_value}|"
    
    return True, None


def validate_sample_size(
    sample_size: int,
    min_size: int = 10
) -> Tuple[bool, Optional[str]]:
    """
    Validate sample size.
    
    Args:
        sample_size: Sample size to validate
        min_size: Minimum required sample size
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if sample_size < 0:
        return False, "Sample size cannot be negative"
    
    if sample_size < min_size:
        return False, f"Sample size {sample_size} is below minimum {min_size}"
    
    return True, None


def validate_standard_error(
    standard_error: float,
    min_value: float = 0.0001,
    max_value: float = 10.0
) -> Tuple[bool, Optional[str]]:
    """
    Validate standard error.
    
    Args:
        standard_error: Standard error to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not np.isfinite(standard_error):
        return False, "Standard error is not finite (NaN or Inf)"
    
    if standard_error <= min_value:
        return False, f"Standard error {standard_error:.6f} is too small (min: {min_value})"
    
    if standard_error > max_value:
        return False, f"Standard error {standard_error:.3f} exceeds maximum {max_value}"
    
    return True, None


def validate_study_data(
    study: StandardizedStudy,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive validation of a StandardizedStudy.
    
    Args:
        study: Study to validate
        strict: If True, treat warnings as errors
        
    Returns:
        Dictionary with validation results:
        - is_valid: Overall validity
        - errors: List of error messages
        - warnings: List of warning messages
    """
    errors = []
    warnings = []
    
    # Validate effect size
    valid, msg = validate_effect_size(study.effect_size)
    if not valid:
        errors.append(f"Effect size: {msg}")
    elif abs(study.effect_size) > 2.0:
        warnings.append(f"Large effect size: {study.effect_size:.3f}")
    
    # Validate standard error
    valid, msg = validate_standard_error(study.standard_error)
    if not valid:
        errors.append(f"Standard error: {msg}")
    
    # Validate sample sizes
    valid, msg = validate_sample_size(study.sample_size_control)
    if not valid:
        errors.append(f"Control sample: {msg}")
    elif study.sample_size_control < 30:
        warnings.append(f"Small control sample: {study.sample_size_control}")
    
    valid, msg = validate_sample_size(study.sample_size_treatment)
    if not valid:
        errors.append(f"Treatment sample: {msg}")
    elif study.sample_size_treatment < 30:
        warnings.append(f"Small treatment sample: {study.sample_size_treatment}")
    
    # Validate confidence interval consistency
    if study.confidence_interval_lower is not None and study.confidence_interval_upper is not None:
        if study.confidence_interval_lower > study.confidence_interval_upper:
            errors.append("CI lower bound exceeds upper bound")
        
        # Check if effect size is within CI
        if not (study.confidence_interval_lower <= study.effect_size <= study.confidence_interval_upper):
            warnings.append("Effect size not within confidence interval")
    
    # Validate p-value if present
    if study.p_value is not None:
        if study.p_value < 0 or study.p_value > 1:
            errors.append(f"Invalid p-value: {study.p_value}")
    
    # Check for study ID uniqueness issues (just a warning about format)
    if not study.study_id or len(study.study_id) < 1:
        errors.append("Missing or empty study ID")
    
    is_valid = len(errors) == 0
    if strict:
        is_valid = is_valid and len(warnings) == 0
    
    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "study_id": study.study_id
    }


def validate_study_collection(
    studies: List[StandardizedStudy],
    min_studies: int = 3,
    check_heterogeneity: bool = True
) -> Dict[str, Any]:
    """
    Validate a collection of studies for meta-analysis.
    
    Args:
        studies: List of studies to validate
        min_studies: Minimum number of studies required
        check_heterogeneity: Whether to check for effect size heterogeneity
        
    Returns:
        Validation results dictionary
    """
    result = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "study_results": [],
        "summary": {}
    }
    
    # Check minimum count
    if len(studies) < min_studies:
        result["errors"].append(f"Need at least {min_studies} studies, got {len(studies)}")
        result["is_valid"] = False
    
    # Validate each study
    valid_studies = []
    for study in studies:
        validation = validate_study_data(study)
        result["study_results"].append(validation)
        if validation["is_valid"]:
            valid_studies.append(study)
        else:
            result["warnings"].extend([f"{study.study_id}: {e}" for e in validation["errors"]])
    
    # Summary statistics
    if valid_studies:
        effect_sizes = [s.effect_size for s in valid_studies]
        result["summary"] = {
            "total_studies": len(studies),
            "valid_studies": len(valid_studies),
            "mean_effect": float(np.mean(effect_sizes)),
            "std_effect": float(np.std(effect_sizes)),
            "min_effect": float(np.min(effect_sizes)),
            "max_effect": float(np.max(effect_sizes)),
            "total_sample_size": sum(s.total_sample_size for s in valid_studies)
        }
        
        # Heterogeneity warning
        if check_heterogeneity:
            effect_range = result["summary"]["max_effect"] - result["summary"]["min_effect"]
            if effect_range > 2.0:
                result["warnings"].append(
                    f"High heterogeneity: effect sizes range from "
                    f"{result['summary']['min_effect']:.3f} to {result['summary']['max_effect']:.3f}"
                )
    
    return result


def detect_outliers(
    studies: List[StandardizedStudy],
    method: str = "iqr",
    threshold: float = 1.5
) -> List[str]:
    """
    Detect outlier studies based on effect sizes.
    
    Args:
        studies: List of studies to check
        method: Detection method ("iqr" or "zscore")
        threshold: Threshold for outlier detection
        
    Returns:
        List of study IDs identified as outliers
    """
    if len(studies) < 3:
        return []
    
    effect_sizes = np.array([s.effect_size for s in studies])
    outlier_ids = []
    
    if method == "iqr":
        q1, q3 = np.percentile(effect_sizes, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        for study in studies:
            if study.effect_size < lower_bound or study.effect_size > upper_bound:
                outlier_ids.append(study.study_id)
    
    elif method == "zscore":
        mean = np.mean(effect_sizes)
        std = np.std(effect_sizes)
        if std > 0:
            for study in studies:
                z_score = abs(study.effect_size - mean) / std
                if z_score > threshold:
                    outlier_ids.append(study.study_id)
    
    return outlier_ids
