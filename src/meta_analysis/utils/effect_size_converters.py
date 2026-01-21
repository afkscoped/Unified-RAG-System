"""
Effect Size Converters

Utility functions for converting between different effect size measures
and calculating standard errors.

Supported conversions:
- Cohen's d ↔ Log odds ratio
- Risk ratio ↔ Log odds ratio
- Various standard error calculations
"""

import math
from typing import Tuple, Optional
import numpy as np


def calculate_log_odds_ratio(
    control_success: int,
    control_failure: int,
    treatment_success: int,
    treatment_failure: int,
    continuity_correction: float = 0.5
) -> float:
    """
    Calculate the log odds ratio from 2x2 contingency table.
    
    Args:
        control_success: Successes in control group
        control_failure: Failures in control group
        treatment_success: Successes in treatment group
        treatment_failure: Failures in treatment group
        continuity_correction: Value to add when zeros present (default 0.5)
        
    Returns:
        Log odds ratio
    """
    # Apply continuity correction if any cell is zero
    if 0 in [control_success, control_failure, treatment_success, treatment_failure]:
        control_success += continuity_correction
        control_failure += continuity_correction
        treatment_success += continuity_correction
        treatment_failure += continuity_correction
    
    # Odds for each group
    odds_control = control_success / control_failure
    odds_treatment = treatment_success / treatment_failure
    
    # Odds ratio and log transformation
    odds_ratio = odds_treatment / odds_control
    return math.log(odds_ratio)


def calculate_standard_error_log_odds(
    control_success: int,
    control_failure: int,
    treatment_success: int,
    treatment_failure: int,
    continuity_correction: float = 0.5
) -> float:
    """
    Calculate standard error of log odds ratio.
    
    Uses the formula: SE = sqrt(1/a + 1/b + 1/c + 1/d)
    where a,b,c,d are the four cells of the 2x2 table.
    
    Args:
        control_success: Successes in control group
        control_failure: Failures in control group
        treatment_success: Successes in treatment group
        treatment_failure: Failures in treatment group
        continuity_correction: Value to add when zeros present
        
    Returns:
        Standard error of log odds ratio
    """
    # Apply continuity correction if any cell is zero
    if 0 in [control_success, control_failure, treatment_success, treatment_failure]:
        control_success += continuity_correction
        control_failure += continuity_correction
        treatment_success += continuity_correction
        treatment_failure += continuity_correction
    
    variance = (
        1/control_success + 1/control_failure + 
        1/treatment_success + 1/treatment_failure
    )
    return math.sqrt(variance)


def cohens_d_to_log_odds(d: float) -> float:
    """
    Convert Cohen's d to log odds ratio.
    
    Uses the approximation: log(OR) ≈ d * π / sqrt(3)
    
    Args:
        d: Cohen's d effect size
        
    Returns:
        Log odds ratio
    """
    return d * math.pi / math.sqrt(3)


def log_odds_to_cohens_d(log_or: float) -> float:
    """
    Convert log odds ratio to Cohen's d.
    
    Uses the approximation: d ≈ log(OR) * sqrt(3) / π
    
    Args:
        log_or: Log odds ratio
        
    Returns:
        Cohen's d effect size
    """
    return log_or * math.sqrt(3) / math.pi


def risk_ratio_to_log_odds(
    risk_ratio: float,
    baseline_risk: float
) -> float:
    """
    Convert risk ratio to log odds ratio.
    
    Args:
        risk_ratio: Risk ratio (RR)
        baseline_risk: Baseline risk in control group (0-1)
        
    Returns:
        Log odds ratio
    """
    if baseline_risk <= 0 or baseline_risk >= 1:
        raise ValueError("Baseline risk must be between 0 and 1")
    
    # Calculate odds ratio from risk ratio
    # OR = RR * (1 - p0) / (1 - p0 * RR)
    # where p0 is baseline risk
    treatment_risk = baseline_risk * risk_ratio
    
    if treatment_risk >= 1:
        raise ValueError("Treatment risk would exceed 1")
    
    odds_control = baseline_risk / (1 - baseline_risk)
    odds_treatment = treatment_risk / (1 - treatment_risk)
    
    return math.log(odds_treatment / odds_control)


def log_odds_to_risk_ratio(
    log_or: float,
    baseline_risk: float
) -> float:
    """
    Convert log odds ratio to risk ratio.
    
    Args:
        log_or: Log odds ratio
        baseline_risk: Baseline risk in control group (0-1)
        
    Returns:
        Risk ratio
    """
    if baseline_risk <= 0 or baseline_risk >= 1:
        raise ValueError("Baseline risk must be between 0 and 1")
    
    odds_ratio = math.exp(log_or)
    
    # RR = OR / (1 - p0 + p0 * OR)
    risk_ratio = odds_ratio / (1 - baseline_risk + baseline_risk * odds_ratio)
    return risk_ratio


def calculate_standard_error(
    effect_size: float,
    sample_size: int,
    effect_type: str = "log_odds"
) -> float:
    """
    Estimate standard error from effect size and sample size.
    
    This is an approximation used when direct calculation is not possible.
    
    Args:
        effect_size: Effect size value
        sample_size: Total sample size
        effect_type: Type of effect size ("log_odds", "cohens_d", "correlation")
        
    Returns:
        Estimated standard error
    """
    if effect_type == "log_odds":
        # Approximate SE for log odds ratio
        # SE ≈ sqrt(4/n) for equal-sized groups
        return math.sqrt(4 / sample_size)
    
    elif effect_type == "cohens_d":
        # SE for Cohen's d
        # SE ≈ sqrt(4/n + d²/2n)
        return math.sqrt(4/sample_size + (effect_size**2) / (2*sample_size))
    
    elif effect_type == "correlation":
        # SE for Fisher's z (transformed correlation)
        # SE = 1/sqrt(n-3)
        if sample_size <= 3:
            return float('inf')
        return 1 / math.sqrt(sample_size - 3)
    
    else:
        # Generic approximation
        return math.sqrt(4 / sample_size)


def convert_effect_size(
    value: float,
    from_type: str,
    to_type: str,
    baseline_risk: Optional[float] = None,
    sample_size: Optional[int] = None
) -> Tuple[float, float]:
    """
    Convert between effect size types.
    
    Args:
        value: Effect size value
        from_type: Source effect size type
        to_type: Target effect size type
        baseline_risk: Required for risk ratio conversions
        sample_size: Required for SE estimation
        
    Returns:
        Tuple of (converted effect size, estimated standard error)
    """
    # First convert to log odds if needed
    if from_type == "log_odds":
        log_or = value
    elif from_type == "cohens_d":
        log_or = cohens_d_to_log_odds(value)
    elif from_type == "risk_ratio" and baseline_risk:
        log_or = risk_ratio_to_log_odds(value, baseline_risk)
    else:
        raise ValueError(f"Cannot convert from {from_type}")
    
    # Then convert to target type
    if to_type == "log_odds":
        result = log_or
    elif to_type == "cohens_d":
        result = log_odds_to_cohens_d(log_or)
    elif to_type == "risk_ratio" and baseline_risk:
        result = log_odds_to_risk_ratio(log_or, baseline_risk)
    else:
        raise ValueError(f"Cannot convert to {to_type}")
    
    # Estimate SE
    se = calculate_standard_error(result, sample_size or 100, to_type)
    
    return result, se


def pooled_standard_deviation(
    n1: int, sd1: float,
    n2: int, sd2: float
) -> float:
    """
    Calculate pooled standard deviation for two groups.
    
    Args:
        n1: Sample size of group 1
        sd1: Standard deviation of group 1
        n2: Sample size of group 2
        sd2: Standard deviation of group 2
        
    Returns:
        Pooled standard deviation
    """
    numerator = (n1 - 1) * sd1**2 + (n2 - 1) * sd2**2
    denominator = n1 + n2 - 2
    return math.sqrt(numerator / denominator)


def hedges_g_correction(n: int) -> float:
    """
    Calculate Hedges' g correction factor for small sample bias.
    
    Args:
        n: Total sample size
        
    Returns:
        Correction factor (multiply Cohen's d by this)
    """
    # Approximation: J = 1 - 3/(4*df - 1)
    df = n - 2
    if df <= 0:
        return 1.0
    return 1 - 3 / (4 * df - 1)
