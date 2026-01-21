"""
Simpson's Paradox Detector

Implements detection and analysis of Simpson's Paradox:
- Stratified meta-analysis
- Subgroup analysis
- Meta-regression for confounders
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy import stats
from loguru import logger

from src.meta_analysis.mcp_servers.base_experiment_mcp import StandardizedStudy
from src.meta_analysis.statistical.meta_analyzer import MetaAnalyzer, MetaAnalysisResult


@dataclass
class SimpsonsParadoxResult:
    """
    Results from Simpson's Paradox detection.
    
    Attributes:
        paradox_detected: Whether paradox was found
        overall_effect: Overall pooled effect
        subgroup_effects: Effects within each subgroup
        reversal_details: Details of any effect reversals
        regression_results: Meta-regression results (if applicable)
        interpretation: Plain-language interpretation
    """
    paradox_detected: bool
    overall_effect: float
    subgroup_effects: Dict[str, Dict[str, float]]
    reversal_details: List[Dict[str, Any]]
    regression_results: Optional[Dict[str, Any]]
    interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "paradox_detected": self.paradox_detected,
            "overall_effect": self.overall_effect,
            "subgroup_effects": self.subgroup_effects,
            "reversal_details": self.reversal_details,
            "regression_results": self.regression_results,
            "interpretation": self.interpretation
        }


class SimpsonsParadoxDetector:
    """
    Detects Simpson's Paradox in meta-analysis.
    
    Simpson's Paradox occurs when the overall effect differs in direction
    from effects within subgroups, typically due to confounding.
    """
    
    def __init__(self, significance_threshold: float = 0.05):
        """
        Initialize detector.
        
        Args:
            significance_threshold: Alpha for statistical tests
        """
        self.significance_threshold = significance_threshold
        self.meta_analyzer = MetaAnalyzer()
    
    def detect_simpsons_paradox(
        self,
        studies: List[StandardizedStudy],
        segment_key: str = "segment"
    ) -> SimpsonsParadoxResult:
        """
        Detect Simpson's Paradox by comparing overall vs subgroup effects.
        
        Args:
            studies: List of studies with segment information
            segment_key: Key in metadata/segments for grouping
            
        Returns:
            SimpsonsParadoxResult with detection results
        """
        # Calculate overall effect
        overall_result = self.meta_analyzer.random_effects_dl(studies)
        overall_effect = overall_result.pooled_effect
        overall_direction = np.sign(overall_effect)
        
        # Group studies by segment
        grouped = self._group_by_segment(studies, segment_key)
        
        if len(grouped) < 2:
            return SimpsonsParadoxResult(
                paradox_detected=False,
                overall_effect=overall_effect,
                subgroup_effects={},
                reversal_details=[],
                regression_results=None,
                interpretation="Insufficient subgroups for Simpson's Paradox detection."
            )
        
        # Analyze each subgroup
        subgroup_effects = {}
        reversal_details = []
        
        for group_name, group_studies in grouped.items():
            if len(group_studies) >= 2:
                try:
                    group_result = self.meta_analyzer.random_effects_dl(group_studies)
                    
                    subgroup_effects[group_name] = {
                        "effect": group_result.pooled_effect,
                        "se": group_result.standard_error,
                        "ci_lower": group_result.confidence_interval[0],
                        "ci_upper": group_result.confidence_interval[1],
                        "n_studies": group_result.n_studies,
                        "p_value": group_result.p_value
                    }
                    
                    # Check for reversal
                    group_direction = np.sign(group_result.pooled_effect)
                    if group_direction != 0 and overall_direction != 0:
                        if group_direction != overall_direction:
                            reversal_details.append({
                                "subgroup": group_name,
                                "subgroup_effect": group_result.pooled_effect,
                                "overall_effect": overall_effect,
                                "reversal_magnitude": abs(group_result.pooled_effect) + abs(overall_effect)
                            })
                except Exception as e:
                    logger.warning(f"Could not analyze subgroup {group_name}: {e}")
        
        paradox_detected = len(reversal_details) > 0
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            paradox_detected, overall_effect, subgroup_effects, reversal_details
        )
        
        return SimpsonsParadoxResult(
            paradox_detected=paradox_detected,
            overall_effect=overall_effect,
            subgroup_effects=subgroup_effects,
            reversal_details=reversal_details,
            regression_results=None,
            interpretation=interpretation
        )
    
    def stratified_meta_analysis(
        self,
        studies: List[StandardizedStudy],
        stratify_by: str
    ) -> Dict[str, MetaAnalysisResult]:
        """
        Perform stratified meta-analysis by a given factor.
        
        Args:
            studies: List of studies
            stratify_by: Key to stratify by (from metadata)
            
        Returns:
            Dictionary mapping strata names to MetaAnalysisResult
        """
        grouped = self._group_by_segment(studies, stratify_by)
        
        results = {}
        for stratum, stratum_studies in grouped.items():
            if len(stratum_studies) >= 2:
                try:
                    results[stratum] = self.meta_analyzer.random_effects_dl(stratum_studies)
                except Exception as e:
                    logger.warning(f"Could not analyze stratum {stratum}: {e}")
        
        return results
    
    def meta_regression(
        self,
        studies: List[StandardizedStudy],
        covariate_key: str
    ) -> Dict[str, Any]:
        """
        Perform simple meta-regression with a single covariate.
        
        Models the relationship between effect size and a study-level covariate.
        
        Args:
            studies: List of studies
            covariate_key: Key for covariate (from metadata)
            
        Returns:
            Regression results
        """
        # Extract effect sizes, SEs, and covariate values
        effects = []
        variances = []
        covariates = []
        
        for study in studies:
            if study.metadata and covariate_key in study.metadata:
                try:
                    cov_value = float(study.metadata[covariate_key])
                    effects.append(study.effect_size)
                    variances.append(study.variance)
                    covariates.append(cov_value)
                except (ValueError, TypeError):
                    continue
        
        if len(effects) < 3:
            return {
                "success": False,
                "error": "Insufficient studies with covariate data"
            }
        
        effects = np.array(effects)
        variances = np.array(variances)
        covariates = np.array(covariates)
        weights = 1.0 / variances
        
        # Weighted least squares
        n = len(effects)
        sum_w = np.sum(weights)
        sum_wx = np.sum(weights * covariates)
        sum_wy = np.sum(weights * effects)
        sum_wxy = np.sum(weights * covariates * effects)
        sum_wxx = np.sum(weights * covariates ** 2)
        
        denom = sum_w * sum_wxx - sum_wx ** 2
        if abs(denom) < 1e-10:
            return {
                "success": False,
                "error": "Singular matrix in regression"
            }
        
        slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
        intercept = (sum_wy - slope * sum_wx) / sum_w
        
        # Residuals and test statistics
        predicted = intercept + slope * covariates
        residuals = effects - predicted
        ss_res = np.sum(weights * residuals ** 2)
        mse = ss_res / (n - 2) if n > 2 else 0
        
        var_slope = mse * sum_w / denom
        se_slope = np.sqrt(var_slope) if var_slope > 0 else 0
        
        t_value = slope / se_slope if se_slope > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_value), n - 2)) if n > 2 else 1
        
        # R-squared (explained variance)
        mean_effect = np.sum(weights * effects) / sum_w
        ss_tot = np.sum(weights * (effects - mean_effect) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            "success": True,
            "covariate": covariate_key,
            "n_studies": n,
            "intercept": float(intercept),
            "slope": float(slope),
            "se_slope": float(se_slope),
            "t_value": float(t_value),
            "p_value": float(p_value),
            "r_squared": float(r_squared),
            "significant": p_value < self.significance_threshold,
            "interpretation": self._interpret_regression(slope, p_value, r_squared, covariate_key)
        }
    
    def _group_by_segment(
        self,
        studies: List[StandardizedStudy],
        segment_key: str
    ) -> Dict[str, List[StandardizedStudy]]:
        """Group studies by segment value."""
        grouped = {}
        
        for study in studies:
            # Try to get segment from various locations
            segment_value = None
            
            # Check metadata
            if study.metadata and segment_key in study.metadata:
                segment_value = str(study.metadata[segment_key])
            # Check segments
            elif study.segments and segment_key in study.segments:
                segment_value = str(study.segments[segment_key])
            
            if segment_value:
                if segment_value not in grouped:
                    grouped[segment_value] = []
                grouped[segment_value].append(study)
        
        return grouped
    
    def _generate_interpretation(
        self,
        paradox_detected: bool,
        overall_effect: float,
        subgroup_effects: Dict,
        reversal_details: List
    ) -> str:
        """Generate interpretation of paradox detection."""
        parts = []
        
        overall_dir = "positive" if overall_effect > 0 else "negative"
        parts.append(f"Overall pooled effect is {overall_dir} ({overall_effect:.3f}).")
        
        if len(subgroup_effects) > 0:
            parts.append(f"Analysis includes {len(subgroup_effects)} subgroups.")
        
        if paradox_detected:
            parts.append(f"⚠️ SIMPSON'S PARADOX DETECTED in {len(reversal_details)} subgroup(s):")
            for reversal in reversal_details:
                sub_dir = "positive" if reversal["subgroup_effect"] > 0 else "negative"
                parts.append(
                    f"  - {reversal['subgroup']}: effect is {sub_dir} "
                    f"({reversal['subgroup_effect']:.3f}), opposite to overall."
                )
            parts.append(
                "This suggests confounding by the stratification variable. "
                "Consider reporting stratified results or investigating the confounder."
            )
        else:
            parts.append("No Simpson's Paradox detected - subgroup effects align with overall direction.")
        
        return " ".join(parts)
    
    def _interpret_regression(
        self,
        slope: float,
        p_value: float,
        r_squared: float,
        covariate: str
    ) -> str:
        """Interpret meta-regression results."""
        if p_value >= self.significance_threshold:
            return (f"The covariate '{covariate}' is not significantly associated "
                   f"with effect size (p={p_value:.3f}).")
        
        direction = "increases" if slope > 0 else "decreases"
        return (
            f"The covariate '{covariate}' is significantly associated with effect size "
            f"(p={p_value:.3f}). Effect size {direction} by {abs(slope):.3f} per unit "
            f"increase in {covariate}. This explains {r_squared*100:.1f}% of between-study variance."
        )
    
    def check_subgroup_differences(
        self,
        subgroup_results: Dict[str, MetaAnalysisResult]
    ) -> Dict[str, Any]:
        """
        Test whether subgroup effects differ significantly.
        
        Uses the Q_between statistic to test for between-subgroup heterogeneity.
        
        Args:
            subgroup_results: Dictionary of MetaAnalysisResult by subgroup
            
        Returns:
            Test results
        """
        if len(subgroup_results) < 2:
            return {
                "success": False,
                "error": "Need at least 2 subgroups for comparison"
            }
        
        effects = []
        variances = []
        n_studies = []
        
        for name, result in subgroup_results.items():
            effects.append(result.pooled_effect)
            variances.append(result.standard_error ** 2)
            n_studies.append(result.n_studies)
        
        effects = np.array(effects)
        variances = np.array(variances)
        weights = 1.0 / variances
        
        # Overall pooled effect (across subgroups)
        pooled_overall = np.sum(weights * effects) / np.sum(weights)
        
        # Q_between statistic
        Q_between = np.sum(weights * (effects - pooled_overall) ** 2)
        df = len(effects) - 1
        p_value = 1 - stats.chi2.cdf(Q_between, df)
        
        return {
            "success": True,
            "Q_between": float(Q_between),
            "df": int(df),
            "p_value": float(p_value),
            "significant_difference": p_value < self.significance_threshold,
            "subgroup_effects": {
                name: result.pooled_effect 
                for name, result in subgroup_results.items()
            },
            "interpretation": self._interpret_subgroup_difference(Q_between, p_value, df)
        }
    
    def _interpret_subgroup_difference(
        self,
        q_between: float,
        p_value: float,
        df: int
    ) -> str:
        """Interpret subgroup difference test."""
        if p_value < self.significance_threshold:
            return (
                f"Significant between-subgroup heterogeneity detected "
                f"(Q={q_between:.2f}, df={df}, p={p_value:.3f}). "
                "Effect sizes differ meaningfully across subgroups."
            )
        else:
            return (
                f"No significant between-subgroup heterogeneity "
                f"(Q={q_between:.2f}, df={df}, p={p_value:.3f}). "
                "Effect sizes are consistent across subgroups."
            )
