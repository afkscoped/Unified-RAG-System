"""
Publication Bias Detection

Implements methods for detecting and adjusting for publication bias:
- Egger's regression test
- Begg's rank correlation test  
- Trim-and-fill method
- Failsafe N calculation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy import stats
from loguru import logger

from src.meta_analysis.mcp_servers.base_experiment_mcp import StandardizedStudy
from src.meta_analysis.statistical.meta_analyzer import MetaAnalyzer, MetaAnalysisResult


@dataclass
class PublicationBiasResult:
    """
    Results from publication bias analysis.
    
    Attributes:
        eggers_test: Egger's regression test results
        beggs_test: Begg's rank correlation results
        trim_and_fill: Trim-and-fill adjusted results
        failsafe_n: Rosenthal's failsafe N
        bias_detected: Overall assessment of bias presence
        adjusted_effect: Bias-adjusted effect estimate
    """
    eggers_test: Dict[str, float]
    beggs_test: Optional[Dict[str, float]]
    trim_and_fill: Optional[Dict[str, Any]]
    failsafe_n: int
    bias_detected: bool
    bias_severity: str  # "none", "possible", "likely", "severe"
    adjusted_effect: Optional[float]
    interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "eggers_test": self.eggers_test,
            "beggs_test": self.beggs_test,
            "trim_and_fill": self.trim_and_fill,
            "failsafe_n": self.failsafe_n,
            "bias_detected": self.bias_detected,
            "bias_severity": self.bias_severity,
            "adjusted_effect": self.adjusted_effect,
            "interpretation": self.interpretation
        }


class PublicationBiasDetector:
    """
    Detects and adjusts for publication bias in meta-analyses.
    
    Implements multiple methods for assessing funnel plot asymmetry
    and estimating bias-adjusted effect sizes.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize detector.
        
        Args:
            alpha: Significance level for tests
        """
        self.alpha = alpha
        self.meta_analyzer = MetaAnalyzer()
    
    def analyze(
        self,
        studies: List[StandardizedStudy],
        meta_result: Optional[MetaAnalysisResult] = None
    ) -> PublicationBiasResult:
        """
        Perform comprehensive publication bias analysis.
        
        Args:
            studies: List of studies
            meta_result: Pre-computed meta-analysis result
            
        Returns:
            PublicationBiasResult with all test results
        """
        if len(studies) < 3:
            raise ValueError("Need at least 3 studies for publication bias analysis")
        
        # Run Egger's test
        eggers = self.eggers_test(studies)
        
        # Run Begg's test
        beggs = self.beggs_test(studies) if len(studies) >= 5 else None
        
        # Run trim-and-fill
        trim_fill = self.trim_and_fill(studies, meta_result)
        
        # Calculate failsafe N
        failsafe = self.failsafe_n(studies, meta_result)
        
        # Assess overall bias
        bias_detected, severity = self._assess_bias(eggers, beggs, trim_fill, len(studies))
        
        # Get adjusted effect
        adjusted_effect = trim_fill.get("adjusted_effect") if trim_fill else None
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            eggers, beggs, trim_fill, failsafe, bias_detected, severity
        )
        
        return PublicationBiasResult(
            eggers_test=eggers,
            beggs_test=beggs,
            trim_and_fill=trim_fill,
            failsafe_n=failsafe,
            bias_detected=bias_detected,
            bias_severity=severity,
            adjusted_effect=adjusted_effect,
            interpretation=interpretation
        )
    
    def eggers_test(
        self,
        studies: List[StandardizedStudy]
    ) -> Dict[str, float]:
        """
        Egger's regression test for funnel plot asymmetry.
        
        Tests whether smaller studies show systematically different
        effects than larger studies - a hallmark of publication bias.
        
        The test regresses standardized effect (effect/SE) on precision (1/SE).
        Significant intercept suggests bias.
        
        Args:
            studies: List of studies
            
        Returns:
            Dictionary with intercept, slope, SE, t-value, p-value
        """
        effects = np.array([s.effect_size for s in studies])
        standard_errors = np.array([s.standard_error for s in studies])
        
        # Precision (independent variable)
        precision = 1.0 / standard_errors
        
        # Standardized effect (dependent variable)
        std_effect = effects / standard_errors
        
        # Weighted least squares regression
        # Weight by precision squared
        weights = precision ** 2
        
        # Fit regression: std_effect = intercept + slope * precision
        n = len(studies)
        sum_w = np.sum(weights)
        sum_wx = np.sum(weights * precision)
        sum_wy = np.sum(weights * std_effect)
        sum_wxy = np.sum(weights * precision * std_effect)
        sum_wxx = np.sum(weights * precision ** 2)
        
        # Regression coefficients
        denom = sum_w * sum_wxx - sum_wx ** 2
        if abs(denom) < 1e-10:
            return {
                "intercept": 0.0,
                "slope": 0.0,
                "se_intercept": float('inf'),
                "t_value": 0.0,
                "p_value": 1.0,
                "df": n - 2
            }
        
        slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
        intercept = (sum_wy - slope * sum_wx) / sum_w
        
        # Residuals and SE
        predicted = intercept + slope * precision
        residuals = std_effect - predicted
        mse = np.sum(weights * residuals ** 2) / (n - 2) if n > 2 else 0
        
        # SE of intercept
        var_intercept = mse * sum_wxx / denom
        se_intercept = np.sqrt(var_intercept) if var_intercept > 0 else 0
        
        # T-test for intercept
        t_value = intercept / se_intercept if se_intercept > 0 else 0
        df = n - 2
        p_value = 2 * (1 - stats.t.cdf(abs(t_value), df)) if df > 0 else 1.0
        
        return {
            "intercept": float(intercept),
            "slope": float(slope),
            "se_intercept": float(se_intercept),
            "t_value": float(t_value),
            "p_value": float(p_value),
            "df": int(df),
            "significant": p_value < self.alpha
        }
    
    def beggs_test(
        self,
        studies: List[StandardizedStudy]
    ) -> Dict[str, float]:
        """
        Begg and Mazumdar's rank correlation test.
        
        Tests correlation between effect sizes and their variances
        using Kendall's tau.
        
        Args:
            studies: List of studies
            
        Returns:
            Dictionary with tau, z-value, p-value
        """
        effects = np.array([s.effect_size for s in studies])
        variances = np.array([s.variance for s in studies])
        
        # Calculate standardized effect sizes (adjusted for pooled effect)
        weights = 1.0 / variances
        pooled = np.sum(weights * effects) / np.sum(weights)
        
        # Standardized residuals
        std_residuals = (effects - pooled) / np.sqrt(variances)
        
        # Kendall's tau between standardized residuals and variances
        tau, p_value = stats.kendalltau(std_residuals, variances)
        
        # Z-value
        n = len(studies)
        z_value = tau / np.sqrt((2 * (2 * n + 5)) / (9 * n * (n - 1)))
        
        return {
            "tau": float(tau),
            "z_value": float(z_value),
            "p_value": float(p_value),
            "significant": p_value < self.alpha
        }
    
    def trim_and_fill(
        self,
        studies: List[StandardizedStudy],
        meta_result: Optional[MetaAnalysisResult] = None,
        estimator: str = "L0",
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Duval and Tweedie's trim-and-fill method.
        
        Estimates the number of "missing" studies due to publication bias
        and provides a bias-adjusted pooled effect.
        
        Args:
            studies: List of studies
            meta_result: Pre-computed meta-analysis result
            estimator: Estimator for k0 ("L0", "R0", "Q0")
            max_iterations: Maximum iterations for convergence
            
        Returns:
            Dictionary with k0, imputed studies, adjusted effect
        """
        if meta_result is None:
            meta_result = self.meta_analyzer.random_effects_dl(studies)
        
        pooled = meta_result.pooled_effect
        
        effects = np.array([s.effect_size for s in studies])
        standard_errors = np.array([s.standard_error for s in studies])
        
        # Center effects around pooled estimate
        centered = effects - pooled
        
        # Rank by deviation from pooled effect
        n = len(studies)
        
        # Identify asymmetry side (studies on one side of distribution)
        # Typically, publication bias causes missing studies with small/null effects
        right_side = np.sum(centered > 0)
        left_side = np.sum(centered < 0)
        
        if right_side > left_side:
            # More studies on positive side - trim from right
            trim_positive = True
        else:
            trim_positive = False
        
        # Estimate k0 (number of missing studies) using L0 estimator
        abs_centered = np.abs(centered)
        ranks = stats.rankdata(abs_centered)
        
        # For L0 estimator
        if trim_positive:
            S_plus = np.sum(ranks[centered > 0])
        else:
            S_plus = np.sum(ranks[centered < 0])
        
        # L0 estimator for k0
        k0_est = max(0, int(4 * S_plus / n - (n + 1) / 2))
        
        # Impute missing studies
        imputed_studies = []
        if k0_est > 0:
            # Sort by absolute deviation
            sorted_indices = np.argsort(-abs_centered)
            
            for i in range(min(k0_est, len(studies))):
                idx = sorted_indices[i]
                if (trim_positive and centered[idx] > 0) or (not trim_positive and centered[idx] < 0):
                    # Create mirror study
                    mirror_effect = 2 * pooled - effects[idx]
                    imputed_studies.append({
                        "effect_size": float(mirror_effect),
                        "standard_error": float(standard_errors[idx]),
                        "original_study_id": studies[idx].study_id
                    })
        
        # Calculate adjusted pooled effect with imputed studies
        adjusted_effect = None
        if imputed_studies:
            # Add imputed effects
            all_effects = list(effects) + [s["effect_size"] for s in imputed_studies]
            all_ses = list(standard_errors) + [s["standard_error"] for s in imputed_studies]
            
            # Random effects pooling with all studies
            variances = np.array(all_ses) ** 2
            weights = 1.0 / variances
            adjusted_effect = float(np.sum(weights * all_effects) / np.sum(weights))
        else:
            adjusted_effect = pooled
        
        return {
            "k0": k0_est,
            "original_n": n,
            "adjusted_n": n + k0_est,
            "imputed_studies": imputed_studies,
            "original_effect": float(pooled),
            "adjusted_effect": adjusted_effect,
            "effect_change": abs(adjusted_effect - pooled) if adjusted_effect else 0,
            "percent_change": abs(adjusted_effect - pooled) / abs(pooled) * 100 if pooled != 0 else 0
        }
    
    def failsafe_n(
        self,
        studies: List[StandardizedStudy],
        meta_result: Optional[MetaAnalysisResult] = None,
        target_alpha: float = 0.05
    ) -> int:
        """
        Rosenthal's failsafe N.
        
        Calculates the number of null-effect studies needed to reduce
        the pooled effect to non-significance.
        
        Args:
            studies: List of studies
            meta_result: Pre-computed meta-analysis result
            target_alpha: Alpha level for significance
            
        Returns:
            Failsafe N value
        """
        if meta_result is None:
            meta_result = self.meta_analyzer.random_effects_dl(studies)
        
        z_pooled = meta_result.z_value
        k = len(studies)
        z_crit = stats.norm.ppf(1 - target_alpha / 2)
        
        # Rosenthal's formula: N = (sum(z) / z_crit)² - k
        # Approximate sum(z) from pooled z
        failsafe = max(0, int((z_pooled * np.sqrt(k) / z_crit) ** 2 - k))
        
        return failsafe
    
    def _assess_bias(
        self,
        eggers: Dict,
        beggs: Optional[Dict],
        trim_fill: Optional[Dict],
        n_studies: int
    ) -> Tuple[bool, str]:
        """Assess overall bias presence and severity."""
        
        indicators = 0
        
        # Check Egger's test
        if eggers.get("significant", False):
            indicators += 2
        elif eggers.get("p_value", 1.0) < 0.1:
            indicators += 1
        
        # Check Begg's test
        if beggs and beggs.get("significant", False):
            indicators += 1
        
        # Check trim-and-fill
        if trim_fill:
            k0 = trim_fill.get("k0", 0)
            if k0 > n_studies * 0.3:  # More than 30% imputed
                indicators += 2
            elif k0 > 0:
                indicators += 1
            
            # Check effect change
            if trim_fill.get("percent_change", 0) > 20:
                indicators += 1
        
        # Determine severity
        if indicators == 0:
            return False, "none"
        elif indicators <= 1:
            return False, "possible"
        elif indicators <= 3:
            return True, "likely"
        else:
            return True, "severe"
    
    def _generate_interpretation(
        self,
        eggers: Dict,
        beggs: Optional[Dict],
        trim_fill: Optional[Dict],
        failsafe: int,
        bias_detected: bool,
        severity: str
    ) -> str:
        """Generate human-readable interpretation."""
        
        parts = []
        
        # Egger's test result
        if eggers.get("significant"):
            parts.append(f"Egger's test is significant (p={eggers['p_value']:.3f}), "
                        "indicating funnel plot asymmetry.")
        else:
            parts.append(f"Egger's test is non-significant (p={eggers['p_value']:.3f}).")
        
        # Trim-and-fill
        if trim_fill:
            k0 = trim_fill.get("k0", 0)
            if k0 > 0:
                parts.append(f"Trim-and-fill identifies {k0} potentially missing studies. "
                           f"Adjusted effect: {trim_fill['adjusted_effect']:.3f} "
                           f"(original: {trim_fill['original_effect']:.3f}).")
            else:
                parts.append("Trim-and-fill does not impute any missing studies.")
        
        # Failsafe N interpretation
        if failsafe > 0:
            parts.append(f"Failsafe N = {failsafe} null studies needed to nullify the effect.")
            if failsafe > 5 * len(eggers) + 10:
                parts.append("This exceeds the 5k+10 threshold, suggesting robust findings.")
        
        # Overall assessment
        if severity == "none":
            parts.append("Overall: No clear evidence of publication bias.")
        elif severity == "possible":
            parts.append("Overall: Some indicators suggest possible publication bias, "
                        "but evidence is not conclusive.")
        elif severity == "likely":
            parts.append("Overall: Publication bias appears likely. "
                        "Consider using adjusted estimates.")
        else:
            parts.append("Overall: Strong evidence of publication bias. "
                        "Results should be interpreted with caution.")
        
        return " ".join(parts)
