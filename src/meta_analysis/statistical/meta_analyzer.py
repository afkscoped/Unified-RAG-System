"""
Meta-Analyzer

Core meta-analysis statistical engine implementing:
- Fixed effects model (inverse-variance weighted)
- Random effects model (DerSimonian-Laird)
- Heterogeneity statistics (I², τ², Q)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy import stats
from loguru import logger

from src.meta_analysis.mcp_servers.base_experiment_mcp import StandardizedStudy


@dataclass
class MetaAnalysisResult:
    """
    Result from a meta-analysis.
    
    Attributes:
        pooled_effect: Pooled effect size estimate
        standard_error: Standard error of pooled estimate
        confidence_interval: 95% CI as (lower, upper)
        z_value: Z-statistic for test of pooled effect = 0
        p_value: P-value for pooled effect
        model_type: "fixed" or "random"
        heterogeneity: Dictionary with Q, I², τ² statistics
        study_weights: Dictionary mapping study IDs to weights
        n_studies: Number of studies included
    """
    pooled_effect: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    z_value: float
    p_value: float
    model_type: str
    heterogeneity: Dict[str, float]
    study_weights: Dict[str, float]
    n_studies: int
    studies: List[StandardizedStudy] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pooled_effect": self.pooled_effect,
            "standard_error": self.standard_error,
            "confidence_interval": {
                "lower": self.confidence_interval[0],
                "upper": self.confidence_interval[1]
            },
            "z_value": self.z_value,
            "p_value": self.p_value,
            "model_type": self.model_type,
            "heterogeneity": self.heterogeneity,
            "study_weights": self.study_weights,
            "n_studies": self.n_studies
        }
    
    @property
    def is_significant(self) -> bool:
        """Whether pooled effect is statistically significant at α=0.05."""
        return self.p_value < 0.05
    
    @property
    def effect_direction(self) -> str:
        """Direction of the pooled effect."""
        if self.pooled_effect > 0:
            return "positive"
        elif self.pooled_effect < 0:
            return "negative"
        return "none"


class MetaAnalyzer:
    """
    Core meta-analysis engine.
    
    Implements fixed effects and random effects meta-analysis models
    with comprehensive heterogeneity statistics.
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        heterogeneity_threshold: float = 0.25
    ):
        """
        Initialize meta-analyzer.
        
        Args:
            confidence_level: Confidence level for intervals (default 0.95)
            heterogeneity_threshold: I² threshold for model selection
        """
        self.confidence_level = confidence_level
        self.heterogeneity_threshold = heterogeneity_threshold
        self._z_crit = stats.norm.ppf((1 + confidence_level) / 2)
    
    def analyze(
        self,
        studies: List[StandardizedStudy],
        model: str = "auto"
    ) -> MetaAnalysisResult:
        """
        Perform meta-analysis on a set of studies.
        
        Args:
            studies: List of standardized studies
            model: Model type - "fixed", "random", or "auto"
            
        Returns:
            MetaAnalysisResult with pooled estimate and statistics
        """
        if len(studies) < 2:
            raise ValueError("Need at least 2 studies for meta-analysis")
        
        # Calculate heterogeneity
        heterogeneity = self.calculate_heterogeneity(studies)
        
        # Auto-select model based on heterogeneity
        if model == "auto":
            if heterogeneity["I2"] > self.heterogeneity_threshold * 100:
                model = "random"
                logger.info(f"Auto-selected random effects model (I²={heterogeneity['I2']:.1f}%)")
            else:
                model = "fixed"
                logger.info(f"Auto-selected fixed effects model (I²={heterogeneity['I2']:.1f}%)")
        
        # Perform analysis
        if model == "fixed":
            return self.fixed_effects(studies, heterogeneity)
        else:
            return self.random_effects_dl(studies, heterogeneity)
    
    def fixed_effects(
        self,
        studies: List[StandardizedStudy],
        heterogeneity: Optional[Dict] = None
    ) -> MetaAnalysisResult:
        """
        Fixed effects meta-analysis (inverse-variance weighted).
        
        Assumes all studies estimate the same true effect.
        
        Args:
            studies: List of studies
            heterogeneity: Pre-calculated heterogeneity (optional)
            
        Returns:
            MetaAnalysisResult
        """
        # Extract effect sizes and variances
        effects = np.array([s.effect_size for s in studies])
        variances = np.array([s.variance for s in studies])
        
        # Calculate weights (inverse variance)
        weights = 1.0 / variances
        total_weight = np.sum(weights)
        
        # Pooled effect (weighted average)
        pooled_effect = np.sum(weights * effects) / total_weight
        
        # Standard error of pooled effect
        pooled_se = np.sqrt(1.0 / total_weight)
        
        # Confidence interval
        ci_lower = pooled_effect - self._z_crit * pooled_se
        ci_upper = pooled_effect + self._z_crit * pooled_se
        
        # Z-test for pooled effect = 0
        z_value = pooled_effect / pooled_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_value)))
        
        # Calculate heterogeneity if not provided
        if heterogeneity is None:
            heterogeneity = self.calculate_heterogeneity(studies)
        
        # Build weight dictionary
        study_weights = {
            s.study_id: float(w / total_weight * 100)
            for s, w in zip(studies, weights)
        }
        
        return MetaAnalysisResult(
            pooled_effect=float(pooled_effect),
            standard_error=float(pooled_se),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            z_value=float(z_value),
            p_value=float(p_value),
            model_type="fixed",
            heterogeneity=heterogeneity,
            study_weights=study_weights,
            n_studies=len(studies),
            studies=studies
        )
    
    def random_effects_dl(
        self,
        studies: List[StandardizedStudy],
        heterogeneity: Optional[Dict] = None
    ) -> MetaAnalysisResult:
        """
        Random effects meta-analysis using DerSimonian-Laird estimator.
        
        Assumes studies estimate different but related true effects
        from a distribution of effects.
        
        Args:
            studies: List of studies
            heterogeneity: Pre-calculated heterogeneity (optional)
            
        Returns:
            MetaAnalysisResult
        """
        # Extract effect sizes and variances
        effects = np.array([s.effect_size for s in studies])
        variances = np.array([s.variance for s in studies])
        
        # Calculate heterogeneity if not provided
        if heterogeneity is None:
            heterogeneity = self.calculate_heterogeneity(studies)
        
        tau2 = heterogeneity["tau2"]
        logger.info(f"Random Effects (DL): tau2={tau2:.4f}, Q={heterogeneity['Q']:.4f}, I2={heterogeneity['I2']:.1f}%")
        
        # Random effects weights (include between-study variance)
        random_variances = variances + tau2
        weights = 1.0 / random_variances
        total_weight = np.sum(weights)
        
        # Debug Log Weights
        weight_dist = [float(w)/total_weight for w in weights]
        logger.debug(f"RE Weights Distribution (First 5): {weight_dist[:5]}")
        
        # Pooled effect (weighted average)
        pooled_effect = np.sum(weights * effects) / total_weight
        
        # Standard error of pooled effect
        pooled_se = np.sqrt(1.0 / total_weight)
        
        logger.info(f"RE Result: Effect={pooled_effect:.4f}, SE={pooled_se:.4f}")
        
        # Confidence interval
        ci_lower = pooled_effect - self._z_crit * pooled_se
        ci_upper = pooled_effect + self._z_crit * pooled_se
        
        # Z-test for pooled effect = 0
        z_value = pooled_effect / pooled_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_value)))
        
        # Build weight dictionary
        study_weights = {
            s.study_id: float(w / total_weight * 100)
            for s, w in zip(studies, weights)
        }
        
        return MetaAnalysisResult(
            pooled_effect=float(pooled_effect),
            standard_error=float(pooled_se),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            z_value=float(z_value),
            p_value=float(p_value),
            model_type="random",
            heterogeneity=heterogeneity,
            study_weights=study_weights,
            n_studies=len(studies),
            studies=studies
        )
    
    def random_effects_hartung_knapp(
        self,
        studies: List[StandardizedStudy],
        heterogeneity: Optional[Dict] = None
    ) -> MetaAnalysisResult:
        """
        Random effects meta-analysis using Hartung-Knapp-Sidik-Jonkman adjustment.
        
        Provides more accurate confidence intervals than DL method, 
        especially when the number of studies is small.
        
        Args:
            studies: List of studies
            heterogeneity: Pre-calculated heterogeneity
            
        Returns:
            MetaAnalysisResult
        """
        # Run DL first to get weights
        dl_result = self.random_effects_dl(studies, heterogeneity)
        
        effects = np.array([s.effect_size for s in studies])
        weights = np.array([dl_result.study_weights[s.study_id] for s in studies])
        # Normalize weights to sum to 1 for calculation
        weights = weights / np.sum(weights)
        
        pooled_effect = dl_result.pooled_effect
        k = len(studies)
        
        # Calculate HK variance adjustment factor
        # q = sum(w * (y - mu)^2) / (k - 1)
        q_factor = np.sum(weights * (effects - pooled_effect) ** 2) / (k - 1)
        
        # Enforce minimum q (can't be negative, but also shouldn't reduce variance below fixed)
        q_factor = max(1.0, q_factor)
        
        # Adjusted standard error
        hk_se = dl_result.standard_error * np.sqrt(q_factor)
        
        # Use t-distribution with k-1 df
        t_crit = stats.t.ppf((1 + self.confidence_level) / 2, k - 1)
        
        ci_lower = pooled_effect - t_crit * hk_se
        ci_upper = pooled_effect + t_crit * hk_se
        
        t_value = pooled_effect / hk_se
        p_value = 2 * (1 - stats.t.cdf(abs(t_value), k - 1))
        
        return MetaAnalysisResult(
            pooled_effect=float(pooled_effect),
            standard_error=float(hk_se),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            z_value=float(t_value),
            p_value=float(p_value),
            model_type="random_hk",
            heterogeneity=dl_result.heterogeneity,
            study_weights=dl_result.study_weights,
            n_studies=k,
            studies=studies
        )
    
    def calculate_heterogeneity(
        self,
        studies: List[StandardizedStudy]
    ) -> Dict[str, float]:
        """
        Calculate heterogeneity statistics.
        
        Returns:
            Dictionary with:
            - Q: Cochran's Q statistic
            - Q_df: Degrees of freedom for Q
            - Q_pvalue: P-value for Q test
            - I2: I² statistic (percentage of variance due to heterogeneity)
            - tau2: τ² (between-study variance) via DerSimonian-Laird
            - tau: τ (standard deviation of true effects)
            - H2: H² statistic
        """
        effects = np.array([s.effect_size for s in studies])
        variances = np.array([s.variance for s in studies])
        weights = 1.0 / variances
        
        k = len(studies)
        df = k - 1
        
        # Fixed effects pooled estimate for Q calculation
        pooled_fixed = np.sum(weights * effects) / np.sum(weights)
        
        # Cochran's Q statistic
        Q = np.sum(weights * (effects - pooled_fixed) ** 2)
        
        # P-value for Q (chi-squared distribution)
        Q_pvalue = 1 - stats.chi2.cdf(Q, df)
        
        # I² statistic
        I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0
        
        # τ² via DerSimonian-Laird method
        C = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
        tau2 = max(0, (Q - df) / C) if C > 0 else 0
        
        # τ (standard deviation)
        tau = np.sqrt(tau2)
        
        # H² statistic
        H2 = Q / df if df > 0 else 1
        
        return {
            "Q": float(Q),
            "Q_df": int(df),
            "Q_pvalue": float(Q_pvalue),
            "I2": float(I2),
            "tau2": float(tau2),
            "tau": float(tau),
            "H2": float(H2)
        }
    
    def calculate_i2(self, studies: List[StandardizedStudy]) -> float:
        """
        Calculate I² heterogeneity statistic.
        
        I² represents the percentage of total variation across studies
        that is due to heterogeneity rather than chance.
        
        Interpretation:
        - 0-25%: Low heterogeneity
        - 25-50%: Moderate heterogeneity
        - 50-75%: Substantial heterogeneity
        - 75-100%: Considerable heterogeneity
        
        Args:
            studies: List of studies
            
        Returns:
            I² value (0-100)
        """
        return self.calculate_heterogeneity(studies)["I2"]
    
    def calculate_tau2(self, studies: List[StandardizedStudy]) -> float:
        """
        Calculate τ² (between-study variance).
        
        τ² quantifies the amount of between-study variability
        in true effect sizes.
        
        Args:
            studies: List of studies
            
        Returns:
            τ² value
        """
        return self.calculate_heterogeneity(studies)["tau2"]
    
    def prediction_interval(
        self,
        result: MetaAnalysisResult
    ) -> Tuple[float, float]:
        """
        Calculate 95% prediction interval for a new study.
        
        The prediction interval estimates where 95% of true effects
        from future studies would fall.
        
        Args:
            result: Meta-analysis result
            
        Returns:
            Tuple of (lower, upper) bounds
        """
        tau2 = result.heterogeneity.get("tau2", 0)
        k = result.n_studies
        
        # Prediction interval variance
        # Var = SE² + τ² + τ²/k (additional uncertainty)
        pred_var = result.standard_error ** 2 + tau2
        
        # Use t-distribution with k-2 df for small samples
        if k > 2:
            t_crit = stats.t.ppf((1 + self.confidence_level) / 2, k - 2)
        else:
            t_crit = self._z_crit
        
        pred_se = np.sqrt(pred_var)
        lower = result.pooled_effect - t_crit * pred_se
        upper = result.pooled_effect + t_crit * pred_se
        
        return (float(lower), float(upper))
    
    def compare_models(
        self,
        studies: List[StandardizedStudy]
    ) -> Dict[str, Any]:
        """
        Compare fixed and random effects models.
        
        Args:
            studies: List of studies
            
        Returns:
            Comparison results
        """
        fixed_result = self.fixed_effects(studies)
        random_result = self.random_effects_dl(studies)
        
        return {
            "fixed_effects": fixed_result.to_dict(),
            "random_effects": random_result.to_dict(),
            "heterogeneity": fixed_result.heterogeneity,
            "recommended_model": "random" if fixed_result.heterogeneity["I2"] > 25 else "fixed",
            "effect_difference": abs(fixed_result.pooled_effect - random_result.pooled_effect),
            "interpretation": self._interpret_comparison(fixed_result, random_result)
        }
    
    def _interpret_comparison(
        self,
        fixed: MetaAnalysisResult,
        random: MetaAnalysisResult
    ) -> str:
        """Generate interpretation of model comparison."""
        i2 = fixed.heterogeneity["I2"]
        diff = abs(fixed.pooled_effect - random.pooled_effect)
        
        interpretation = []
        
        if i2 < 25:
            interpretation.append("Low heterogeneity suggests homogeneous effects.")
            interpretation.append("Fixed effects model is appropriate.")
        elif i2 < 50:
            interpretation.append("Moderate heterogeneity detected.")
            interpretation.append("Consider random effects model for generalizability.")
        elif i2 < 75:
            interpretation.append("Substantial heterogeneity indicates varying true effects.")
            interpretation.append("Random effects model recommended. Explore sources of heterogeneity.")
        else:
            interpretation.append("High heterogeneity suggests very different true effects across studies.")
            interpretation.append("Consider subgroup analysis or meta-regression.")
        
        if diff > 0.1:
            interpretation.append(f"Models differ by {diff:.3f} - model choice matters.")
        else:
            interpretation.append("Models give similar estimates.")
        
        return " ".join(interpretation)
