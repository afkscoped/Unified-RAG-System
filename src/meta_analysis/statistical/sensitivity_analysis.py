"""
Sensitivity Analysis

Implements robustness checks for meta-analyses:
- Leave-one-out analysis
- Influence diagnostics
- Cumulative meta-analysis
- Subgroup sensitivity
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from loguru import logger

from src.meta_analysis.mcp_servers.base_experiment_mcp import StandardizedStudy
from src.meta_analysis.statistical.meta_analyzer import MetaAnalyzer, MetaAnalysisResult


@dataclass
class SensitivityResult:
    """
    Results from sensitivity analysis.
    
    Attributes:
        leave_one_out: Results from leave-one-out analysis
        influence: Influence diagnostics for each study
        cumulative: Cumulative meta-analysis results
        robust: Whether results are robust to sensitivity checks
        influential_studies: List of influential study IDs
        interpretation: Plain-language interpretation
    """
    leave_one_out: Dict[str, Dict[str, float]]
    influence: Dict[str, Dict[str, float]]
    cumulative: Optional[List[Dict[str, Any]]]
    robust: bool
    influential_studies: List[str]
    interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "leave_one_out": self.leave_one_out,
            "influence": self.influence,
            "cumulative": self.cumulative,
            "robust": self.robust,
            "influential_studies": self.influential_studies,
            "interpretation": self.interpretation
        }


class SensitivityAnalyzer:
    """
    Performs sensitivity analyses to assess robustness of meta-analysis results.
    
    Identifies influential studies and checks whether conclusions
    are dependent on single studies.
    """
    
    def __init__(
        self,
        influence_threshold: float = 0.1,
        direction_change_critical: bool = True
    ):
        """
        Initialize analyzer.
        
        Args:
            influence_threshold: Change in effect considered influential
            direction_change_critical: Whether direction change is critical
        """
        self.influence_threshold = influence_threshold
        self.direction_change_critical = direction_change_critical
        self.meta_analyzer = MetaAnalyzer()
    
    def analyze(
        self,
        studies: List[StandardizedStudy],
        original_result: Optional[MetaAnalysisResult] = None
    ) -> SensitivityResult:
        """
        Perform comprehensive sensitivity analysis.
        
        Args:
            studies: List of studies
            original_result: Pre-computed meta-analysis result
            
        Returns:
            SensitivityResult with all sensitivity checks
        """
        if original_result is None:
            original_result = self.meta_analyzer.random_effects_dl(studies)
        
        # Leave-one-out analysis
        loo_results = self.leave_one_out_analysis(studies, original_result)
        
        # Influence diagnostics
        influence_results = self.influence_diagnostics(studies, original_result)
        
        # Cumulative meta-analysis (if we have timestamps)
        cumulative = self._cumulative_analysis(studies)
        
        # Identify influential studies
        influential = self._identify_influential_studies(
            loo_results, influence_results, original_result
        )
        
        # Assess robustness
        robust = self._assess_robustness(loo_results, original_result, influential)
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            loo_results, influence_results, influential, robust, original_result
        )
        
        return SensitivityResult(
            leave_one_out=loo_results,
            influence=influence_results,
            cumulative=cumulative,
            robust=robust,
            influential_studies=influential,
            interpretation=interpretation
        )
    
    def leave_one_out_analysis(
        self,
        studies: List[StandardizedStudy],
        original_result: Optional[MetaAnalysisResult] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Leave-one-out sensitivity analysis.
        
        Re-runs meta-analysis excluding each study in turn
        to assess impact of individual studies.
        
        Args:
            studies: List of studies
            original_result: Original meta-analysis result
            
        Returns:
            Dictionary mapping study IDs to results when excluded
        """
        if len(studies) < 3:
            return {}
        
        if original_result is None:
            original_result = self.meta_analyzer.random_effects_dl(studies)
        
        original_effect = original_result.pooled_effect
        results = {}
        
        for i, excluded_study in enumerate(studies):
            remaining = [s for j, s in enumerate(studies) if j != i]
            
            try:
                loo_result = self.meta_analyzer.random_effects_dl(remaining)
                
                effect_change = loo_result.pooled_effect - original_effect
                relative_change = abs(effect_change / original_effect) if original_effect != 0 else 0
                direction_changed = np.sign(loo_result.pooled_effect) != np.sign(original_effect)
                
                results[excluded_study.study_id] = {
                    "pooled_effect": loo_result.pooled_effect,
                    "standard_error": loo_result.standard_error,
                    "ci_lower": loo_result.confidence_interval[0],
                    "ci_upper": loo_result.confidence_interval[1],
                    "p_value": loo_result.p_value,
                    "effect_change": effect_change,
                    "relative_change": relative_change,
                    "direction_changed": direction_changed,
                    "i2": loo_result.heterogeneity["I2"]
                }
            except Exception as e:
                logger.warning(f"LOO analysis failed for {excluded_study.study_id}: {e}")
        
        return results
    
    def influence_diagnostics(
        self,
        studies: List[StandardizedStudy],
        original_result: Optional[MetaAnalysisResult] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate influence diagnostics for each study.
        
        Computes various influence measures:
        - DFBETAS (change in pooled effect)
        - Cook's distance analogue
        - Leverage (study weight)
        - Externally standardized residual
        
        Args:
            studies: List of studies
            original_result: Original meta-analysis result
            
        Returns:
            Dictionary mapping study IDs to influence measures
        """
        if len(studies) < 3:
            return {}
        
        if original_result is None:
            original_result = self.meta_analyzer.random_effects_dl(studies)
        
        original_effect = original_result.pooled_effect
        original_se = original_result.standard_error
        
        effects = np.array([s.effect_size for s in studies])
        variances = np.array([s.variance for s in studies])
        
        # Random effects weights
        tau2 = original_result.heterogeneity.get("tau2", 0)
        weights = 1.0 / (variances + tau2)
        total_weight = np.sum(weights)
        
        results = {}
        
        for i, study in enumerate(studies):
            # Leverage (relative weight)
            leverage = float(weights[i] / total_weight)
            
            # Residual
            residual = study.effect_size - original_effect
            
            # Standardized residual
            # Variance of residual ≈ (1 - leverage) * study_variance
            residual_var = (1 - leverage) * (variances[i] + tau2)
            std_residual = residual / np.sqrt(residual_var) if residual_var > 0 else 0
            
            # DFBETAS (change in beta per SE when study removed)
            loo_results = self.leave_one_out_analysis(studies, original_result)
            if study.study_id in loo_results:
                loo_effect = loo_results[study.study_id]["pooled_effect"]
                dfbetas = (original_effect - loo_effect) / original_se
            else:
                dfbetas = 0
            
            # Cook's distance analogue
            # D = (residual² * leverage) / (k * SE² * (1-leverage)²)
            k = len(studies)
            cooks_d = (
                (residual ** 2 * leverage) / 
                (k * original_se ** 2 * (1 - leverage) ** 2)
                if (1 - leverage) > 0 else 0
            )
            
            results[study.study_id] = {
                "leverage": leverage,
                "residual": float(residual),
                "std_residual": float(std_residual),
                "dfbetas": float(dfbetas),
                "cooks_distance": float(cooks_d),
                "weight_percent": leverage * 100,
                "is_outlier": abs(std_residual) > 2,
                "is_influential": abs(dfbetas) > 2 / np.sqrt(k) or cooks_d > 4 / k
            }
        
        return results
    
    def _cumulative_analysis(
        self,
        studies: List[StandardizedStudy]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Cumulative meta-analysis ordered by timestamp.
        
        Shows how pooled effect estimate evolved as studies accumulated.
        """
        # Sort by timestamp if available
        dated_studies = [s for s in studies if s.timestamp is not None]
        
        if len(dated_studies) < 3:
            return None
        
        dated_studies.sort(key=lambda s: s.timestamp)
        
        results = []
        cumulative_studies = []
        
        for study in dated_studies:
            cumulative_studies.append(study)
            
            if len(cumulative_studies) >= 2:
                try:
                    result = self.meta_analyzer.random_effects_dl(cumulative_studies)
                    results.append({
                        "study_id": study.study_id,
                        "timestamp": study.timestamp.isoformat() if study.timestamp else None,
                        "n_studies": len(cumulative_studies),
                        "pooled_effect": result.pooled_effect,
                        "ci_lower": result.confidence_interval[0],
                        "ci_upper": result.confidence_interval[1],
                        "p_value": result.p_value
                    })
                except Exception as e:
                    logger.warning(f"Cumulative analysis failed at {study.study_id}: {e}")
        
        return results if results else None
    
    def _identify_influential_studies(
        self,
        loo_results: Dict[str, Dict],
        influence_results: Dict[str, Dict],
        original_result: MetaAnalysisResult
    ) -> List[str]:
        """Identify studies that are influential for conclusions."""
        influential = []
        
        for study_id, loo in loo_results.items():
            # Check for large effect change
            if loo.get("relative_change", 0) > self.influence_threshold:
                influential.append(study_id)
                continue
            
            # Check for direction change
            if self.direction_change_critical and loo.get("direction_changed", False):
                influential.append(study_id)
                continue
            
            # Check for significance change (was significant, now not)
            original_sig = original_result.p_value < 0.05
            loo_sig = loo.get("p_value", 1.0) < 0.05
            if original_sig != loo_sig:
                influential.append(study_id)
                continue
        
        # Also check influence diagnostics
        for study_id, inf in influence_results.items():
            if inf.get("is_influential", False) and study_id not in influential:
                influential.append(study_id)
        
        return influential
    
    def _assess_robustness(
        self,
        loo_results: Dict[str, Dict],
        original_result: MetaAnalysisResult,
        influential: List[str]
    ) -> bool:
        """Assess overall robustness of meta-analysis."""
        if len(loo_results) == 0:
            return True
        
        # If many studies are influential, not robust
        n_studies = original_result.n_studies
        if len(influential) > n_studies * 0.3:
            return False
        
        # If any single study causes direction change, not robust
        for loo in loo_results.values():
            if loo.get("direction_changed", False):
                return False
        
        # If confidence intervals are consistent across LOO
        all_ci_overlap = True
        orig_ci = original_result.confidence_interval
        
        for loo in loo_results.values():
            loo_ci = (loo["ci_lower"], loo["ci_upper"])
            # Check if CIs overlap
            if loo_ci[1] < orig_ci[0] or loo_ci[0] > orig_ci[1]:
                all_ci_overlap = False
                break
        
        return all_ci_overlap
    
    def _generate_interpretation(
        self,
        loo_results: Dict,
        influence_results: Dict,
        influential: List[str],
        robust: bool,
        original_result: MetaAnalysisResult
    ) -> str:
        """Generate interpretation of sensitivity analysis."""
        parts = []
        
        n_studies = original_result.n_studies
        parts.append(f"Sensitivity analysis of {n_studies} studies.")
        
        if len(influential) == 0:
            parts.append("No individual studies substantially influence the results.")
            parts.append("✓ Results are robust to removal of single studies.")
        elif len(influential) == 1:
            parts.append(f"Study '{influential[0]}' is influential.")
            parts.append("Removing this study meaningfully changes the pooled estimate.")
        else:
            parts.append(f"{len(influential)} studies are influential: {', '.join(influential)}.")
        
        # Check for outliers
        outliers = [sid for sid, inf in influence_results.items() if inf.get("is_outlier", False)]
        if outliers:
            parts.append(f"Potential outliers detected: {', '.join(outliers)}.")
        
        # Overall robustness
        if robust:
            parts.append("Overall, the meta-analysis appears robust.")
        else:
            parts.append("⚠️ Results may not be robust. Interpret pooled estimate with caution.")
        
        # Recommendations
        if not robust or len(influential) > 0:
            parts.append("Consider reporting results with and without influential studies.")
        
        return " ".join(parts)
    
    def generate_report(
        self,
        sensitivity_result: SensitivityResult,
        original_result: MetaAnalysisResult
    ) -> str:
        """
        Generate a detailed sensitivity analysis report.
        
        Args:
            sensitivity_result: Results from sensitivity analysis
            original_result: Original meta-analysis result
            
        Returns:
            Formatted report string
        """
        lines = [
            "# Sensitivity Analysis Report",
            "",
            f"**Original Pooled Effect:** {original_result.pooled_effect:.4f} "
            f"(95% CI: {original_result.confidence_interval[0]:.4f}, "
            f"{original_result.confidence_interval[1]:.4f})",
            "",
            "## Leave-One-Out Analysis",
            ""
        ]
        
        # LOO table
        if sensitivity_result.leave_one_out:
            lines.append("| Study Removed | Pooled Effect | 95% CI | Change | Direction |")
            lines.append("|--------------|---------------|--------|--------|-----------|")
            
            for study_id, loo in sensitivity_result.leave_one_out.items():
                direction = "Changed" if loo["direction_changed"] else "Same"
                lines.append(
                    f"| {study_id} | {loo['pooled_effect']:.4f} | "
                    f"({loo['ci_lower']:.4f}, {loo['ci_upper']:.4f}) | "
                    f"{loo['effect_change']:+.4f} | {direction} |"
                )
            lines.append("")
        
        # Influential studies
        if sensitivity_result.influential_studies:
            lines.append("## Influential Studies")
            lines.append("")
            for study_id in sensitivity_result.influential_studies:
                lines.append(f"- **{study_id}**")
            lines.append("")
        
        # Interpretation
        lines.append("## Interpretation")
        lines.append("")
        lines.append(sensitivity_result.interpretation)
        
        return "\n".join(lines)
