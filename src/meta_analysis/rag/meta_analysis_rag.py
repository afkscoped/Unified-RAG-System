"""
Meta-Analysis RAG

Specialized RAG system for statistical interpretation and recommendations.
Provides AI-powered insights for meta-analysis results.
"""

from typing import Dict, List, Optional, Any
from loguru import logger

from src.meta_analysis.statistical.meta_analyzer import MetaAnalysisResult
from src.meta_analysis.statistical.publication_bias import PublicationBiasResult
from src.meta_analysis.statistical.sensitivity_analysis import SensitivityResult


# Statistical term definitions for tooltips
STATISTICAL_TOOLTIPS = {
    "I2": {
        "name": "I² (I-squared)",
        "definition": "The percentage of total variation across studies that is due to heterogeneity rather than chance.",
        "interpretation": {
            "0-25": "Low heterogeneity - studies are quite consistent",
            "25-50": "Moderate heterogeneity - some variation between studies",
            "50-75": "Substantial heterogeneity - considerable variation exists",
            "75-100": "High heterogeneity - very different true effects across studies"
        },
        "reference": "Higgins JPT, Thompson SG. Quantifying heterogeneity in a meta‐analysis. Statistics in Medicine. 2002;21(11):1539-1558."
    },
    "tau2": {
        "name": "τ² (tau-squared)",
        "definition": "The between-study variance in a random-effects meta-analysis. Represents how much the true effects vary across studies.",
        "interpretation": "Higher values indicate greater heterogeneity. τ (tau) is the standard deviation of true effects.",
        "reference": "DerSimonian R, Laird N. Meta-analysis in clinical trials. Controlled Clinical Trials. 1986;7(3):177-188."
    },
    "eggers_test": {
        "name": "Egger's Test",
        "definition": "A statistical test for funnel plot asymmetry. Tests whether smaller studies show systematically different effects than larger studies.",
        "interpretation": "Significant result (p < 0.05) suggests potential publication bias, though could also indicate other small-study effects.",
        "reference": "Egger M, Smith GD, Schneider M, Minder C. Bias in meta-analysis detected by a simple, graphical test. BMJ. 1997;315(7109):629-634."
    },
    "trim_and_fill": {
        "name": "Trim and Fill",
        "definition": "A method to estimate the number of 'missing' studies due to publication bias and provide an adjusted effect estimate.",
        "interpretation": "Imputes potentially suppressed studies and recalculates the pooled effect. The difference between original and adjusted estimates indicates bias impact.",
        "reference": "Duval S, Tweedie R. Trim and fill: a simple funnel‐plot–based method of testing and adjusting for publication bias in meta‐analysis. Biometrics. 2000;56(2):455-463."
    },
    "forest_plot": {
        "name": "Forest Plot",
        "definition": "A graphical display showing individual study effects as squares (sized by weight) with confidence intervals, and the pooled effect as a diamond.",
        "interpretation": "Allows visual assessment of study consistency. If CIs largely overlap, studies are consistent. The diamond shows the overall conclusion.",
        "reference": "Lewis S, Clarke M. Forest plots: trying to see the wood and the trees. BMJ. 2001;322(7300):1479-1480."
    },
    "funnel_plot": {
        "name": "Funnel Plot",
        "definition": "A scatter plot of effect sizes against their precision (usually 1/SE). Used to assess publication bias.",
        "interpretation": "Symmetric funnel shape suggests no bias. Asymmetry (missing studies in one corner) suggests possible publication bias.",
        "reference": "Sterne JAC, Egger M. Funnel plots for detecting bias in meta-analysis: Guidelines on choice of axis. Journal of Clinical Epidemiology. 2001;54(10):1046-1055."
    },
    "simpsons_paradox": {
        "name": "Simpson's Paradox",
        "definition": "A phenomenon where an association present in aggregate data reverses or disappears within subgroups.",
        "interpretation": "If overall and subgroup effects have opposite directions, the stratification variable is likely a confounder. Report stratified results.",
        "reference": "Simpson EH. The interpretation of interaction in contingency tables. Journal of the Royal Statistical Society: Series B. 1951;13(2):238-241."
    },
    "random_effects": {
        "name": "Random Effects Model",
        "definition": "Assumes studies estimate different but related true effects from a distribution. Accounts for between-study heterogeneity.",
        "interpretation": "More conservative than fixed effects. Gives more weight to smaller studies. Appropriate when heterogeneity exists.",
        "reference": "DerSimonian R, Laird N. Meta-analysis in clinical trials. Controlled Clinical Trials. 1986;7(3):177-188."
    },
    "fixed_effects": {
        "name": "Fixed Effects Model",
        "definition": "Assumes all studies estimate the same true effect. Weights studies by precision only.",
        "interpretation": "Appropriate when studies are very similar (low heterogeneity). Results apply to included studies specifically.",
        "reference": "Mantel N, Haenszel W. Statistical aspects of the analysis of data from retrospective studies of disease. JNCI. 1959;22(4):719-748."
    }
}


class MetaAnalysisRAG:
    """
    RAG-powered insights for meta-analysis.
    
    Provides natural language explanations, recommendations,
    and comparisons based on analysis results.
    """
    
    def __init__(self, llm_router=None, rag_system=None):
        """
        Initialize MetaAnalysisRAG.
        
        Args:
            llm_router: LLMRouter instance for generating responses
            rag_system: UnifiedRAGSystem for retrieving methodology docs
        """
        self.llm_router = llm_router
        self.rag_system = rag_system
        self.tooltips = STATISTICAL_TOOLTIPS
    
    def get_tooltip(self, term: str) -> Optional[Dict[str, Any]]:
        """Get tooltip information for a statistical term."""
        return self.tooltips.get(term.lower().replace(" ", "_").replace("²", "2"))
    
    def get_all_tooltips(self) -> Dict[str, Dict[str, Any]]:
        """Get all available tooltips."""
        return self.tooltips
    
    def enhance_statistical_results(
        self,
        meta_result: MetaAnalysisResult,
        bias_result: Optional[PublicationBiasResult] = None,
        sensitivity_result: Optional[SensitivityResult] = None
    ) -> str:
        """
        Generate plain-language explanation of meta-analysis results.
        
        Args:
            meta_result: MetaAnalysisResult from analysis
            bias_result: Publication bias results (optional)
            sensitivity_result: Sensitivity analysis results (optional)
            
        Returns:
            Natural language explanation
        """
        parts = []
        
        # Main findings
        effect = meta_result.pooled_effect
        ci = meta_result.confidence_interval
        direction = "positive" if effect > 0 else "negative"
        significant = meta_result.p_value < 0.05
        
        parts.append("## Summary of Findings\n")
        parts.append(
            f"The pooled effect estimate is **{effect:.3f}** "
            f"(95% CI: {ci[0]:.3f} to {ci[1]:.3f}), "
            f"based on **{meta_result.n_studies} studies**.\n"
        )
        
        if significant:
            parts.append(
                f"This {direction} effect is **statistically significant** (p = {meta_result.p_value:.4f}).\n"
            )
        else:
            parts.append(
                f"The effect is **not statistically significant** (p = {meta_result.p_value:.3f}).\n"
            )
        
        # Heterogeneity interpretation
        i2 = meta_result.heterogeneity["I2"]
        parts.append("\n## Heterogeneity Assessment\n")
        
        if i2 < 25:
            parts.append(f"I² = {i2:.1f}% indicates **low heterogeneity**. Studies show consistent effects.\n")
        elif i2 < 50:
            parts.append(f"I² = {i2:.1f}% indicates **moderate heterogeneity**. Some variation between studies exists.\n")
        elif i2 < 75:
            parts.append(f"I² = {i2:.1f}% indicates **substantial heterogeneity**. Consider exploring sources of variation.\n")
        else:
            parts.append(f"I² = {i2:.1f}% indicates **high heterogeneity**. True effects likely vary considerably across studies.\n")
        
        # Publication bias
        if bias_result:
            parts.append("\n## Publication Bias Assessment\n")
            if bias_result.bias_detected:
                parts.append(f"⚠️ **Publication bias {bias_result.bias_severity}**: {bias_result.interpretation}\n")
                if bias_result.adjusted_effect is not None:
                    parts.append(f"Bias-adjusted estimate: {bias_result.adjusted_effect:.3f}\n")
            else:
                parts.append("No clear evidence of publication bias detected.\n")
        
        # Sensitivity
        if sensitivity_result:
            parts.append("\n## Sensitivity Analysis\n")
            if sensitivity_result.robust:
                parts.append("✓ Results appear **robust** to removal of individual studies.\n")
            else:
                parts.append("⚠️ Results may be **sensitive** to individual studies.\n")
            if sensitivity_result.influential_studies:
                parts.append(f"Influential studies: {', '.join(sensitivity_result.influential_studies)}\n")
        
        return "".join(parts)
    
    def generate_recommendations(
        self,
        meta_result: MetaAnalysisResult,
        bias_result: Optional[PublicationBiasResult] = None,
        sensitivity_result: Optional[SensitivityResult] = None,
        benchmark_data: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations based on analysis.
        
        Args:
            meta_result: MetaAnalysisResult
            bias_result: Publication bias results
            sensitivity_result: Sensitivity analysis results
            benchmark_data: Historical benchmark data
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Heterogeneity recommendations
        i2 = meta_result.heterogeneity["I2"]
        
        if i2 > 75:
            recommendations.append({
                "type": "heterogeneity",
                "priority": "high",
                "title": "High Heterogeneity Detected",
                "message": "Consider stratified analysis or meta-regression to identify sources of variation.",
                "action": "Investigate moderator variables that may explain between-study differences.",
                "rationale": f"I² = {i2:.1f}% suggests true effects vary substantially across studies."
            })
        elif i2 > 50:
            recommendations.append({
                "type": "heterogeneity",
                "priority": "medium",
                "title": "Substantial Heterogeneity",
                "message": "Random effects model is appropriate. Prediction intervals may be more informative than confidence intervals.",
                "action": "Report both the pooled estimate and the prediction interval.",
                "rationale": f"I² = {i2:.1f}% indicates considerable between-study variation."
            })
        
        # Publication bias recommendations
        if bias_result:
            if bias_result.bias_severity in ["likely", "severe"]:
                recommendations.append({
                    "type": "publication_bias",
                    "priority": "high",
                    "title": "Publication Bias Concerns",
                    "message": "The pooled effect may be overestimated due to publication bias.",
                    "action": "Report both original and bias-adjusted estimates. Consider the adjusted estimate for conclusions.",
                    "rationale": bias_result.interpretation
                })
            elif bias_result.bias_severity == "possible":
                recommendations.append({
                    "type": "publication_bias",
                    "priority": "medium",
                    "title": "Possible Publication Bias",
                    "message": "Some indicators suggest potential publication bias.",
                    "action": "Interpret findings with appropriate caution.",
                    "rationale": "Funnel plot asymmetry detected but not conclusive."
                })
        
        # Sensitivity recommendations
        if sensitivity_result:
            if not sensitivity_result.robust:
                recommendations.append({
                    "type": "sensitivity",
                    "priority": "high",
                    "title": "Sensitivity Concerns",
                    "message": "Results depend heavily on specific studies.",
                    "action": "Report results with and without influential studies.",
                    "rationale": sensitivity_result.interpretation
                })
            
            if sensitivity_result.influential_studies:
                recommendations.append({
                    "type": "influential_studies",
                    "priority": "medium",
                    "title": "Influential Studies Identified",
                    "message": f"{len(sensitivity_result.influential_studies)} study(ies) substantially influence results.",
                    "action": "Examine these studies for methodological quality.",
                    "rationale": f"Studies: {', '.join(sensitivity_result.influential_studies)}"
                })
        
        # Sample size recommendations
        if meta_result.n_studies < 10:
            recommendations.append({
                "type": "sample_size",
                "priority": "medium",
                "title": "Small Number of Studies",
                "message": "Meta-analysis includes fewer than 10 studies.",
                "action": "Publication bias tests have limited power. Interpret Egger's test cautiously.",
                "rationale": f"Only {meta_result.n_studies} studies included."
            })
        
        # Significance recommendations
        if 0.01 < meta_result.p_value < 0.05:
            recommendations.append({
                "type": "significance",
                "priority": "low",
                "title": "Marginal Significance",
                "message": "P-value is close to 0.05 threshold.",
                "action": "Focus on effect size and confidence interval rather than p-value alone.",
                "rationale": f"p = {meta_result.p_value:.3f} is borderline significant."
            })
        
        # Benchmark comparison recommendations
        if benchmark_data:
            if "historical_effect" in benchmark_data:
                hist_effect = benchmark_data["historical_effect"]
                current_effect = meta_result.pooled_effect
                diff = abs(current_effect - hist_effect)
                
                if diff > 0.2:
                    recommendations.append({
                        "type": "benchmark",
                        "priority": "medium",
                        "title": "Deviation from Historical Benchmark",
                        "message": f"Current effect ({current_effect:.3f}) differs from historical average ({hist_effect:.3f}).",
                        "action": "Investigate whether methodological or contextual differences explain this deviation.",
                        "rationale": f"Difference of {diff:.3f} may be meaningful."
                    })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return recommendations
    
    def suggest_analysis_approaches(
        self,
        n_studies: int,
        heterogeneity_level: str,
        has_subgroups: bool,
        has_moderators: bool
    ) -> List[str]:
        """
        Suggest appropriate analysis approaches based on data characteristics.
        
        Args:
            n_studies: Number of studies
            heterogeneity_level: "low", "moderate", "substantial", "high"
            has_subgroups: Whether subgroup data is available
            has_moderators: Whether moderator variables are available
            
        Returns:
            List of recommended approaches
        """
        suggestions = []
        
        # Model selection
        if heterogeneity_level in ["low"]:
            suggestions.append("Fixed effects model is appropriate for homogeneous studies.")
        else:
            suggestions.append("Random effects model recommended due to heterogeneity.")
        
        # Heterogeneity investigation
        if heterogeneity_level in ["substantial", "high"]:
            if has_subgroups:
                suggestions.append("Subgroup analysis recommended to explore heterogeneity sources.")
            if has_moderators:
                suggestions.append("Meta-regression can quantify moderator effects on heterogeneity.")
        
        # Publication bias
        if n_studies >= 10:
            suggestions.append("Sufficient studies for Egger's test publication bias assessment.")
        else:
            suggestions.append("Consider funnel plot visual inspection; formal tests have low power with few studies.")
        
        # Sensitivity analysis
        suggestions.append("Leave-one-out analysis recommended to check result robustness.")
        
        if n_studies >= 5:
            suggestions.append("Cumulative meta-analysis can show how evidence evolved over time.")
        
        return suggestions
    
    def benchmark_against_history(
        self,
        current_result: MetaAnalysisResult,
        historical_data: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Compare current meta-analysis against historical benchmarks.
        
        Args:
            current_result: Current MetaAnalysisResult
            historical_data: List of historical analysis results
            
        Returns:
            Comparison results
        """
        comparison = {
            "current_effect": current_result.pooled_effect,
            "current_i2": current_result.heterogeneity["I2"],
            "benchmarks": [],
            "interpretation": ""
        }
        
        if not historical_data:
            comparison["interpretation"] = "No historical data available for comparison."
            return comparison
        
        # Compare against historical data
        historical_effects = [h.get("pooled_effect", 0) for h in historical_data if "pooled_effect" in h]
        
        if historical_effects:
            avg_historical = sum(historical_effects) / len(historical_effects)
            comparison["historical_average"] = avg_historical
            comparison["effect_difference"] = current_result.pooled_effect - avg_historical
            
            if abs(comparison["effect_difference"]) < 0.1:
                comparison["interpretation"] = "Current results are consistent with historical findings."
            elif comparison["effect_difference"] > 0:
                comparison["interpretation"] = "Current effect is larger than historical average."
            else:
                comparison["interpretation"] = "Current effect is smaller than historical average."
        
        return comparison
    
    def interpret_statistic(
        self,
        stat_name: str,
        value: float,
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate interpretation for a specific statistic.
        
        Args:
            stat_name: Name of the statistic (e.g., "I2", "tau2", "p_value")
            value: Value of the statistic
            context: Additional context (metric type, etc.)
            
        Returns:
            Natural language interpretation
        """
        interpretations = {
            "i2": self._interpret_i2,
            "tau2": self._interpret_tau2,
            "p_value": self._interpret_pvalue,
            "eggers_p": self._interpret_eggers,
            "q": self._interpret_q
        }
        
        interpreter = interpretations.get(stat_name.lower().replace("²", "2"))
        if interpreter:
            return interpreter(value, context or {})
        
        return f"{stat_name} = {value}"
    
    def _interpret_i2(self, value: float, context: Dict) -> str:
        """Interpret I² statistic."""
        if value < 25:
            level = "low"
            meaning = "Studies show consistent effects."
        elif value < 50:
            level = "moderate"
            meaning = "Some variation between studies exists."
        elif value < 75:
            level = "substantial"
            meaning = "Considerable variation exists. Random effects model appropriate."
        else:
            level = "high"
            meaning = "True effects vary substantially. Consider subgroup analysis."
        
        return f"I² = {value:.1f}% indicates **{level} heterogeneity**. {meaning}"
    
    def _interpret_tau2(self, value: float, context: Dict) -> str:
        """Interpret τ² statistic."""
        tau = value ** 0.5
        return (
            f"τ² = {value:.4f} (τ = {tau:.3f}). "
            f"This represents the estimated variance of true effects across studies."
        )
    
    def _interpret_pvalue(self, value: float, context: Dict) -> str:
        """Interpret p-value."""
        if value < 0.001:
            return f"p < 0.001: Highly significant. Strong evidence against null hypothesis."
        elif value < 0.01:
            return f"p = {value:.3f}: Significant. Good evidence against null hypothesis."
        elif value < 0.05:
            return f"p = {value:.3f}: Marginally significant. Some evidence, but borderline."
        elif value < 0.10:
            return f"p = {value:.3f}: Not significant, but approaching significance."
        else:
            return f"p = {value:.3f}: Not statistically significant."
    
    def _interpret_eggers(self, value: float, context: Dict) -> str:
        """Interpret Egger's test p-value."""
        if value < 0.05:
            return f"Egger's test p = {value:.3f}: Significant asymmetry detected. Publication bias possible."
        elif value < 0.10:
            return f"Egger's test p = {value:.3f}: Borderline asymmetry. Consider cautiously."
        else:
            return f"Egger's test p = {value:.3f}: No significant asymmetry detected."
    
    def _interpret_q(self, value: float, context: Dict) -> str:
        """Interpret Q statistic."""
        df = context.get("df", "unknown")
        return f"Q = {value:.2f} (df = {df}): Measures total heterogeneity in the meta-analysis."
