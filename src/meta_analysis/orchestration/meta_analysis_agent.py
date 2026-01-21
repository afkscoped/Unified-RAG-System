"""
Meta-Analysis Agent

LangGraph-based orchestration for meta-analysis workflow.
Provides a state machine for running complete analysis pipelines.
"""

from typing import Dict, List, Optional, Any, TypedDict, Annotated
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger

from src.meta_analysis.mcp_servers.base_experiment_mcp import StandardizedStudy
from src.meta_analysis.mcp_servers.csv_experiment_server import CSVExperimentMCP
from src.meta_analysis.statistical.data_harmonizer import DataHarmonizer
from src.meta_analysis.statistical.meta_analyzer import MetaAnalyzer, MetaAnalysisResult
from src.meta_analysis.statistical.publication_bias import PublicationBiasDetector
from src.meta_analysis.statistical.simpsons_detector import SimpsonsParadoxDetector
from src.meta_analysis.statistical.sensitivity_analysis import SensitivityAnalyzer
from src.meta_analysis.statistical.visualization import MetaAnalysisVisualizer
from src.meta_analysis.rag.meta_analysis_rag import MetaAnalysisRAG


class AnalysisStage(Enum):
    """Stages of the meta-analysis workflow."""
    INIT = "init"
    PARSE_QUERY = "parse_query"
    COLLECT_DATA = "collect_data"
    HARMONIZE = "harmonize"
    ANALYZE = "analyze"
    DETECT_BIAS = "detect_bias"
    SENSITIVITY = "sensitivity"
    RAG_ENHANCE = "rag_enhance"
    REPORT = "report"
    COMPLETE = "complete"
    ERROR = "error"

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            ordering = [
                "init",
                "parse_query",
                "collect_data",
                "harmonize",
                "analyze",
                "detect_bias",
                "sensitivity",
                "rag_enhance",
                "report",
                "complete",
                "error"
            ]
            try:
                return ordering.index(self.value) < ordering.index(other.value)
            except ValueError:
                return False
        return NotImplemented

    def __le__(self, other):
        return self < other or self == other


@dataclass
class MetaAnalysisState:
    """
    State machine for meta-analysis workflow.
    
    Tracks all data and results through the analysis pipeline.
    """
    # Input
    user_query: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Parsed intent
    parsed_intent: Dict[str, Any] = field(default_factory=dict)
    experiment_filters: Dict[str, Any] = field(default_factory=dict)
    
    # Data collection
    experiment_sources: List[str] = field(default_factory=list)
    raw_experiments: List[Dict] = field(default_factory=list)
    
    # Processed data
    standardized_studies: List[StandardizedStudy] = field(default_factory=list)
    quality_checks: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis results
    meta_result: Optional[MetaAnalysisResult] = None
    heterogeneity_stats: Dict[str, float] = field(default_factory=dict)
    publication_bias_result: Optional[Dict] = None
    sensitivity_result: Optional[Dict] = None
    simpsons_result: Optional[Dict] = None
    
    # RAG enhancements
    rag_context: str = ""
    recommendations: List[Dict] = field(default_factory=list)
    benchmark_comparison: Dict[str, Any] = field(default_factory=dict)
    
    # Outputs
    visualizations: Dict[str, Any] = field(default_factory=dict)
    final_report: str = ""
    
    # Workflow tracking
    current_stage: AnalysisStage = AnalysisStage.INIT
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class MetaAnalysisAgent:
    """
    Orchestrates the meta-analysis workflow.
    
    Manages the state machine and executes each stage of analysis.
    """
    
    def __init__(
        self,
        csv_mcp: Optional[CSVExperimentMCP] = None,
        llm_bridge=None,
        rag_bridge=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the agent.
        
        Args:
            csv_mcp: CSV experiment MCP server
            llm_bridge: LLM bridge for AI features
            rag_bridge: RAG bridge for knowledge retrieval
            config: Configuration dictionary
        """
        self.csv_mcp = csv_mcp or CSVExperimentMCP()
        self.llm_bridge = llm_bridge
        self.rag_bridge = rag_bridge
        self.config = config or {}
        
        # Initialize components
        self.harmonizer = DataHarmonizer()
        self.meta_analyzer = MetaAnalyzer()
        self.bias_detector = PublicationBiasDetector()
        self.simpsons_detector = SimpsonsParadoxDetector()
        self.sensitivity_analyzer = SensitivityAnalyzer()
        self.visualizer = MetaAnalysisVisualizer()
        self.meta_rag = MetaAnalysisRAG()
    
    def run(
        self,
        query: str = "",
        studies: Optional[List[StandardizedStudy]] = None,
        config: Optional[Dict] = None
    ) -> MetaAnalysisState:
        """
        Run the complete meta-analysis workflow.
        
        Args:
            query: User query or analysis description
            studies: Pre-loaded studies (optional)
            config: Runtime configuration
            
        Returns:
            Final MetaAnalysisState with all results
        """
        # Initialize state
        state = MetaAnalysisState(
            user_query=query,
            config={**self.config, **(config or {})},
            started_at=datetime.now()
        )
        
        # If studies provided, skip collection
        if studies:
            state.standardized_studies = studies
            state.current_stage = AnalysisStage.HARMONIZE
        
        try:
            # Run workflow stages
            state = self._run_workflow(state)
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            state.errors.append(str(e))
            state.current_stage = AnalysisStage.ERROR
        
        state.completed_at = datetime.now()
        return state
    
    def _run_workflow(self, state: MetaAnalysisState) -> MetaAnalysisState:
        """Execute workflow stages."""
        
        # Stage: Parse Query
        if state.current_stage <= AnalysisStage.PARSE_QUERY:
            state = self._parse_query(state)
        
        # Stage: Collect Data
        if state.current_stage <= AnalysisStage.COLLECT_DATA and not state.standardized_studies:
            state = self._collect_data(state)
        
        # Stage: Harmonize
        if state.current_stage <= AnalysisStage.HARMONIZE:
            state = self._harmonize_data(state)
        
        # Check minimum studies
        if len(state.standardized_studies) < 2:
            state.errors.append("Need at least 2 studies for meta-analysis")
            state.current_stage = AnalysisStage.ERROR
            return state
        
        # Stage: Analyze
        if state.current_stage <= AnalysisStage.ANALYZE:
            state = self._run_analysis(state)
        
        # Stage: Detect Bias
        if state.current_stage <= AnalysisStage.DETECT_BIAS:
            state = self._detect_bias(state)
        
        # Stage: Sensitivity
        if state.current_stage <= AnalysisStage.SENSITIVITY:
            state = self._run_sensitivity(state)
        
        # Stage: RAG Enhancement
        if state.current_stage <= AnalysisStage.RAG_ENHANCE:
            state = self._enhance_with_rag(state)
        
        # Stage: Generate Report
        if state.current_stage <= AnalysisStage.REPORT:
            state = self._generate_report(state)
        
        state.current_stage = AnalysisStage.COMPLETE
        return state
    
    def _parse_query(self, state: MetaAnalysisState) -> MetaAnalysisState:
        """Parse user query to extract intent and filters."""
        state.current_stage = AnalysisStage.PARSE_QUERY
        
        # Simple query parsing (can be enhanced with LLM)
        query_lower = state.user_query.lower()
        
        parsed = {
            "run_meta_analysis": True,
            "check_publication_bias": True,
            "check_sensitivity": True,
            "model_type": "auto"
        }
        
        # Extract model preference
        if "fixed effect" in query_lower:
            parsed["model_type"] = "fixed"
        elif "random effect" in query_lower:
            parsed["model_type"] = "random"
        
        # Extract filters
        filters = {}
        if "minimum" in query_lower and "sample" in query_lower:
            # Try to extract sample size
            import re
            numbers = re.findall(r'\d+', query_lower)
            if numbers:
                filters["min_sample_size"] = int(numbers[0])
        
        state.parsed_intent = parsed
        state.experiment_filters = filters
        
        return state
    
    def _collect_data(self, state: MetaAnalysisState) -> MetaAnalysisState:
        """Collect experiments from MCP sources."""
        state.current_stage = AnalysisStage.COLLECT_DATA
        
        try:
            # Get experiments from CSV MCP
            experiments = self.csv_mcp.get_experiments(
                filters=None  # Could use state.experiment_filters
            )
            state.standardized_studies = experiments
            state.experiment_sources = ["csv"]
            
            logger.info(f"Collected {len(experiments)} experiments")
        except Exception as e:
            logger.warning(f"Data collection failed: {e}")
            state.warnings.append(f"Data collection issue: {e}")
        
        return state
    
    def _harmonize_data(self, state: MetaAnalysisState) -> MetaAnalysisState:
        """Harmonize and validate data."""
        state.current_stage = AnalysisStage.HARMONIZE
        
        # Harmonize effect sizes
        harmonized = self.harmonizer.harmonize_studies(state.standardized_studies)
        
        # Validate
        validation = self.csv_mcp.validate_data(harmonized)
        state.quality_checks = validation.to_dict()
        
        # Keep only valid studies
        state.standardized_studies = harmonized
        
        if validation.warnings:
            state.warnings.extend([f"Validation: {w.message}" for w in validation.warnings])
        
        logger.info(f"Harmonized {len(harmonized)} studies")
        return state
    
    def _run_analysis(self, state: MetaAnalysisState) -> MetaAnalysisState:
        """Run meta-analysis."""
        state.current_stage = AnalysisStage.ANALYZE
        
        model_type = state.parsed_intent.get("model_type", "auto")
        
        # Run meta-analysis
        result = self.meta_analyzer.analyze(
            studies=state.standardized_studies,
            model=model_type
        )
        
        state.meta_result = result
        state.heterogeneity_stats = result.heterogeneity
        
        # Generate visualizations
        try:
            forest_fig = self.visualizer.create_forest_plot(result)
            state.visualizations["forest_plot"] = forest_fig.to_json()
            
            # New: Heterogeneity Chart
            het_fig = self.visualizer.create_heterogeneity_chart(result)
            state.visualizations["heterogeneity_chart"] = het_fig.to_json()
            
            # New: Bubble Plot (needs > 2 studies)
            if len(state.standardized_studies) > 2:
                bubble_fig = self.visualizer.create_bubble_plot(result)
                state.visualizations["bubble_plot"] = bubble_fig.to_json()
                
        except Exception as e:
            logger.warning(f"Visualization generation failed in analysis stage: {e}")
        
        logger.info(f"Meta-analysis complete: effect={result.pooled_effect:.4f}")
        return state
    
    def _detect_bias(self, state: MetaAnalysisState) -> MetaAnalysisState:
        """Detect publication bias."""
        state.current_stage = AnalysisStage.DETECT_BIAS
        
        if len(state.standardized_studies) < 3:
            state.warnings.append("Too few studies for publication bias analysis")
            return state
        
        try:
            bias_result = self.bias_detector.analyze(
                studies=state.standardized_studies,
                meta_result=state.meta_result
            )
            state.publication_bias_result = bias_result.to_dict()
            
            # Generate funnel plot
            try:
                funnel_fig = self.visualizer.create_funnel_plot(state.meta_result)
                state.visualizations["funnel_plot"] = funnel_fig.to_json()
            except Exception as e:
                logger.warning(f"Funnel plot failed: {e}")
            
            logger.info(f"Bias detection complete: severity={bias_result.bias_severity}")
        except Exception as e:
            logger.warning(f"Bias detection failed: {e}")
            state.warnings.append(f"Bias detection issue: {e}")
        
        return state
    
    def _run_sensitivity(self, state: MetaAnalysisState) -> MetaAnalysisState:
        """Run sensitivity analysis."""
        state.current_stage = AnalysisStage.SENSITIVITY
        
        if len(state.standardized_studies) < 3:
            state.warnings.append("Too few studies for sensitivity analysis")
            return state
        
        try:
            sensitivity = self.sensitivity_analyzer.analyze(
                studies=state.standardized_studies,
                original_result=state.meta_result
            )
            state.sensitivity_result = sensitivity.to_dict()
            
            # New: Influence Plot
            try:
                influence_fig = self.visualizer.create_influence_plot(sensitivity)
                state.visualizations["influence_plot"] = influence_fig.to_json()
            except Exception as e:
                logger.warning(f"Influence plot generation failed: {e}")
            
            logger.info(f"Sensitivity analysis complete: robust={sensitivity.robust}")
        except Exception as e:
            logger.warning(f"Sensitivity analysis failed: {e}")
            state.warnings.append(f"Sensitivity analysis issue: {e}")
        
        return state
    
    def _enhance_with_rag(self, state: MetaAnalysisState) -> MetaAnalysisState:
        """Enhance results with RAG insights."""
        state.current_stage = AnalysisStage.RAG_ENHANCE
        
        # Generate interpretation
        from src.meta_analysis.statistical.publication_bias import PublicationBiasResult
        from src.meta_analysis.statistical.sensitivity_analysis import SensitivityResult
        
        bias_obj = None
        if state.publication_bias_result:
            # Reconstruct object for RAG
            bias_obj = type('BiasResult', (), state.publication_bias_result)()
        
        sensitivity_obj = None
        if state.sensitivity_result:
            sensitivity_obj = type('SensitivityResult', (), state.sensitivity_result)()
        
        # Get enhanced interpretation
        state.rag_context = self.meta_rag.enhance_statistical_results(
            meta_result=state.meta_result,
            bias_result=bias_obj,
            sensitivity_result=sensitivity_obj
        )
        
        # Generate recommendations
        state.recommendations = self.meta_rag.generate_recommendations(
            meta_result=state.meta_result,
            bias_result=bias_obj,
            sensitivity_result=sensitivity_obj
        )
        
        return state
    
    def _generate_report(self, state: MetaAnalysisState) -> MetaAnalysisState:
        """Generate final report."""
        state.current_stage = AnalysisStage.REPORT
        
        report_parts = []
        
        # Header
        report_parts.append("# Meta-Analysis Report\n")
        report_parts.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
        
        # Summary
        if state.meta_result:
            report_parts.append("\n## Summary\n")
            report_parts.append(f"- **Pooled Effect**: {state.meta_result.pooled_effect:.4f}\n")
            report_parts.append(f"- **95% CI**: ({state.meta_result.confidence_interval[0]:.4f}, "
                              f"{state.meta_result.confidence_interval[1]:.4f})\n")
            report_parts.append(f"- **P-value**: {state.meta_result.p_value:.4f}\n")
            report_parts.append(f"- **Studies**: {state.meta_result.n_studies}\n")
            report_parts.append(f"- **Model**: {state.meta_result.model_type}\n")
        
        # Heterogeneity
        if state.heterogeneity_stats:
            report_parts.append("\n## Heterogeneity\n")
            report_parts.append(f"- **I²**: {state.heterogeneity_stats.get('I2', 0):.1f}%\n")
            report_parts.append(f"- **τ²**: {state.heterogeneity_stats.get('tau2', 0):.4f}\n")
        
        # RAG insights
        if state.rag_context:
            report_parts.append("\n## Interpretation\n")
            report_parts.append(state.rag_context)
        
        # Recommendations
        if state.recommendations:
            report_parts.append("\n## Recommendations\n")
            for rec in state.recommendations[:5]:
                report_parts.append(f"- **{rec.get('title', 'Recommendation')}**: {rec.get('message', '')}\n")
        
        # Warnings
        if state.warnings:
            report_parts.append("\n## Warnings\n")
            for warning in state.warnings:
                report_parts.append(f"- {warning}\n")
        
        state.final_report = "".join(report_parts)
        
        return state
    
    def get_results_summary(self, state: MetaAnalysisState) -> Dict[str, Any]:
        """Get a condensed summary of results."""
        summary = {
            "status": state.current_stage.value,
            "n_studies": len(state.standardized_studies),
            "completed": state.current_stage == AnalysisStage.COMPLETE,
            "errors": state.errors,
            "warnings": state.warnings
        }
        
        if state.meta_result:
            summary["meta_analysis"] = {
                "pooled_effect": state.meta_result.pooled_effect,
                "ci_lower": state.meta_result.confidence_interval[0],
                "ci_upper": state.meta_result.confidence_interval[1],
                "p_value": state.meta_result.p_value,
                "model": state.meta_result.model_type,
                "i2": state.heterogeneity_stats.get("I2", 0),
                "significant": state.meta_result.p_value < 0.05
            }
        
        if state.publication_bias_result:
            summary["publication_bias"] = {
                "detected": state.publication_bias_result.get("bias_detected", False),
                "severity": state.publication_bias_result.get("bias_severity", "unknown")
            }
        
        if state.sensitivity_result:
            summary["sensitivity"] = {
                "robust": state.sensitivity_result.get("robust", True),
                "influential_studies": state.sensitivity_result.get("influential_studies", [])
            }
        
        return summary
