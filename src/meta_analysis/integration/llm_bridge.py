"""
LLM Bridge

Connects meta-analysis engine to existing LLMRouter.
"""

from typing import Dict, Optional, Any
from loguru import logger


class MetaAnalysisLLMBridge:
    """
    Bridges meta-analysis to the existing LLM infrastructure.
    
    Provides specialized prompts for statistical interpretation
    and recommendations.
    """
    
    # System prompts for different tasks
    SYSTEM_PROMPTS = {
        "interpretation": """You are a statistical expert specializing in meta-analysis.
Your task is to explain statistical results in clear, accessible language.
Always be accurate about what the statistics mean and avoid overstating conclusions.
When explaining heterogeneity, publication bias, or other concepts, use analogies when helpful.""",
        
        "recommendation": """You are an experimentation advisor helping teams make decisions based on meta-analysis results.
Provide actionable recommendations that are practical and evidence-based.
Consider the limitations of the data and be appropriately cautious.
Prioritize recommendations by importance.""",
        
        "report": """You are a technical writer creating meta-analysis reports.
Write in a professional but accessible style.
Include appropriate caveats and limitations.
Structure content logically with clear headings.""",
        
        "query_parser": """You are parsing user queries about A/B test meta-analysis.
Extract the intent, filters, and parameters from the user's question.
Output should indicate what type of analysis is requested and any constraints."""
    }
    
    def __init__(self, llm_router=None):
        """
        Initialize LLM bridge.
        
        Args:
            llm_router: Existing LLMRouter instance (optional)
        """
        self.llm_router = llm_router
        self._initialized = False
    
    def initialize(self):
        """Initialize the bridge."""
        if self.llm_router is None:
            logger.warning("No LLM router provided - operating in template mode")
        self._initialized = True
        logger.info("MetaAnalysisLLMBridge initialized")
    
    def generate(
        self,
        prompt: str,
        task_type: str = "interpretation",
        provider: Optional[str] = None
    ) -> str:
        """
        Generate response using LLM.
        
        Args:
            prompt: User/task prompt
            task_type: Type of task (interpretation, recommendation, report, query_parser)
            provider: Override LLM provider (optional)
            
        Returns:
            Generated response
        """
        system_prompt = self.SYSTEM_PROMPTS.get(task_type, self.SYSTEM_PROMPTS["interpretation"])
        
        if self.llm_router:
            try:
                response = self.llm_router.generate(
                    prompt=prompt,
                    provider=provider,
                    system_prompt=system_prompt
                )
                return response
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                return self._fallback_response(task_type, prompt)
        else:
            return self._fallback_response(task_type, prompt)
    
    def _fallback_response(self, task_type: str, prompt: str) -> str:
        """Generate fallback response when LLM unavailable."""
        if task_type == "interpretation":
            return "Statistical interpretation is available when LLM is connected."
        elif task_type == "recommendation":
            return "Recommendations are available when LLM is connected."
        elif task_type == "report":
            return "Report generation is available when LLM is connected."
        else:
            return "Response pending LLM connection."
    
    def interpret_statistic(
        self,
        stat_name: str,
        value: float,
        context: Optional[Dict] = None
    ) -> str:
        """
        Get LLM interpretation of a statistic.
        
        Args:
            stat_name: Name of the statistic
            value: Value of the statistic
            context: Additional context
            
        Returns:
            Natural language interpretation
        """
        context_str = ""
        if context:
            context_str = f"\nContext: {context}"
        
        prompt = f"""Please interpret the following meta-analysis statistic:

Statistic: {stat_name}
Value: {value}{context_str}

Provide a clear, accessible explanation of what this means, including:
1. What this statistic measures
2. Whether this value is concerning or normal
3. Implications for interpreting the meta-analysis"""
        
        return self.generate(prompt, task_type="interpretation")
    
    def generate_executive_summary(
        self,
        meta_result: Dict[str, Any],
        bias_result: Optional[Dict] = None,
        sensitivity_result: Optional[Dict] = None
    ) -> str:
        """
        Generate executive summary for meta-analysis.
        
        Args:
            meta_result: Meta-analysis result as dict
            bias_result: Publication bias result
            sensitivity_result: Sensitivity analysis result
            
        Returns:
            Executive summary text
        """
        prompt = f"""Generate an executive summary for the following meta-analysis:

**Meta-Analysis Results:**
- Pooled Effect: {meta_result.get('pooled_effect', 'N/A')}
- 95% CI: ({meta_result.get('confidence_interval', {}).get('lower', 'N/A')}, {meta_result.get('confidence_interval', {}).get('upper', 'N/A')})
- P-value: {meta_result.get('p_value', 'N/A')}
- Number of Studies: {meta_result.get('n_studies', 'N/A')}
- I²: {meta_result.get('heterogeneity', {}).get('I2', 'N/A')}%

"""
        
        if bias_result:
            prompt += f"""**Publication Bias:**
- Bias Detected: {bias_result.get('bias_detected', 'Unknown')}
- Severity: {bias_result.get('bias_severity', 'Unknown')}

"""
        
        if sensitivity_result:
            prompt += f"""**Sensitivity:**
- Robust: {sensitivity_result.get('robust', 'Unknown')}
- Influential Studies: {sensitivity_result.get('influential_studies', [])}

"""
        
        prompt += """Please write a brief (3-4 paragraph) executive summary covering:
1. The main finding and its significance
2. Key concerns or limitations
3. Practical implications"""
        
        return self.generate(prompt, task_type="report")
    
    def is_available(self) -> bool:
        """Check if LLM bridge is available."""
        return self._initialized and self.llm_router is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM router statistics if available."""
        if self.llm_router:
            return self.llm_router.get_stats()
        return {"status": "not_connected"}
