"""
Evaluation Engine: Grades RAG responses using LLM-as-a-Judge.
Calculates Faithfulness, Relevance, and Latency metrics.
"""

import time
import json
from typing import Dict, Any, List, Optional
from loguru import logger
from src.core.llm_router import LLMRouter

class EvaluationEngine:
    """
    Evaluates RAG results using light-weight LLM judgment (Groq).
    """
    
    def __init__(self, llm_router: Optional[LLMRouter] = None):
        self.llm_router = llm_router or LLMRouter()
        
    def evaluate(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        """
        Run a quick LLM-based evaluation on the response.
        Returns metrics in a structured format.
        """
        start_time = time.time()
        
        # Simple Judge Prompt
        judge_prompt = f"""
        Evaluate the following RAG response based on 3 criteria (0-10 scale):
        1. Faithfulness: Is the answer derived ONLY from the context? (No hallucinations)
        2. Relevance: Does the answer directly address the user query?
        3. Clarity: Is the response well-structured and easy to understand?

        [QUERY]
        {query}

        [CONTEXT]
        {context[:3000]} 

        [ANSWER]
        {answer}

        Return ONLY a JSON object with keys: faithfulness, relevance, clarity, and a brief reasoning string.
        Example: {{"faithfulness": 9, "relevance": 10, "clarity": 8, "reasoning": "Direct answer, grounded in docs."}}
        """
        
        metrics = {
            "faithfulness": 0.0,
            "relevance": 0.0,
            "clarity": 0.0,
            "reasoning": "Evaluation failed",
            "latency": 0.0
        }
        
        try:
            raw_eval = self.llm_router.generate(
                prompt=judge_prompt,
                system_prompt="You are a strict RAG Quality Auditor. Output ONLY valid JSON."
            )
            
            # Clean JSON if LLM added fluff
            json_str = raw_eval.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "{" in json_str:
                json_str = "{" + json_str.split("{", 1)[1].rsplit("}", 1)[0] + "}"
                
            parsed = json.loads(json_str)
            metrics.update(parsed)
            
        except Exception as e:
            logger.error(f"Evaluation judgment failed: {e}")
            metrics["reasoning"] = f"Error during evaluation: {str(e)}"
            
        metrics["latency"] = round((time.time() - start_time) * 1000, 2)
        return metrics

    def batch_compare(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize multiple evaluations for comparison"""
        if not evaluations:
            return {}
            
        avg_faith = sum(e.get('faithfulness', 0) for e in evaluations) / len(evaluations)
        avg_rel = sum(e.get('relevance', 0) for e in evaluations) / len(evaluations)
        avg_clarity = sum(e.get('clarity', 0) for e in evaluations) / len(evaluations)
        
        return {
            "avg_faithfulness": round(avg_faith, 2),
            "avg_relevance": round(avg_rel, 2),
            "avg_clarity": round(avg_clarity, 2),
            "count": len(evaluations)
        }
