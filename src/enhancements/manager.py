import time
from typing import Optional, Dict, Any, List
from loguru import logger

from src.core.rag_system import UnifiedRAGSystem
from .config import config
from .persona_engine import PersonaEngine
from .web_enhancer import WebEnhancer, SearchResult
from .synthesizer import ResultSynthesizer
from .evaluator import EvaluationEngine

class EnhancedAnalyzer:
    """
    Orchestrates the interaction between:
    1. Local RAG (UnifiedRAGSystem)
    2. Web Search (WebEnhancer)
    3. Persona Logic (PersonaEngine)
    4. Evaluation Logic (EvaluationEngine)
    """
    
    def __init__(self, rag_system: UnifiedRAGSystem):
        self.rag = rag_system
        
        # Initialize enhancements if enabled
        self.persona_engine = PersonaEngine(self.rag.llm_router) if config.enable_personas else None
        self.web_enhancer = WebEnhancer() if config.enable_web_search else None
        self.evaluation_engine = EvaluationEngine(self.rag.llm_router)
        self.synthesizer = ResultSynthesizer()
        
        # In-memory session stats for visualization
        self.query_history = []
        
    def query(
        self,
        question: str,
        search_mode: str = 'docs',
        persona_id: str = 'scientist',
        top_k: int = 5,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Unified query method handling all search modes.
        Returns a dict compatible with the UI expectation, but enhanced.
        """
        
        # 1. Local Search (if applicable)
        local_results: List[SearchResult] = []
        rag_answer = ""
        
        if search_mode != 'web':
            try:
                # OPTIMIZATION: If personas are enabled, we only need retrieval, not generation.
                # This saves 1 full LLM call (approx 1-3 seconds).
                do_generate = not config.enable_personas
                
                # Call core system
                core_result = self.rag.query(
                    question, 
                    top_k=top_k, 
                    use_cache=use_cache,
                    generate_answer=do_generate
                )
                rag_answer = core_result.answer
                
                # Adapt sources
                if core_result.sources:
                    for s in core_result.sources:
                        local_results.append(SearchResult(
                            content=s.content,
                            source=s.metadata.get('source_file', 'Local Doc'),
                            score=s.score,
                            type='document',
                            metadata=s.metadata
                        ))
            except Exception as e:
                logger.error(f"Local search failed: {e}")
                
        # 2. Web Search (if applicable)
        web_results: List[SearchResult] = []
        if search_mode != 'docs' and self.web_enhancer and self.web_enhancer.enabled:
            # Check config/logic if we should search
            try:
                web_results = self.web_enhancer.search(question)
            except Exception as e:
                logger.error(f"Web search step failed: {e}")

        # 3. Synthesis
        merged_results = self.synthesizer.merge(local_results, web_results, search_mode)
        
        # 4. Persona Generation (Final Answer)
        final_answer = rag_answer # Might be empty if do_generate was False
        
        if self.persona_engine and config.enable_personas:
            context = self.synthesizer.format_context(merged_results)
            if context.strip():
                try:
                    persona_response = self.persona_engine.generate_response(
                        query=question,
                        context=context,
                        persona_id=persona_id
                    )
                    if persona_response and not persona_response.startswith("Error"):
                        final_answer = persona_response
                    elif not final_answer: # Fallback if persona failed and no rag_answer
                        logger.warning("Persona failed and no rag_answer, doing basic generation")
                        final_answer = self.rag.llm_router.generate(
                            prompt=f"Question: {question}\nContext: {context}",
                            system_prompt="Provide a detailed answer based on the context."
                        )
                except Exception as e:
                    logger.error(f"Persona engine fail: {e}")
                    if not final_answer:
                        final_answer = f"Error generating persona response: {e}"
            elif search_mode == 'web':
                final_answer = "I couldn't find any relevant results on the web."
                
        elif search_mode == 'web' and not rag_answer:
            final_answer = " **Web Search Results:**\n\n" + "\n\n".join([r.content[:500] for r in web_results])
        
        # FINAL SAFETY: If somehow still empty
        if not final_answer:
            final_answer = "I'm sorry, I couldn't generate a response. Please try again or check your documents."

        # 5. Final Evaluation
        eval_metrics = {}
        if config.enable_personas: # Only evaluate if persona engine used for quality check
            context_str = self.synthesizer.format_context(merged_results)
            eval_metrics = self.evaluation_engine.evaluate(question, final_answer, context_str)
        
        # 6. Build final payload
        result = {
            "query": question,
            "answer": final_answer,
            "sources": merged_results,
            "mode": search_mode,
            "persona": persona_id,
            "metrics": eval_metrics
        }
        
        # Store in history for analytics
        self.query_history.append({
            "timestamp": time.time(),
            "query": question,
            "mode": search_mode,
            "persona": persona_id,
            "metrics": eval_metrics,
            "source_count": len(merged_results)
        })
        
        return result

    def get_corpus_data(self):
        """Pass-through for corpus data extraction"""
        return self.rag.get_corpus_data()

    def get_metrics(self):
        """Pass-through for metrics"""
        return self.rag.get_metrics()
