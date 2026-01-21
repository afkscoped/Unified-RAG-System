"""
Dialectical Engine: Implements the 'Thesis-Antithesis-Synthesis' loop.
Uses local search for support and web search for counter-arguments.
"""

from typing import Dict, Any, List
from loguru import logger
from src.enhancements.web_enhancer import WebEnhancer

class DialecticalSynthesizer:
    """
    Orchestrates the adversarial debate logic.
    """
    
    def __init__(self, llm_router, search_engine):
        self.llm = llm_router
        self.search = search_engine
        self.web = WebEnhancer() # Re-use the existing web module
        
    def generate_debate(self, query: str, deep_audit: bool = False) -> Dict[str, Any]:
        """
        Runs the dialectical process:
        1. Thesis (Proponent): Based on Local Docs.
        2. Antithesis (Skeptic): Based on Web Search (if Deep Audit).
        3. Synthesis: Final adjudicated answer.
        """
        logger.info(f"Starting Dialectical Debate for: {query}")
        
        # 1. Thesis (Local Support)
        # Reuse existing search logic basically, but explicitly framed as Proponent
        # We assume 'search' is the HybridSearchEngine instance
        local_results = self.search.search_hybrid(query, k=5)
        local_context = "\n\n".join([f"Doc {i}: {r.content}" for i, r in enumerate(local_results)])
        
        thesis_prompt = f"""
        You are the PROPONENT. Argue FOR the following query based ONLY on the provided context.
        Query: {query}
        Context:
        {local_context}
        
        Output a concise, strong argument supporting the premise found in the docs.
        """
        thesis = self.llm.generate(thesis_prompt, system_prompt="You are a focused researcher finding supporting evidence.")
        
        antithesis = "Deep Audit disabled. No counter-argument generated."
        web_context = ""
        
        if deep_audit:
            # 2. Antithesis (Web Counter-Evidence)
            # Invert query to find criticism? Or just general search and ask LLM to find conflicts?
            # A simple inversion is adding "criticism", "problems", "alternatives"
            counter_query = f"{query} criticism problems counter-evidence"
            web_results = self.web.search(counter_query)
            web_context = "\n\n".join([f"Web {i}: {r.content}" for i, r in enumerate(web_results)])
            
            antithesis_prompt = f"""
            You are the SKEPTIC. Argue AGAINST the following query or provide alternative viewpoints based on the provided web context.
            Query: {query}
            Web Context:
            {web_context}
            
            Output a concise, critical argument highlighting limitations, errors, or alternative views.
            """
            antithesis = self.llm.generate(antithesis_prompt, system_prompt="You are a critical auditor looking for flaws and contradictions.")
            
        # 3. Synthesis
        synthesis_prompt = f"""
        You are the ADJUDICATOR. Synthesize a final answer based on the debate below.
        
        Query: {query}
        
        [THESIS - SUPPORTING EVIDENCE]
        {thesis}
        
        [ANTITHESIS - CONFLICTING EVIDENCE]
        {antithesis}
        
        Provide a balanced, comprehensive conclusion. Acknowledge the local data but qualify it with web insights if valid.
        """
        
        synthesis = self.llm.generate(synthesis_prompt, system_prompt="You are an unbiased judge synthesizing a final research conclusion.")
        
        return {
            "thesis": thesis,
            "antithesis": antithesis,
            "synthesis": synthesis,
            "local_refs": [r.metadata.get('source_file', 'unknown') for r in local_results],
            "is_debate": True
        }
