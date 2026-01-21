"""
Synthesizer: Merges results from local documents and web search.
Prepares unified context for the Persona Engine.
"""

from typing import List
from .web_enhancer import SearchResult

class ResultSynthesizer:
    """Combines outcomes from different retrieval strategies"""
    
    @staticmethod
    def merge(
        doc_results: List[SearchResult],
        web_results: List[SearchResult],
        mode: str = 'hybrid'
    ) -> List[SearchResult]:
        """
        Merge results based on the activve search mode.
        """
        if mode == 'docs':
            return doc_results
        elif mode == 'web':
            return web_results
        
        # Hybrid mode: Interleave or concatenate?
        # For now, we prefer Docs (Ground Truth) then Web (Enhancement)
        # But we sort by score first if possible.
        
        # Simple concatenation with Docs first is usually safer for RAG
        # to ground the answer in local data first.
        return doc_results + web_results

    @staticmethod
    def format_context(results: List[SearchResult]) -> str:
        """
        Formats the list of SearchResult objects into a string context
        for the LLM.
        """
        formatted = []
        for i, res in enumerate(results, 1):
            tag = "[LOCAL DOC]" if res.type == 'document' else "[WEB SEARCH]"
            snippet = res.content[:1500]  # Soft truncate to fit context
            
            entry = f"""
            --- SOURCE {i} {tag} ---
            Title: {res.source}
            URL: {res.url or 'N/A'}
            Content: {snippet}
            """
            formatted.append(entry)
            
        return "\n".join(formatted)
