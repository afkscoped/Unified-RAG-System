"""
Web Enhancer: Provides web search capabilities via DuckDuckGo.
Wraps the functionality to appear as a structured tool for the system.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from loguru import logger
from .config import config

@dataclass
class SearchResult:
    """Unified structure for search results (Doc or Web)"""
    content: str
    source: str
    url: Optional[str] = None
    score: float = 0.0
    type: str = 'web'  # 'document' or 'web'
    metadata: Dict[str, Any] = None

class WebEnhancer:
    """
    Wrapper for DuckDuckGo Search to enable web capabilities 
    without heavy local models or complex API setups.
    """
    
    def __init__(self):
        self.enabled = config.enable_web_search
        self.ddgs = None
        if self.enabled:
            try:
                from duckduckgo_search import DDGS
                self.ddgs = DDGS()
                logger.info("WebEnhancer initialized with DuckDuckGo")
            except ImportError:
                logger.error("duckduckgo-search not installed. Web search disabled.")
                self.enabled = False

    def search(self, query: str) -> List[SearchResult]:
        """Execute web search and return unified SearchResult objects"""
        if not self.enabled or not self.ddgs:
            return []
            
        try:
            logger.info(f"Searching web for: {query}")
            # Use text search with configured max results
            results = self.ddgs.text(
                query, 
                max_results=config.max_web_results
            )
            
            # Convert to unified format
            enhanced_results = []
            for i, res in enumerate(results):
                # DDGS returns dict with 'title', 'href', 'body'
                enhanced_results.append(SearchResult(
                    content=res.get('body', ''),
                    source=res.get('title', 'Web Result'),
                    url=res.get('href', ''),
                    score=0.95 - (i * 0.05),  # Artificial decaying relevance
                    type='web',
                    metadata={'raw': res}
                ))
                
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            if config.fallback_to_local_on_error:
                return []
            raise e
