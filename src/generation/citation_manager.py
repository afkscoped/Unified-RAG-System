"""
[ELITE ARCHITECTURE] citation_manager.py
Manages mapping between generated tokens and source document metadata.
"""

from typing import List, Any, Dict
import re

class CitationManager:
    """
    Innovation: Truth Grounding.
    Scans generated text for [Source X] patterns and provides a detailed 
    metadata mapping for the UI to render tooltips/links.
    """
    
    @staticmethod
    def resolve_citations(text: str, sources: List[Any]) -> List[Dict[str, Any]]:
        """
        Extracts all citation tags from text and maps them to actual source objects.
        """
        # Find all patterns like [Source 1], [1], [Source 2]
        pattern = r"\[(?:Source\s*)?(\d+)\]"
        matches = re.findall(pattern, text)
        
        unique_indices = sorted(list(set([int(m) for m in matches])))
        
        resolved = []
        for idx in unique_indices:
            # Source indices in text are usually 1-based
            if 0 < idx <= len(sources):
                doc = sources[idx-1]
                resolved.append({
                    "id": idx,
                    "file": doc.metadata.get("source_file", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "snippet": doc.content[:200] + "..."
                })
                
        return resolved

if __name__ == "__main__":
    print("Citation Manager Ready.")
