"""
Manuscript Manager: Handles the 'Living Manuscript' for Elite Research Mode.
Persists research drafts in session state and allows export to Markdown.
"""

import streamlit as st
import datetime
from typing import List, Dict
import os

class ManuscriptManager:
    """
    Manages the persistent research draft (Living Manuscript).
    """
    
    def __init__(self):
        if 'manuscript_sections' not in st.session_state:
            st.session_state.manuscript_sections = []
            
    def add_section(self, title: str, content: str, source_query: str = ""):
        """
        Adds a new section to the manuscript.
        """
        section = {
            "timestamp": datetime.datetime.now().isoformat(),
            "title": title,
            "content": content,
            "source_query": source_query
        }
        st.session_state.manuscript_sections.append(section)
        
    def get_markdown(self) -> str:
        """
        Compiles the manuscript into a Markdown string.
        """
        if not st.session_state.manuscript_sections:
            return "# New Research Draft\n\n*No content added yet.*"
            
        md = f"# Research Manuscript\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        for i, section in enumerate(st.session_state.manuscript_sections):
            md += f"## {i+1}. {section['title']}\n\n"
            md += f"{section['content']}\n\n"
            if section['source_query']:
                md += f"> *Derived from query: {section['source_query']}*\n\n"
            md += "---\n\n"
            
        return md

    def clear(self):
        """Clears the current manuscript."""
        st.session_state.manuscript_sections = []
