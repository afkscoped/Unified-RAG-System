"""
Search Mode Toggle Component
"""
import streamlit as st
from src.enhancements.config import config

def render_search_mode_toggle(default_mode: str = 'docs') -> str:
    """Renders radio buttons for search mode"""
    st.markdown("### 🔍 Search Mode")
    
    modes = ['docs', 'web', 'hybrid']
    labels = {
        'docs': '📚 Docs Only',
        'web': '🌐 Web Only',
        'hybrid': '🚀 Hybrid (Best)'
    }
    
    # Check if web search is enabled globally
    if not config.enable_web_search:
        st.info("Web search is disabled in config.")
        return 'docs'
        
    if 'search_mode' not in st.session_state:
        st.session_state.search_mode = default_mode
        
    selected = st.radio(
        "Select Source:",
        options=modes,
        format_func=lambda x: labels[x],
        key='mode_radio',
        index=modes.index(st.session_state.search_mode)
    )
    
    st.session_state.search_mode = selected
    return selected
