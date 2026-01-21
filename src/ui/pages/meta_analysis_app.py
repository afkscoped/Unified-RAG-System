import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import streamlit as st
from src.meta_analysis.ui.streamlit_app import render_meta_analysis_page

if __name__ == "__main__":
    st.set_page_config(
        page_title="Meta-Analysis Engine",
        page_icon="🧪",
        layout="wide"
    )
    from src.ui.styles.cyberpunk_theme import load_cyberpunk_theme
    load_cyberpunk_theme()
    
    render_meta_analysis_page()
