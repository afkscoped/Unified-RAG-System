"""
Unified RAG System - Master Entry Point
"""
import os
import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).parent
sys.path.insert(0, str(root))

# Set the working directory to the project root
os.chdir(root)

import streamlit as st

st.set_page_config(
    page_title="QLoRA Training Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.elite_app.main import render_app

render_app(set_page_config=False)
