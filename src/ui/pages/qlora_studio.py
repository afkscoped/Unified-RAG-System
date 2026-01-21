"""
QLoRA Training Studio Gateway
"""
import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path and change working directory
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Change to project root so relative config paths work
os.chdir(project_root)

from src.ui.styles.cyberpunk_theme import load_cyberpunk_theme
from src.elite_app.main import render_app

if __name__ == "__main__":
    # Load Theme
    load_cyberpunk_theme()
    
    # Render App
    render_app(set_page_config=False)
