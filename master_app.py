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

# Import the landing page
import streamlit as st

# We need to run the landing page logic
# Since st.set_page_config must be the first command, 
# we'll just shell out to the original landing page for proper multi-page support
# OR, we can just tell the user to run: streamlit run src/ui/landing_page.py
# But providing a master entry point is better.

from src.ui import landing_page
