"""
Elite RAG Gateway
"""
import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app_elite import main

if __name__ == "__main__":
    main()
