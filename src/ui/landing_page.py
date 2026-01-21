# RELOAD_FORCE: 2026-01-22
"""
Unified RAG System Landing Page

Main entry point for the dual-mode platform.
Provides navigation to Document Analyzer, QLoRA Training Studio, and Story Mode.
"""

import streamlit as st
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ui.styles.cyberpunk_theme import load_cyberpunk_theme

st.set_page_config(
    page_title="Unified RAG System",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load Cyberpunk Design System
load_cyberpunk_theme()

# Hero Section
st.markdown("""
<div class="main-header" style="text-align: center; margin-bottom: 2rem;">
    <h1 class="glitch-hover" style="font-size: 3.5rem; margin-bottom: 0;">UNIFIED_RAG_SYSTEM</h1>
    <p style="color: var(--neon-cyan); letter-spacing: 2px; font-family: var(--font-ui);">
        > SYSTEM_READY // MODE_SELECT_REQUIRED <span style="animation: blink 1s infinite">_</span>
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Four-column mode selection
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="cyber-card">
        <div class="cyber-header">DOC_ANALYZER</div>
        <p style="font-size: 0.9rem; color: var(--text-secondary);">
            Hybrid search matrix for document ingestion and semantic analysis.
        </p>
        <ul style="list-style: none; padding: 0; font-family: var(--font-ui); font-size: 0.8rem; color: var(--neon-green);">
            <li>[+] HYBRID_SEARCH</li>
            <li>[+] SEMANTIC_CACHE</li>
            <li>[+] ADAPTIVE_WEIGHTS</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("INITIATE_ANALYSIS", key="doc_btn", use_container_width=True):
        st.switch_page("pages/document_app.py")

with col2:
    st.markdown("""
    <div class="cyber-card">
        <div class="cyber-header">QLORA_STUDIO</div>
        <p style="font-size: 0.9rem; color: var(--text-secondary);">
            Fine-tune LLM weights on local hardware via quantized adaptors.
        </p>
        <ul style="list-style: none; padding: 0; font-family: var(--font-ui); font-size: 0.8rem; color: var(--neon-purple);">
            <li>[+] GROQ_DATASETS</li>
            <li>[+] QLORA_TRAINING</li>
            <li>[+] MODEL_EXPORT</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("ACCESS_LAB", key="qlora_btn", use_container_width=True):
        st.switch_page("pages/qlora_studio.py")

with col3:
    st.markdown("""
    <div class="cyber-card">
        <div class="cyber-header">STORY_WEAVER</div>
        <p style="font-size: 0.9rem; color: var(--text-secondary);">
            Narrative generation engine with Graph RAG entity tracking.
        </p>
        <ul style="list-style: none; padding: 0; font-family: var(--font-ui); font-size: 0.8rem; color: var(--neon-cyan);">
            <li>[+] GRAPH_RETRIEVAL</li>
            <li>[+] ENTITY_TRACKING</li>
            <li>[+] PLOT_BRANCHING</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("ENTER_SIMULATION", key="story_btn", use_container_width=True):
        st.switch_page("pages/story_app.py")

with col4:
    st.markdown("""
    <div class="cyber-card">
        <div class="cyber-header">META_ANALYSIS</div>
        <p style="font-size: 0.9rem; color: var(--text-secondary);">
            Statistical aggregation engine like "Feature 3" for A/B tests.
        </p>
        <ul style="list-style: none; padding: 0; font-family: var(--font-ui); font-size: 0.8rem; color: var(--danger);">
            <li>[+] STAT_ENGINE</li>
            <li>[+] BIAS_DETECTION</li>
            <li>[+] DATA_SYNTHESIS</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("RUN_DIAGNOSTICS", key="meta_btn", use_container_width=True):
        st.switch_page("pages/meta_analysis_app.py")

# Footer section
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="border-left: 2px solid var(--border-dim); padding-left: 10px;">
        <div style="font-family: var(--font-ui); color: var(--text-secondary); font-size: 0.8rem;">HARDWARE</div>
        <div style="font-family: var(--font-headings); color: var(--text-primary);">RTX 4050 / 16GB</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="border-left: 2px solid var(--border-dim); padding-left: 10px;">
        <div style="font-family: var(--font-ui); color: var(--text-secondary); font-size: 0.8rem;">ENGINE</div>
        <div style="font-family: var(--font-headings); color: var(--text-primary);">FP16 PRECISION</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="border-left: 2px solid var(--border-dim); padding-left: 10px;">
        <div style="font-family: var(--font-ui); color: var(--text-secondary); font-size: 0.8rem;">INTERFACE</div>
        <div style="font-family: var(--font-headings); color: var(--text-primary);">DUAL_API_LINK</div>
    </div>
    """, unsafe_allow_html=True)
