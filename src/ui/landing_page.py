# RELOAD_FORCE: 2026-01-21 22:36
"""
StoryWeaver Landing Page

Main entry point for the dual-mode platform.
Provides navigation to Document Analyzer and Story Mode.
"""

import streamlit as st
import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

st.set_page_config(
    page_title="StoryWeaver Platform",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
    }
    .main-header h1 {
        color: #4A90E2;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #666;
        font-size: 1.2rem;
    }
    .mode-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 2rem;
        color: white;
        min-height: 300px;
    }
    .mode-card-doc {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .mode-card-story {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .feature-list {
        list-style: none;
        padding: 0;
        margin-top: 1rem;
    }
    .feature-list li {
        padding: 0.5rem 0;
        display: flex;
        align-items: center;
    }
    .stButton > button {
        width: 100%;
        padding: 1rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="main-header">
    <h1>✨ StoryWeaver Platform ✨</h1>
    <p>Dual-Mode RAG System: Document Analysis & AI Story Generation</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Three-column mode selection
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📚 Document Analyzer")
    st.markdown("""
    **Unified RAG System**
    
    Standard hybrid search for general document analysis.
    """)
    
    st.markdown("""
    - 🔍 **Hybrid Search**
    - 💾 **Semantic Caching**
    - ⚖️ **Adaptive Weights**
    """)
    
    st.markdown("")
    if st.button("📖 Open Document Analyzer", key="doc_btn", use_container_width=True):
        st.switch_page("pages/document_app.py")

with col2:
    st.markdown("### 🧬 Elite Research Mode")
    st.markdown("""
    **Advanced Intelligence Hub**
    
    Research-grade RAG with Chain-of-Thought and QLoRA labs.
    """)
    
    st.markdown("""
    - 🧠 **Adaptive CoT Reasoning**
    - 🔗 **Grounded Citations**
    - 🧪 **QLoRA Fine-tuning**
    - 📊 **Matrix Analytics**
    """)
    
    st.markdown("")
    if st.button("🔱 Enter Elite Mode", key="elite_btn", use_container_width=True):
        st.switch_page("pages/elite_app.py")

with col3:
    st.markdown("### 🎭 Story Mode")
    st.markdown("""
    **Narrative Generation**
    
    Generate stories using Graph RAG and comparison engines.
    """)
    
    st.markdown("""
    - 🆚 **Graph vs Unified RAG**
    - 👥 **Character Tracking**
    - 🕸️ **Relationship Graphs**
    """)
    
    st.markdown("")
    if st.button("🎭 Enter Story Mode", key="story_btn", use_container_width=True):
        st.switch_page("pages/story_app.py")

with col3:
    st.markdown("### 🧪 Meta-Analysis")
    st.markdown("""
    **A/B Test Engine**
    
    Aggregate experiments, detect bias, and find the true effect.
    """)
    
    st.markdown("""
    - 📊 **Statistical Engine**: Fixed & Random effects
    - 🔍 **Bias Detection**: Publication bias & outliers
    - 🧠 **AI Insights**: RAG-powered interpretation
    - 📈 **Visualizations**: Interactive plots
    """)
    
    st.markdown("")
    if st.button("🧪 Open Meta-Analysis", key="meta_btn", use_container_width=True):
        st.switch_page("pages/meta_analysis_app.py")

# Footer section
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🔧 Hardware Optimized")
    st.caption("Runs on 16GB RAM + RTX 4050 (6GB VRAM)")

with col2:
    st.markdown("#### ⚡ Fast & Efficient")
    st.caption("fp16 precision, batch processing, auto cleanup")

with col3:
    st.markdown("#### 🌐 Dual API")
    st.caption("FastAPI REST + Streamlit Web UI")
