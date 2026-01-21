"""
Document Analyzer UI (Enhanced)
Interactive chat interface with AI Personas and Web Search integration.
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from loguru import logger

# Add src to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.rag_system import UnifiedRAGSystem
from src.enhancements.manager import EnhancedAnalyzer
from src.enhancements.config import config
from src.ui_components.persona_selector import render_persona_selector
from src.ui_components.search_mode_toggle import render_search_mode_toggle
from src.ui_components.enhanced_results_viewer import render_enhanced_results
from src.ui_components.evaluation_dashboard import render_evaluation_dashboard

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Document analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #6c5ce7 0%, #a29bfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .stChatMessage {
        background-color: #1a1a2e;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────

def init_session_state():
    """Initialize session state variables."""
    if 'rag_system' not in st.session_state:
        config_path = "config/config.yaml"
        # 1. Init Core System
        if os.path.exists(config_path):
            core_rag = UnifiedRAGSystem(config_path)
        else:
            core_rag = UnifiedRAGSystem()
            
        # 2. Wrap with Enhanced Analyzer
        st.session_state.rag_system = EnhancedAnalyzer(core_rag)
        st.session_state.is_enhanced = True
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

init_session_state()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    if st.button("← Back to Home", use_container_width=True):
        st.switch_page("landing_page.py")
    
    st.divider()
    
    # 🎯 NEW: Persona Selector (if enabled)
    selected_persona = 'scientist'
    if config.enable_personas:
        selected_persona = render_persona_selector(config.default_persona)
        st.divider()
        
    # 🔍 NEW: Search Mode (if enabled)
    search_mode = 'docs'
    if config.enable_web_search:
        search_mode = render_search_mode_toggle(config.default_search_mode)
        st.divider()
    
    st.markdown("## 📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("🚀 Process", type="primary"):
        with st.spinner("Processing..."):
            total = 0
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                try:
                    # Access core system for ingestion
                    chunks = st.session_state.rag_system.rag.ingest_file(
                        tmp_path, 
                        original_name=file.name
                    )
                    total += chunks
                    st.success(f"✅ {file.name}: {chunks} chunks")
                finally:
                    os.unlink(tmp_path)
            
            if total > 0:
                st.session_state.rag_system.rag.build_index()
                st.success("Indexed successfully!")

    # Metrics Panel
    st.divider()
    with st.expander("📊 System Metrics"):
        metrics = st.session_state.rag_system.get_metrics()
        
        # Clean summary
        m_cols = st.columns(2)
        with m_cols[0]:
            st.metric("Queries", metrics.get("queries", 0))
            st.metric("Cache Hub", metrics.get("cache_size", 0))
        with m_cols[1]:
            st.metric("Docs", metrics.get("documents_ingested", 0))
            st.metric("Chunks", metrics.get("chunks_created", 0))
            
        if st.checkbox("Show Advanced Debug", False):
            st.json(metrics)

# ─────────────────────────────────────────────────────────────────────────────
# Main Interface
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-header">🧬 Document analyzer</h1>', unsafe_allow_html=True)

tab_chat, tab_analytics = st.tabs(["💬 Chat Interface", "📈 Analytics Dashboard"])

with tab_chat:
    # Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("is_enhanced_result"):
                render_enhanced_results(msg["content"])
            else:
                st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask anything..."):
        # Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking as {selected_persona.title()}..."):
                try:
                    # Call wrapper
                    response_dict = st.session_state.rag_system.query(
                        question=prompt,
                        search_mode=search_mode,
                        persona_id=selected_persona
                    )
                    
                    # Render using new component
                    render_enhanced_results(response_dict)
                    
                    # Store full dict in history for clean re-rendering
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_dict, 
                        "is_enhanced_result": True
                    })
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"Enhanced query error: {e}")

with tab_analytics:
    render_evaluation_dashboard(st.session_state.rag_system)
