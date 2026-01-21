# RELOAD_FORCE: 2026-01-21 22:38
"""
[ELITE ARCHITECTURE] app_elite.py
MASTER MISSION CONTROL for the Unified RAG + QLoRA System.
"""

import streamlit as st
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.core.elite_rag import EliteRAGSystem
from src.core.llm_router import LLMRouter
from src.core.embeddings import EmbeddingManager
from src.core.search_engine import SearchEngine
from src.ui.enhanced_document_app.tabs.document_manager import render_document_manager
from src.ui.enhanced_document_app.tabs.query_interface import render_query_interface
from src.finetuning.ui.finetuning_tab import render_finetuning_tab
from src.evaluation.ui.quality_dashboard import render_quality_dashboard

# Page Configuration
st.set_page_config(
    page_title="ANTIGRAVITY // ELITE_RAG",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Robotic/Industrial Aesthetic)
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #0e1117;
        border-right: 1px solid #3d3d3d;
    }
    .stMetric {
        background: #1e1e1e;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_system():
    # Load config to get standard model ID
    import yaml
    with open("config/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Concrete Engine Initialization
    router = LLMRouter(provider="groq", model=cfg['model']['llm_model'])
    embedder = EmbeddingManager(model_name=cfg['model']['embedding'])
    search = SearchEngine(dimension=cfg['model']['embedding_dim'])
    
    return EliteRAGSystem(
        llm_router=router,
        embedding_manager=embedder,
        search_engine=search
    )

def main():
    # 1. Background State
    system = get_system()
    
    # 2. Sidebar Navigation
    st.sidebar.title("🧬 SYSTEM_AETHER_v2")
    st.sidebar.markdown("`GTX_1650_4GB_ACTIVE`")
    
    navigation = st.sidebar.radio(
        "MODULE_SELECT",
        ["01_DOC_VECTORS", "02_RESEARCH_HUB", "03_QLORA_LAB", "04_MATRIX_ANALYTICS"],
        index=1
    )
    
    # Global Hardware Watcher
    st.sidebar.markdown("---")
    st.sidebar.caption("SYS_TELEMETRY")
    st.sidebar.progress(72, text="VRAM: 2.9GB / 4.0GB") # Dynamic stat in production
    
    # 3. Tab Rendering
    if navigation == "01_DOC_VECTORS":
        render_document_manager(system)
    elif navigation == "02_RESEARCH_HUB":
        render_query_interface(system)
    elif navigation == "03_QLORA_LAB":
        render_finetuning_tab(system)
    elif navigation == "04_MATRIX_ANALYTICS":
        eval_history = [res.get('evaluation') for res in st.session_state.get('chat_history', []) if res.get('evaluation')]
        corpus_data = system.get_corpus_data()
        render_quality_dashboard(
            eval_history, 
            graph_data=system.kg.get_graph_data(),
            corpus_data=corpus_data
        )

if __name__ == "__main__":
    main()
