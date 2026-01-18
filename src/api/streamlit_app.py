"""
Streamlit Web UI

Interactive chat interface for the RAG system with:
- Document upload
- Chat interface
- Settings panel
- Metrics display
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.rag_system import UnifiedRAGSystem


# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🤖 Advanced RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #0f3460;
    }
    
    .stChatMessage {
        background-color: #1a1a2e;
        border-radius: 10px;
    }
    
    .source-box {
        background: #1a1a2e;
        border-left: 3px solid #667eea;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
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
        if os.path.exists(config_path):
            st.session_state.rag_system = UnifiedRAGSystem(config_path)
        else:
            st.session_state.rag_system = UnifiedRAGSystem()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    if 'last_query' not in st.session_state:
        st.session_state.last_query = None


init_session_state()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📄 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        help="Upload PDF, TXT, or DOCX files to index"
    )
    
    if uploaded_files and st.button("🚀 Process Documents", type="primary"):
        with st.spinner("Processing documents..."):
            total_chunks = 0
            
            for file in uploaded_files:
                # Save to temp file
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(file.name).suffix
                ) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                
                try:
                    chunks = st.session_state.rag_system.ingest_file(tmp_path)
                    total_chunks += chunks
                    st.success(f"✅ {file.name}: {chunks} chunks")
                except Exception as e:
                    st.error(f"❌ {file.name}: {e}")
                finally:
                    os.unlink(tmp_path)
            
            if total_chunks > 0:
                with st.spinner("Building index..."):
                    st.session_state.rag_system.build_index()
                st.success(f"🎉 Indexed {total_chunks} total chunks!")
    
    st.divider()
    
    # Settings
    st.markdown("## ⚙️ Settings")
    
    top_k = st.slider(
        "Results to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of source documents to retrieve"
    )
    
    use_cache = st.checkbox(
        "Enable semantic cache",
        value=True,
        help="Cache responses for similar queries"
    )
    
    st.divider()
    
    # Metrics
    st.markdown("## 📊 Metrics")
    
    metrics = st.session_state.rag_system.get_metrics()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", metrics.get('queries', 0))
        st.metric("Cache Hits", metrics.get('cache_hits', 0))
    with col2:
        st.metric("Documents", metrics.get('documents_ingested', 0))
        st.metric("Chunks", metrics.get('chunks_created', 0))
    
    cache_rate = metrics.get('cache_hit_rate', 0)
    st.progress(cache_rate, text=f"Cache Hit Rate: {cache_rate:.1%}")
    
    st.divider()
    
    # Memory status
    memory = metrics.get('memory', {})
    ram = memory.get('ram', {})
    vram = memory.get('vram', {})
    
    if ram:
        st.markdown("### 💾 Memory")
        st.progress(
            ram.get('percent', 0),
            text=f"RAM: {ram.get('used_gb', 0):.1f}/{ram.get('total_gb', 0):.1f} GB"
        )
    
    if vram:
        st.progress(
            vram.get('percent', 0),
            text=f"VRAM: {vram.get('used_gb', 0):.1f}/{vram.get('total_gb', 0):.1f} GB"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main Chat Interface
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-header">🤖 Advanced RAG System</h1>', unsafe_allow_html=True)

# Status indicator
if st.session_state.rag_system._indexed:
    doc_count = len(st.session_state.rag_system.documents)
    st.success(f"✅ System ready with {doc_count} indexed chunks")
else:
    st.warning("⚠️ No documents indexed. Upload files in the sidebar.")

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources", expanded=False):
                for i, src in enumerate(message["sources"]):
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {i+1}</strong> (score: {src['score']:.3f})<br>
                        {src['content'][:300]}...
                    </div>
                    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Check if indexed
    if not st.session_state.rag_system._indexed:
        st.error("Please upload and process documents first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.last_query = prompt
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.rag_system.query(
                        question=prompt,
                        top_k=top_k,
                        use_cache=use_cache
                    )
                    
                    st.markdown(result.answer)
                    
                    # Show cache status
                    if result.cached:
                        st.caption("⚡ Response from cache")
                    else:
                        st.caption(
                            f"🔍 Query type: {result.query_type} | "
                            f"Weights: sem={result.semantic_weight:.2f}, lex={result.lexical_weight:.2f}"
                        )
                    
                    # Store for history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.answer,
                        "sources": [
                            {"content": s.content, "score": s.score}
                            for s in result.sources
                        ]
                    })
                    
                    # Show sources
                    if result.sources:
                        with st.expander("📚 Sources", expanded=False):
                            for i, src in enumerate(result.sources):
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Source {i+1}</strong> (score: {src.score:.3f})<br>
                                    {src.content[:300]}...
                                </div>
                                """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"Query failed: {e}")

# Feedback section
if st.session_state.last_query:
    st.divider()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        rating = st.slider(
            "Rate the last response",
            min_value=1,
            max_value=5,
            value=3,
            help="1 = Poor, 5 = Excellent"
        )
    
    with col2:
        if st.button("Submit Feedback"):
            st.session_state.rag_system.submit_feedback(
                st.session_state.last_query,
                rating
            )
            st.success("Thanks for your feedback!")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "🚀 Advanced RAG System | "
    "Hybrid Search + Semantic Caching + Adaptive Weights | "
    "Powered by Groq & FAISS"
)

