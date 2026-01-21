"""
Document Analyzer App - Production Interface with Deep Persona Integration
Complete RAG pipeline shaped by active persona
"""

import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Imports with error handling
IMPORTS_OK = True
import_errors = []

try:
    from src.core.hybrid_search_engine import HybridSearchEngine
except ImportError as e:
    import_errors.append(f"HybridSearchEngine: {e}")
    HybridSearchEngine = None

try:
    from src.core.semantic_cache import SemanticCache
except ImportError as e:
    import_errors.append(f"SemanticCache: {e}")
    SemanticCache = None

try:
    from src.core.persona_engine import PersonaEngine, PersonaType
except ImportError as e:
    import_errors.append(f"PersonaEngine: {e}")
    PersonaEngine = None
    PersonaType = None

try:
    from src.core.judicial_evaluator import JudicialEvaluator
except ImportError as e:
    import_errors.append(f"JudicialEvaluator: {e}")
    JudicialEvaluator = None

try:
    from src.core.response_generator import ResponseGenerator
except ImportError as e:
    import_errors.append(f"ResponseGenerator: {e}")
    ResponseGenerator = None

try:
    from src.utils.document_processor import DocumentProcessor
except ImportError as e:
    import_errors.append(f"DocumentProcessor: {e}")
    DocumentProcessor = None

try:
    from src.utils.metrics_tracker import MetricsTracker
except ImportError as e:
    import_errors.append(f"MetricsTracker: {e}")
    MetricsTracker = None

if import_errors:
    IMPORTS_OK = False

# Page config
st.set_page_config(
    page_title="Document Analyzer Pro",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Cyberpunk Theme
from src.ui.styles.cyberpunk_theme import load_cyberpunk_theme
load_cyberpunk_theme()

# Custom CSS
st.markdown("""
<style>
    .persona-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .query-transform {
        background: #1e1e2e;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .follow-up-btn {
        border: 1px solid #4CAF50;
        background: transparent;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize all session state variables"""
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'all_chunks' not in st.session_state:
        st.session_state.all_chunks = []
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = None
    if 'semantic_cache' not in st.session_state:
        st.session_state.semantic_cache = SemanticCache() if SemanticCache else None
    if 'persona_engine' not in st.session_state:
        st.session_state.persona_engine = PersonaEngine() if PersonaEngine else None
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = JudicialEvaluator() if JudicialEvaluator else None
    if 'response_generator' not in st.session_state:
        st.session_state.response_generator = ResponseGenerator() if ResponseGenerator else None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = MetricsTracker() if MetricsTracker else None
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'pending_query' not in st.session_state:
        st.session_state.pending_query = None


def main():
    init_session_state()
    
    # Show import errors if any
    if not IMPORTS_OK:
        with st.expander("⚠️ Some modules failed to import", expanded=False):
            for err in import_errors:
                st.code(err)
    
    # Sidebar
    with st.sidebar:
        st.title("📚 Document Analyzer Pro")
        st.caption("Persona-Driven Hybrid RAG")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["📤 Upload & Process", "🔍 Intelligent Search", "📊 Analytics", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Real-time System Metrics
        render_sidebar_metrics()
        
        st.markdown("---")
        
        # Enhanced Persona Selector
        if st.session_state.persona_engine:
            render_persona_selector()
    
    # Main content area
    if page == "📤 Upload & Process":
        render_upload_page()
    elif page == "🔍 Intelligent Search":
        render_search_page()
    elif page == "📊 Analytics":
        render_analytics_page()
    elif page == "⚙️ Settings":
        render_settings_page()


def render_sidebar_metrics():
    """Show REAL metrics only"""
    st.subheader("📊 System Status")
    
    num_docs = len(st.session_state.documents)
    num_chunks = len(st.session_state.all_chunks)
    num_queries = len(st.session_state.query_history)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", num_docs)
        st.metric("Queries", num_queries)
    with col2:
        st.metric("Chunks", num_chunks)
        if st.session_state.semantic_cache:
            stats = st.session_state.semantic_cache.get_stats()
            st.metric("Cache Hits", stats['total_hits'])
    
    # Index status
    if st.session_state.search_engine:
        st.success("✅ Search Index Ready")
    else:
        st.info("⏳ No index yet")


def render_persona_selector():
    """Enhanced persona selection with visual indicators"""
    st.subheader("🎭 AI Persona")
    
    personas = st.session_state.persona_engine.get_all_personas()
    persona_options = {f"{p.icon} {p.name}": p.type for p in personas}
    
    selected = st.selectbox(
        "Reasoning Style",
        list(persona_options.keys()),
        index=0,
        label_visibility="collapsed"
    )
    
    persona_type = persona_options[selected]
    st.session_state.persona_engine.set_persona(persona_type)
    
    active = st.session_state.persona_engine.get_active_persona()
    st.caption(f"_{active.thinking_style}_")
    
    # Show persona profile
    with st.expander("📋 Persona Profile"):
        # Search preferences
        st.markdown("**Search Strategy:**")
        weights = active.search_weights
        st.progress(weights.bm25_weight, text=f"Lexical: {weights.bm25_weight:.0%}")
        st.progress(weights.semantic_weight, text=f"Semantic: {weights.semantic_weight:.0%}")
        st.progress(weights.diversity_preference, text=f"Diversity: {weights.diversity_preference:.0%}")
        
        st.markdown("---")
        st.markdown("**Response Style:**")
        template = active.response_template
        st.markdown(f"- Opening: *{template.opening_style}*")
        st.markdown(f"- Depth: *{template.explanation_depth}*")
        st.markdown(f"- Analogies: {'✅' if template.use_analogies else '❌'}")
        st.markdown(f"- Counterpoints: {'✅' if template.include_counterpoints else '❌'}")
        st.markdown(f"- Query expansion: *{active.query_expansion_strategy}*")


def render_upload_page():
    """Document upload and processing interface"""
    st.title("📤 Document Upload & Processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Drag and drop files here",
            type=['pdf', 'txt', 'docx', 'pptx', 'xlsx'],
            accept_multiple_files=True,
            help="PDF, TXT, DOCX, PPTX, XLSX supported"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                process_documents(uploaded_files)
    
    with col2:
        st.subheader("Processing Options")
        chunk_size = st.slider("Chunk Size", 128, 1024, 512)
        chunk_overlap = st.slider("Overlap", 0, 200, 50)
    
    # Show uploaded documents
    if st.session_state.documents:
        st.markdown("---")
        st.subheader(f"📁 Loaded Documents ({len(st.session_state.documents)})")
        
        for idx, doc in enumerate(st.session_state.documents):
            with st.expander(f"📄 {doc.get('filename', 'Unknown')} - {doc.get('type', 'N/A')}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Chunks", doc.get('num_chunks', 0))
                col2.metric("Size", f"{doc.get('size_kb', 0):.1f} KB")
                col3.metric("Type", doc.get('type', 'N/A'))


def process_documents(uploaded_files):
    """Process uploaded documents"""
    if not DocumentProcessor:
        st.error("DocumentProcessor not available")
        return
    
    processor = DocumentProcessor()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processed_docs = []
    all_chunks = []
    
    for idx, file in enumerate(uploaded_files):
        status_text.text(f"Processing {file.name}...")
        
        try:
            doc_data = processor.process_file(file, file.name)
            doc_data['size_kb'] = file.size / 1024
            doc_data['upload_time'] = datetime.now().strftime("%H:%M:%S")
            
            processed_docs.append(doc_data)
            all_chunks.extend(doc_data.get('chunks', []))
            
            if st.session_state.metrics:
                st.session_state.metrics.log_document(len(doc_data.get('chunks', [])))
                
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")
        
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    # Update session state
    st.session_state.documents.extend(processed_docs)
    st.session_state.all_chunks.extend(all_chunks)
    
    # Build search index
    if all_chunks and HybridSearchEngine:
        status_text.text("Building search indices...")
        try:
            st.session_state.search_engine = HybridSearchEngine()
            st.session_state.search_engine.index_documents(all_chunks)
            st.success(f"✅ Processed {len(uploaded_files)} documents, indexed {len(all_chunks)} chunks!")
        except Exception as e:
            st.error(f"Error building index: {e}")
    
    progress_bar.empty()
    status_text.empty()


def render_search_page():
    """Intelligent search interface with deep persona integration"""
    st.title("🔍 Intelligent Hybrid Search")
    
    if not st.session_state.search_engine:
        st.warning("⚠️ No documents indexed. Please upload documents first.")
        return
    
    # Get active persona
    persona = st.session_state.persona_engine.get_active_persona()
    
    # Show active persona badge
    st.markdown(f"""
    <div class="persona-badge">
        {persona.icon} Active: {persona.name} | {persona.thinking_style}
    </div>
    """, unsafe_allow_html=True)
    
    # Search input
    query = st.text_input(
        "Enter your question or search query",
        placeholder="What are the key findings about...",
        key="search_query",
        value=st.session_state.pending_query or ""
    )
    
    # Clear pending query after use
    if st.session_state.pending_query:
        st.session_state.pending_query = None
    
    # Search options
    col1, col2, col3 = st.columns(3)
    with col1:
        use_cache = st.checkbox("Use Semantic Cache", value=True)
    with col2:
        show_transform = st.checkbox("Show Query Transform", value=True)
    with col3:
        top_k = st.slider("Results", 3, 20, 10)
    
    if query and st.button("🔍 Search", type="primary"):
        execute_persona_search(query, use_cache, show_transform, top_k)


def execute_persona_search(query: str, use_cache: bool, show_transform: bool, top_k: int):
    """Execute persona-aware hybrid search"""
    start_time = time.time()
    
    # Get active persona
    persona = st.session_state.persona_engine.get_active_persona()
    
    # Check cache first
    if use_cache and st.session_state.semantic_cache:
        cached = st.session_state.semantic_cache.get(query)
        if cached:
            response, sources, similarity = cached
            st.success(f"💾 Cache hit! (Similarity: {similarity:.3f})")
            display_cached_results(response, sources)
            return
    
    # Transform query based on persona
    with st.spinner(f"{persona.icon} Analyzing query with {persona.name} perspective..."):
        query_transformation = st.session_state.persona_engine.transform_query(query)
    
    # Show query transformation
    if show_transform:
        with st.expander("🔄 Query Transformation", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Original:** {query}")
                st.markdown(f"**Strategy:** `{query_transformation.search_strategy}`")
            with col2:
                st.markdown(f"**Expansion:** `{persona.query_expansion_strategy}`")
                st.markdown(f"**Keywords:** {', '.join(query_transformation.focus_keywords[:5])}")
            
            if query_transformation.sub_queries:
                st.markdown("**Sub-queries:**")
                for sq in query_transformation.sub_queries:
                    st.markdown(f"- _{sq}_")
    
    # Get persona search weights
    persona_weights = st.session_state.persona_engine.get_search_weights()
    
    # Show search weight profile
    st.caption(f"🔍 Search weights: BM25={persona_weights.bm25_weight:.0%} | Semantic={persona_weights.semantic_weight:.0%}")
    
    # Execute persona-aware search
    with st.spinner(f"Searching with {persona.name}'s priorities..."):
        results = st.session_state.search_engine.search_with_persona(
            query=query,
            persona_weights=persona_weights,
            query_transformation=query_transformation,
            top_k=top_k
        )
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Generate persona-specific response
    response = None
    if st.session_state.response_generator and results:
        with st.spinner(f"{persona.icon} Generating {persona.name} response..."):
            response = st.session_state.response_generator.generate_response(
                query=query,
                search_results=results,
                persona=persona
            )
    
    # Evaluate response
    eval_result = None
    if st.session_state.evaluator and response and results:
        eval_result = st.session_state.evaluator.evaluate(
            query=query,
            response=response,
            sources=[r.content for r in results]
        )
    
    # Generate follow-up questions
    follow_ups = st.session_state.persona_engine.generate_follow_up_questions(query, response or "")
    
    # Cache result
    if use_cache and st.session_state.semantic_cache and response:
        sources_for_cache = [{'content': r.content, 'score': r.score} for r in results]
        st.session_state.semantic_cache.set(query, response, sources_for_cache)
    
    # Log to history
    st.session_state.query_history.append({
        'query': query,
        'timestamp': datetime.now(),
        'num_results': len(results),
        'latency_ms': latency_ms,
        'persona': persona.name,
        'eval_score': eval_result.overall_score if eval_result else None
    })
    
    # Display results
    display_search_results(query, response, results, eval_result, follow_ups, latency_ms, persona)


def display_cached_results(response: str, sources: list):
    """Display cached results"""
    st.markdown("### 💬 Response (Cached)")
    st.markdown(response)
    
    st.markdown("---")
    st.markdown("### 📚 Cached Sources")
    for idx, source in enumerate(sources[:5], 1):
        with st.expander(f"Source {idx} (Score: {source.get('score', 0):.3f})"):
            st.markdown(source.get('content', '')[:500] + "...")


def display_search_results(query, response, results, eval_result, follow_ups, latency_ms, persona):
    """Display search results with persona-specific formatting"""
    
    # Performance metrics
    st.caption(f"⏱️ {latency_ms:.0f}ms | {len(results)} results | {persona.icon} {persona.name}")
    
    st.markdown("---")
    
    # Show persona-formatted response
    if response:
        st.markdown(response)
    
    # Show evaluation
    if eval_result:
        st.markdown("---")
        st.markdown("### 📊 Response Quality")
        
        col1, col2, col3, col4 = st.columns(4)
        
        def get_emoji(score):
            if score >= 0.7: return "🟢"
            elif score >= 0.5: return "🟡"
            return "🔴"
        
        col1.metric("Faithfulness", f"{eval_result.faithfulness_score:.0%}", 
                   delta=get_emoji(eval_result.faithfulness_score))
        col2.metric("Relevance", f"{eval_result.relevance_score:.0%}",
                   delta=get_emoji(eval_result.relevance_score))
        col3.metric("Citations", f"{eval_result.citation_coverage:.0%}",
                   delta=get_emoji(eval_result.citation_coverage))
        col4.metric("Overall", f"{eval_result.overall_score:.0%}",
                   delta=get_emoji(eval_result.overall_score))
        
        if eval_result.warnings:
            for warning in eval_result.warnings:
                st.warning(warning)
    
    # Show follow-up questions
    if follow_ups:
        st.markdown("---")
        st.markdown(f"### 🎯 {persona.icon} Suggested Follow-ups")
        
        cols = st.columns(len(follow_ups))
        for i, (col, follow_up) in enumerate(zip(cols, follow_ups)):
            with col:
                if st.button(follow_up[:40] + "...", key=f"followup_{i}", use_container_width=True):
                    st.session_state.pending_query = follow_up
                    st.rerun()
    
    # Show sources
    st.markdown("---")
    st.markdown("### 📚 Source Documents")
    
    for idx, result in enumerate(results[:10], 1):
        source_name = result.metadata.get('source', result.chunk_id)
        with st.expander(f"#{result.rank} | {source_name} | Score: {result.score:.4f}"):
            st.markdown(result.content)


def render_analytics_page():
    """Analytics dashboard"""
    st.title("📊 Analytics Dashboard")
    
    if not st.session_state.query_history:
        st.info("No queries yet. Execute some searches to see analytics.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_queries = len(st.session_state.query_history)
    eval_scores = [q.get('eval_score') for q in st.session_state.query_history if q.get('eval_score')]
    avg_eval = sum(eval_scores) / len(eval_scores) if eval_scores else 0
    avg_latency = sum(q.get('latency_ms', 0) for q in st.session_state.query_history) / max(total_queries, 1)
    
    # Persona distribution
    persona_counts = {}
    for q in st.session_state.query_history:
        p = q.get('persona', 'Unknown')
        persona_counts[p] = persona_counts.get(p, 0) + 1
    
    col1.metric("Total Queries", total_queries)
    col2.metric("Avg Quality", f"{avg_eval:.0%}")
    col3.metric("Avg Latency", f"{avg_latency:.0f}ms")
    col4.metric("Personas Used", len(persona_counts))
    
    st.markdown("---")
    
    # Persona usage
    st.subheader("🎭 Persona Usage")
    for persona, count in sorted(persona_counts.items(), key=lambda x: x[1], reverse=True):
        st.progress(count / total_queries, text=f"{persona}: {count} queries")
    
    st.markdown("---")
    
    # Recent queries
    st.subheader("📝 Recent Queries")
    for q in reversed(st.session_state.query_history[-10:]):
        eval_badge = f"({q.get('eval_score', 0):.0%})" if q.get('eval_score') else ""
        persona_badge = q.get('persona', '')
        st.markdown(f"**{q['query'][:50]}...** - {persona_badge} {eval_badge} - {q['timestamp'].strftime('%H:%M:%S')}")


def render_settings_page():
    """Settings and configuration"""
    st.title("⚙️ Settings")
    
    st.subheader("Cache Management")
    if st.session_state.semantic_cache:
        stats = st.session_state.semantic_cache.get_stats()
        
        col1, col2 = st.columns(2)
        col1.metric("Cached Entries", stats['total_entries'])
        col2.metric("Total Hits", stats['total_hits'])
        
        if st.button("🗑️ Clear Cache"):
            st.session_state.semantic_cache.clear()
            st.success("Cache cleared")
    
    st.markdown("---")
    st.subheader("Reset System")
    
    if st.button("🔄 Reset All Data", type="secondary"):
        st.session_state.documents = []
        st.session_state.all_chunks = []
        st.session_state.search_engine = None
        st.session_state.query_history = []
        if st.session_state.metrics:
            st.session_state.metrics.reset()
        st.success("System reset complete")
        st.rerun()


if __name__ == "__main__":
    main()
