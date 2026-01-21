"""
[ELITE ARCHITECTURE] query_interface.py
High-Fidelity Query and Reasoning Interface.
"""

import streamlit as st
from loguru import logger
from src.elite.manuscript_manager import ManuscriptManager

def render_query_interface(orchestrator):
    st.header("🔍 RESEARCH_QUERY_TERMINAL")
    st.markdown("---")
    
    # Init Manuscript Manager
    manuscript = ManuscriptManager()
    
    # Sidebar Controls
    with st.sidebar:
        st.markdown("### 🔬 Protocol Settings")
        deep_audit = st.toggle("🛡️ Deep Audit Protocol", help="Enables Adversarial Web Search", value=False)
        
        st.divider()
        st.markdown("### 📜 Living Manuscript")
        st.download_button(
            "Export Markdown Draft",
            data=manuscript.get_markdown(),
            file_name="research_draft.md",
            mime="text/markdown"
        )
        if st.button("🗑️ Clear Draft"):
            manuscript.clear()
            st.rerun()
    
    # Session state for history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 1. Main Input
    query = st.chat_input("Enter research objective or query...")

    if query:
        msg_label = "Executing Adversarial Reasoning..." if deep_audit else "Executing Reasoning Protocol..."
        with st.status(msg_label, expanded=True) as status:
            st.write("Checking Multi-Level Cache...")
            if deep_audit:
                 st.write("📡 Scanning Web for Conflicting Evidence...")
                 st.write("⚔️ Generating Thesis & Antithesis...")
            else:
                 st.write("Synthesizing Hypothetical Context (HyDE)...")
                 st.write("Reranking Candidates with Cross-Attention...")
            
            # Execute Core Orchestration
            result = orchestrator.query(query, deep_audit=deep_audit)
            
            st.session_state.chat_history.append(result)
            status.update(label="REASONING_COMPLETE", state="complete", expanded=False)

    # 2. Display Results (Latest first)
    for i, res in enumerate(reversed(st.session_state.chat_history)):
        with st.container(border=True):
            st.markdown(f"**Q: {res['query']}**")
            
            # Thought Trace (Expandable) & Debate Trace
            debate = res.get('debate_trace', {})
            if debate and debate.get('is_debate'):
                with st.expander("⚔️ ADVERSARIAL DEBATE TRACE", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### ✅ Proponent (Local)")
                        st.info(debate.get('thesis', 'N/A'))
                    with col2:
                        st.markdown("#### ⚠️ Skeptic (Web)")
                        st.warning(debate.get('antithesis', 'N/A'))
            elif res.get('thought'):
                with st.expander("👁️ SHOW_REASONING_TRACE"):
                    st.code(res['thought'], language="markdown")
            
            # Final Answer
            st.markdown(res['answer'])
            
            # Citations (Pills)
            if res.get('citations'):
                st.markdown("---")
                cols = st.columns(len(res['citations'][:4]) if res['citations'] else 1)
                for i, cite in enumerate(res['citations'][:4]):
                    with cols[i]:
                        st.caption(f"📍 [Source {cite['id']}]")
                        st.caption(f"{cite['file']} (P.{cite['page']})")
            
            # Quality Score (Small)
            eval_data = res.get('evaluation', {})
            st.caption(f"Accuracy Confidence: {eval_data.get('overall_quality', 0):.2f} | Groundedness: {eval_data.get('faithfulness', 0):.2f}")
            
            # Add to Manuscript Button (Unique Key per message)
            if st.button("📌 Add to Research Draft", key=f"add_draft_{i}"):
                manuscript.add_section(
                    title=f"Insight from: {res['query']}",
                    content=res['answer'],
                    source_query=res['query']
                )
                st.toast("Snippet added to manuscript!")

    if not st.session_state.chat_history:
        st.info("Input a query to initiate semantic search.")
