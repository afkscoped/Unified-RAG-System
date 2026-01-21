"""
Enhanced Results Viewer Component
"""
import streamlit as st

def render_enhanced_results(result_dict: dict):
    """
    Renders the unified result dictionary from EnhancedAnalyzer.
    Distinctly visualizes Local vs Web sources.
    """
    answer = result_dict.get('answer', '')
    sources = result_dict.get('sources', [])
    mode = result_dict.get('mode', 'docs')
    persona = result_dict.get('persona', 'scientist')
    
    metrics = result_dict.get('metrics', {})
    
    # 1. Answer with Persona Badge & Metrics
    st.markdown(f"""
    <div style="padding:10px; border-bottom: 2px solid #444; margin-bottom:15px; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="font-size:1.2em; margin-right:10px;">🛡️ <strong>Answer ({persona.title()})</strong></span>
            <span style="background:#333; padding:2px 8px; border-radius:4px; font-size:0.8em;">Mode: {mode.upper()}</span>
        </div>
        <div style="display: flex; gap: 10px;">
            <div title="Faithfulness" style="background: rgba(46, 139, 87, 0.2); border: 1px solid #2E8B57; padding: 2px 6px; border-radius: 4px; font-size: 0.7em;">🎯 F: {metrics.get('faithfulness', 'N/A')}</div>
            <div title="Relevance" style="background: rgba(30, 144, 255, 0.2); border: 1px solid #1E90FF; padding: 2px 6px; border-radius: 4px; font-size: 0.7em;">📈 R: {metrics.get('relevance', 'N/A')}</div>
            <div title="Clarity" style="background: rgba(255, 165, 0, 0.2); border: 1px solid #FFA500; padding: 2px 6px; border-radius: 4px; font-size: 0.7em;">💡 C: {metrics.get('clarity', 'N/A')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(answer)
    
    # 2. Sources Section
    if sources:
        st.divider()
        st.markdown("### 📚 Integrated Sources")
        
        for i, src in enumerate(sources, 1):
            # Styling based on type
            is_web = (src.type == 'web')
            color = "#1E90FF" if is_web else "#2E8B57" # Blue for Web, Green for Local
            icon = "🌐" if is_web else "📄"
            label = "WEB SEARCH" if is_web else "LOCAL DOC"
            
            with st.expander(f"{icon} Source {i}: {src.source}"):
                st.markdown(f"""
                <div style="border-left: 3px solid {color}; padding-left: 10px; margin-bottom: 10px;">
                    <strong style="color:{color};">{label}</strong> | Score: {src.score:.2f}
                    <p style="font-size:0.9em; margin-top:5px;">{src.content}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if src.url:
                    st.markdown(f"[View Original Source]({src.url})")
                    
    # 3. Adaptive Feedback
    if 'query' in result_dict:
        st.divider()
        f_cols = st.columns([0.7, 0.3])
        with f_cols[0]:
            st.caption("🧠 **Adaptive Learning**: Rate this answer to optimize future search weights.")
        with f_cols[1]:
            # Unique key generation
            u_key = str(hash(result_dict['query'] + answer))[:8]
            
            # Simple 5-star simulation using columns
            b_cols = st.columns(5)
            for i in range(1, 6):
                with b_cols[i-1]:
                    if st.button(f"{i}⭐", key=f"rate_{u_key}_{i}", help=f"Rate {i}/5"):
                        if 'rag_system' in st.session_state:
                            st.session_state.rag_system.rag.submit_feedback(
                                result_dict['query'],
                                i
                            )
                            st.toast(f"Feedback Recieved: {i}/5. Weights Updated.")
