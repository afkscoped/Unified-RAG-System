"""
[ELITE ARCHITECTURE] quality_dashboard.py
Visualizes the Research Evaluation Matrix in Streamlit.
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import os
import json
from src.ui.enhanced_document_app.visualizations.graph_viz import render_knowledge_graph
from src.ui.enhanced_document_app.visualizations.embedding_viz import EmbeddingVisualizer
from src.analysis.predictive_metrics import DocumentProfiler

def render_quality_dashboard(eval_history: list, graph_data: dict = None, corpus_data: dict = None):
    """
    Renders the evaluation matrix and drift analysis.
    """
    st.header("📊 RESEARCH_MATRIX_01")
    st.markdown("---")
    
    if not eval_history:
        st.info("No evaluation data recorded. Run a query to generate matrix.")
        return

    # 1. Summary Metrics
    latest = eval_history[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FAITHFULNESS", f"{latest['faithfulness']:.2f}")
    c2.metric("RELEVANCE", f"{latest['relevance']:.2f}")
    c3.metric("PRECISION", f"{latest['context_precision']:.2f}")
    c4.metric("Q_OVERALL", f"{latest['overall_quality']:.2f}", delta=f"{latest['overall_quality'] - 0.5:.2f}")

    # 2. Polar Chart for Radar Analysis
    st.subheader("Semantic Stability Hexagon")
    df_radar = pd.DataFrame(dict(
        r=[latest['faithfulness'], latest['relevance'], latest['context_precision'], latest['overall_quality']],
        theta=['Faithfulness', 'Relevance', 'Precision', 'Overall']
    ))
    fig = px.line_polar(df_radar, r='r', theta='theta', line_close=True)
    st.plotly_chart(fig, use_container_width=True)

    # 3. Drift Analysis (Trend over time)
    st.subheader("Performance Drift (Temporal)")
    df_trend = pd.DataFrame(eval_history)
    st.line_chart(df_trend[['faithfulness', 'relevance', 'overall_quality']])
    
    # 4. Knowledge Graph
    if graph_data:
        render_knowledge_graph(graph_data)
        
    # 5. Corpus Analytics (Predictive)
    if corpus_data and 'embeddings' in corpus_data:
        st.markdown("---")
        st.subheader("🧪 INTELLIGENCE_PROFILING")
        
        # Predictive Metrics Column
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown("**Corpus Complexity**")
            # Analyze a sample of the text for metrics
            all_text = " ".join(corpus_data['texts'][:10]) if corpus_data.get('texts') else ""
            metrics = DocumentProfiler.calculate_metrics(all_text)
            st.write(f"Reading Level: Grade {metrics.get('reading_grade_level', 'N/A')}")
            st.write(f"Lexical Diversity: {metrics.get('lexical_diversity', 'N/A')}")
            st.write(f"Status: {metrics.get('complexity_label', 'Neutral')}")
            
        with c2:
            viz = EmbeddingVisualizer()
            viz.render_projection(
                embeddings=corpus_data['embeddings'],
                labels=corpus_data['texts'],
                source_files=corpus_data['sources']
            )
            
    # 6. Audit Log Explorer
    st.markdown("---")
    st.subheader("🕵️ SYSTEM_AUDIT_LOG_v1")
    log_path = "logs/audit.jsonl"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            logs = [json.loads(line) for line in f.readlines()]
        
        df_logs = pd.DataFrame(logs)
        if not df_logs.empty:
            df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'], unit='s')
            st.dataframe(df_logs.sort_values('timestamp', ascending=False), use_container_width=True)
    else:
        st.info("Log buffer is currently empty.")

if __name__ == "__main__":
    # Test stub
    render_quality_dashboard([{"faithfulness": 0.9, "relevance": 0.8, "context_precision": 0.7, "overall_quality": 0.8}])
