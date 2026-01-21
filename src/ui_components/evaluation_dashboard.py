"""
Evaluation Dashboard: Visualizes RAG performance metrics using Plotly.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any

from .embedding_viz import render_knowledge_map

def render_evaluation_dashboard(rag_system):
    """
    Renders an interactive dashboard based on history.
    """
    query_history = rag_system.query_history
    if not query_history:
        st.info("📊 No evaluation data available yet. Start querying to see analytics!")
        # Still show Knowledge map if indexed
        if rag_system.rag._indexed:
            st.divider()
            render_knowledge_map(rag_system)
        return

    # Convert history to DataFrame
    df = pd.json_normalize(query_history)
    
    # Pre-process: handle metrics columns
    # Ensure mandatory columns exist to avoid KeyErrors
    for col in ['metrics.faithfulness', 'metrics.relevance', 'metrics.clarity', 'metrics.latency']:
        if col not in df.columns:
            df[col] = 0.0
    
    # Fill NaN with 0
    df = df.fillna(0)
    
    st.markdown("## 📊 Performance Analytics")
    
    # 1. Row 1: Key Metrics Summary
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Queries", len(df))
    with cols[1]:
        avg_faith = df['metrics.faithfulness'].mean()
        st.metric("Avg Faithfulness", f"{avg_faith:.1f}/10")
    with cols[2]:
        avg_lat = df['metrics.latency'].mean()
        st.metric("Avg Eval Latency", f"{avg_lat:.0f}ms")
    with cols[3]:
        latest_persona = df['persona'].iloc[-1].title()
        st.metric("Active Persona", latest_persona)

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Agent Comparison", "⏱️ Latency Trends", "📚 Source Analysis", "🌐 Knowledge Topology"])

    with tab1:
        st.markdown("### Agent Radar (Personality Profile)")
        # Group by persona and calculate averages
        agent_stats = df.groupby('persona')[['metrics.faithfulness', 'metrics.relevance', 'metrics.clarity']].mean().reset_index()
        
        # Plotly Radar Chart
        fig = go.Figure()
        
        for _, row in agent_stats.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['metrics.faithfulness'], row['metrics.relevance'], row['metrics.clarity'], row['metrics.faithfulness']],
                theta=['Faithfulness', 'Relevance', 'Clarity', 'Faithfulness'],
                fill='toself',
                name=row['persona'].title()
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Evaluation Latency over Time")
        # Line chart for latency
        df['time_str'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%H:%M:%S')
        fig_lat = px.line(
            df, 
            x='time_str', 
            y='metrics.latency', 
            color='persona',
            labels={'metrics.latency': 'Latency (ms)', 'time_str': 'Time'},
            markers=True
        )
        fig_lat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_lat, use_container_width=True)

    with tab3:
        st.markdown("### Source Distribution")
        # Pie chart for Duck/Web distribution or source counts
        fig_src = px.pie(
            df, 
            names='mode', 
            title='Search Mode Usage',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_src, use_container_width=True)

    with tab4:
        st.markdown("### Document Topology")
        render_knowledge_map(rag_system)

    # 3. Raw Data (Optional Expander)
    with st.expander("📝 View Raw Evaluation Logs"):
        st.dataframe(df[['timestamp', 'query', 'persona', 'mode', 'metrics.faithfulness', 'metrics.relevance', 'metrics.clarity']])
