"""
Embedding Visualizer: Reduces 384D/768D embeddings to 2D using UMAP
and renders an interactive scatter plot with Plotly.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from typing import List, Dict, Any

# Conditional UMAP import with fallback
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    UMAP = None


class EmbedVisualizer:
    """
    Handles dimension reduction and visualization of document vectors.
    """
    
    def __init__(self):
        if UMAP_AVAILABLE:
            self.reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        else:
            self.reducer = None

    def render(self, embeddings: np.ndarray, texts: List[str], sources: List[str]):
        """
        Renders the cluster map.
        """
        if not UMAP_AVAILABLE:
            st.warning("⚠️ UMAP not installed. Install with: `pip install umap-learn`")
            return
            
        if embeddings is None or len(embeddings) < 2:
            st.info("🌐 Need at least 2 documents to visualize clusters.")
            return

        with st.spinner("🌍 Reducing dimensions (UMAP)..."):

            try:
                # Reduce to 2D
                projections = self.reducer.fit_transform(embeddings)
                
                # Create DataFrame
                df = pd.DataFrame(projections, columns=['x', 'y'])
                df['text'] = [t[:100] + "..." for t in texts]
                df['source'] = sources
                
                # Plotly Scatter
                fig = px.scatter(
                    df, 
                    x='x', 
                    y='y', 
                    color='source',
                    hover_data=['text'],
                    title="Document Knowledge Map (UMAP Clusters)",
                    labels={'x': '', 'y': ''},
                    template="plotly_dark"
                )
                
                fig.update_layout(
                    showlegend=True,
                    xaxis_visible=False,
                    yaxis_visible=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Visualization error: {e}")

def render_knowledge_map(rag_system):
    """Entry point for UI integration"""
    try:
        # Get raw data from hybrid engine
        embeddings = rag_system.rag.search_engine.get_all_embeddings()
        texts = rag_system.rag.search_engine.get_all_texts()
        sources = rag_system.rag.search_engine.get_all_sources()
        
        if len(embeddings) > 0:
            viz = EmbedVisualizer()
            viz.render(embeddings, texts, sources)
        else:
            st.info("📚 No documents in index to visualize.")
    except Exception as e:
        st.error(f"Failed to load knowledge map: {e}")
