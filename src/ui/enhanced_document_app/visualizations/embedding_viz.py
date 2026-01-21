"""
[ELITE ARCHITECTURE] embedding_viz.py
Visualizes the high-dimensional document space in 2D.
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

class EmbeddingVisualizer:
    """
    Innovation: Knowledge Mapping.
    Projects high-dimensional embeddings into a 2D plane to show 
    topical clusters and outliers.
    """
    
    def __init__(self):
        pass

    def render_projection(self, embeddings: np.ndarray, labels: list, source_files: list):
        """
        Calculates and renders the projection.
        """
        st.subheader("🌐 KNOWLEDGE_CLUSTER_MAP")
        
        if embeddings is None or len(embeddings) < 3:
            st.info("Insufficient vectors for spatial projection (need at least 3).")
            return

        # 1. Dimensionality Reduction
        method = "UMAP" if HAS_UMAP else "PCA"
        st.caption(f"Using {method} for spatial projection.")
        
        if method == "UMAP":
            reducer = UMAP(n_neighbors=min(15, len(embeddings)-1), n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)
            
        projections = reducer.fit_transform(embeddings)

        # 2. DataFrame Construction
        df = pd.DataFrame({
            'x': projections[:, 0],
            'y': projections[:, 1],
            'snippet': [l[:100] + "..." for l in labels],
            'source': source_files
        })

        # 3. Plotting
        fig = px.scatter(
            df, x='x', y='y', 
            hover_data=['snippet'], 
            color='source',
            template='plotly_dark',
            title=f"Semantic Document Landscape ({method})"
        )
        
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig, use_container_width=True)
