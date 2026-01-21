"""
[ELITE ARCHITECTURE] graph_viz.py
Interactive Knowledge Graph Visualizer using Plotly.
"""

import plotly.graph_objects as go
import streamlit as st
import networkx as nx

def render_knowledge_graph(graph_data: dict):
    """
    Renders an interactive 2D node-link diagram from the KnowledgeGraphEngine data.
    """
    st.subheader("🕸️ INTELLECTUAL_LANDSCAPE_v1")
    
    if not graph_data['nodes']:
        st.info("No semantic nodes detected in current corpus.")
        return

    # 1. Coordinate Generation (Spring Layout)
    G = nx.Graph()
    for edge in graph_data['edges']:
        G.add_edge(edge['source'], edge['target'])
    
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # 2. Edge Visualization
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # 3. Node Visualization
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in graph_data['nodes']:
        if node['id'] in pos:
            x, y = pos[node['id']]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"LABEL: {node['label']}<br>TYPE: {node['type']}")
            # Color coding
            color = '#ff4b4b' if node['type'] == 'document' else '#00d4ff'
            node_colors.append(color)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=15,
            line_width=2))

    # 4. Figure Assembly
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    template='plotly_dark'
                ))
    
    st.plotly_chart(fig, use_container_width=True)
