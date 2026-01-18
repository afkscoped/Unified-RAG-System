"""
Interactive Graph Visualizer

Creates interactive network visualizations of the story knowledge graph.
Uses Plotly for web-friendly, interactive graphs.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
from loguru import logger

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Graph visualization will be limited.")


@dataclass
class NodeStyle:
    """Styling for graph nodes."""
    color: str
    size: int
    symbol: str


class GraphVisualizer:
    """
    Creates interactive network visualizations from StoryKnowledgeGraph.
    
    Features:
    - Color-coded node types
    - Edge styles by relationship type
    - Temporal filtering by chapter
    - Multiple view modes
    - Interactive tooltips
    """
    
    # Node type styling
    NODE_STYLES = {
        "CHARACTER": NodeStyle("#3498db", 30, "circle"),      # Blue
        "LOCATION": NodeStyle("#27ae60", 25, "square"),       # Green
        "EVENT": NodeStyle("#e74c3c", 22, "diamond"),         # Red
        "ARTIFACT": NodeStyle("#f39c12", 20, "triangle-up"),  # Orange
        "FACTION": NodeStyle("#9b59b6", 25, "pentagon"),      # Purple
        "THEME": NodeStyle("#1abc9c", 18, "star"),            # Teal
        "DEFAULT": NodeStyle("#95a5a6", 20, "circle")         # Gray
    }
    
    # Edge type styling
    EDGE_STYLES = {
        "ALLIES_WITH": {"color": "#2ecc71", "dash": "solid", "width": 2},
        "CONFLICTS_WITH": {"color": "#e74c3c", "dash": "solid", "width": 3},
        "LOVES": {"color": "#e84393", "dash": "solid", "width": 2},
        "FEARS": {"color": "#636e72", "dash": "dash", "width": 1},
        "LOCATED_IN": {"color": "#74b9ff", "dash": "dot", "width": 1},
        "CAUSED_BY": {"color": "#fdcb6e", "dash": "solid", "width": 2},
        "FAMILY": {"color": "#a29bfe", "dash": "solid", "width": 2},
        "INTERACTS_WITH": {"color": "#b2bec3", "dash": "dot", "width": 1},
        "DEFAULT": {"color": "#636e72", "dash": "dot", "width": 1}
    }
    
    def __init__(self, story_graph=None):
        """
        Initialize graph visualizer.
        
        Args:
            story_graph: StoryKnowledgeGraph instance
        """
        self.story_graph = story_graph
        logger.info("GraphVisualizer initialized")
    
    def create_full_graph(
        self,
        max_chapter: Optional[int] = None,
        height: int = 600,
        width: int = 800
    ) -> Optional[go.Figure]:
        """
        Create full story graph visualization.
        
        Args:
            max_chapter: Only show entities up to this chapter
            height: Figure height in pixels
            width: Figure width in pixels
            
        Returns:
            Plotly Figure object
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        if not self.story_graph:
            return self._empty_figure("No story graph available", height, width)
        
        nodes = list(self.story_graph.graph.nodes())
        if not nodes:
            return self._empty_figure("No entities in graph yet", height, width)
        
        # Filter by chapter if specified
        if max_chapter:
            nodes = [n for n in nodes 
                    if self.story_graph.get_entity(n) 
                    and self.story_graph.get_entity(n).first_appearance <= max_chapter]
        
        if not nodes:
            return self._empty_figure(f"No entities by chapter {max_chapter}", height, width)
        
        # Calculate layout positions
        positions = self._calculate_layout(nodes)
        
        # Build node traces
        node_traces = self._build_node_traces(nodes, positions)
        
        # Build edge traces
        edge_traces = self._build_edge_traces(nodes, positions, max_chapter)
        
        # Create figure
        fig = go.Figure(data=edge_traces + node_traces)
        
        fig.update_layout(
            title="Story Knowledge Graph",
            showlegend=True,
            hovermode='closest',
            height=height,
            width=width,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def create_character_network(
        self,
        height: int = 500,
        width: int = 700
    ) -> Optional[go.Figure]:
        """Create character-only relationship network."""
        if not PLOTLY_AVAILABLE or not self.story_graph:
            return None
        
        # Filter to characters only
        characters = [n for n in self.story_graph.graph.nodes()
                     if self.story_graph.get_entity(n)
                     and self.story_graph.get_entity(n).type == 'CHARACTER']
        
        if not characters:
            return self._empty_figure("No characters tracked yet", height, width)
        
        positions = self._calculate_layout(characters)
        node_traces = self._build_node_traces(characters, positions)
        edge_traces = self._build_edge_traces(characters, positions, filter_nodes=characters)
        
        fig = go.Figure(data=edge_traces + node_traces)
        
        fig.update_layout(
            title="Character Relationship Network",
            showlegend=True,
            hovermode='closest',
            height=height,
            width=width,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def create_event_chain(
        self,
        height: int = 400,
        width: int = 800
    ) -> Optional[go.Figure]:
        """Create event causality chain visualization."""
        if not PLOTLY_AVAILABLE or not self.story_graph:
            return None
        
        # Filter to events only
        events = [n for n in self.story_graph.graph.nodes()
                 if self.story_graph.get_entity(n)
                 and self.story_graph.get_entity(n).type == 'EVENT']
        
        if not events:
            return self._empty_figure("No events tracked yet", height, width)
        
        # Sort by first appearance
        events.sort(key=lambda e: self.story_graph.get_entity(e).first_appearance if self.story_graph.get_entity(e) else 0)
        
        # Horizontal layout for timeline
        positions = {}
        for i, event in enumerate(events):
            positions[event] = (i * 2, 0)
        
        node_traces = self._build_node_traces(events, positions)
        edge_traces = self._build_edge_traces(events, positions, filter_nodes=events)
        
        fig = go.Figure(data=edge_traces + node_traces)
        
        fig.update_layout(
            title="Event Causality Chain",
            showlegend=False,
            hovermode='closest',
            height=height,
            width=width,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def create_location_map(
        self,
        height: int = 500,
        width: int = 700
    ) -> Optional[go.Figure]:
        """Create location-character co-occurrence map."""
        if not PLOTLY_AVAILABLE or not self.story_graph:
            return None
        
        # Get locations and characters
        locations = [n for n in self.story_graph.graph.nodes()
                    if self.story_graph.get_entity(n)
                    and self.story_graph.get_entity(n).type == 'LOCATION']
        
        characters = [n for n in self.story_graph.graph.nodes()
                     if self.story_graph.get_entity(n)
                     and self.story_graph.get_entity(n).type == 'CHARACTER']
        
        nodes = locations + characters
        
        if not nodes:
            return self._empty_figure("No locations tracked yet", height, width)
        
        positions = self._calculate_layout(nodes)
        node_traces = self._build_node_traces(nodes, positions)
        
        # Only show LOCATED_IN edges
        edge_traces = []
        for edge in self.story_graph.graph.edges(data=True):
            if edge[2].get('relation_type') == 'LOCATED_IN':
                if edge[0] in nodes and edge[1] in nodes:
                    pos0 = positions.get(edge[0], (0, 0))
                    pos1 = positions.get(edge[1], (0, 0))
                    edge_traces.append(go.Scatter(
                        x=[pos0[0], pos1[0]], y=[pos0[1], pos1[1]],
                        mode='lines',
                        line=dict(color='#74b9ff', width=1, dash='dot'),
                        hoverinfo='none',
                        showlegend=False
                    ))
        
        fig = go.Figure(data=edge_traces + node_traces)
        
        fig.update_layout(
            title="Location-Character Map",
            showlegend=True,
            height=height,
            width=width,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def get_graph_data_for_export(self) -> Dict:
        """Get graph data as JSON-serializable dict."""
        if not self.story_graph:
            return {"nodes": [], "edges": []}
        
        nodes = []
        for node_id in self.story_graph.graph.nodes():
            entity = self.story_graph.get_entity(node_id)
            if entity:
                nodes.append({
                    "id": node_id,
                    "type": entity.type,
                    "name": entity.name,
                    "first_appearance": entity.first_appearance,
                    "attributes": entity.attributes
                })
        
        edges = []
        for source, target, data in self.story_graph.graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "relation_type": data.get('relation_type', 'UNKNOWN'),
                "strength": data.get('strength', 0.5),
                "temporal_context": data.get('temporal_context', 1)
            })
        
        return {"nodes": nodes, "edges": edges}
    
    def _calculate_layout(self, nodes: List[str]) -> Dict[str, Tuple[float, float]]:
        """Calculate force-directed layout positions."""
        if not nodes:
            return {}
        
        # Simple circular layout
        n = len(nodes)
        positions = {}
        
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / n
            x = math.cos(angle) * 5
            y = math.sin(angle) * 5
            positions[node] = (x, y)
        
        # Apply simple force-directed adjustments
        for _ in range(50):  # Iterations
            forces = {n: [0.0, 0.0] for n in nodes}
            
            # Repulsion between nodes
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i+1:]:
                    dx = positions[n1][0] - positions[n2][0]
                    dy = positions[n1][1] - positions[n2][1]
                    dist = max(0.1, math.sqrt(dx*dx + dy*dy))
                    force = 1.0 / (dist * dist)
                    forces[n1][0] += dx * force * 0.1
                    forces[n1][1] += dy * force * 0.1
                    forces[n2][0] -= dx * force * 0.1
                    forces[n2][1] -= dy * force * 0.1
            
            # Attraction along edges
            if self.story_graph:
                for source, target in self.story_graph.graph.edges():
                    if source in positions and target in positions:
                        dx = positions[target][0] - positions[source][0]
                        dy = positions[target][1] - positions[source][1]
                        dist = max(0.1, math.sqrt(dx*dx + dy*dy))
                        force = (dist - 2) * 0.05
                        forces[source][0] += dx * force
                        forces[source][1] += dy * force
                        forces[target][0] -= dx * force
                        forces[target][1] -= dy * force
            
            # Apply forces
            for node in nodes:
                positions[node] = (
                    positions[node][0] + forces[node][0],
                    positions[node][1] + forces[node][1]
                )
        
        return positions
    
    def _build_node_traces(
        self,
        nodes: List[str],
        positions: Dict[str, Tuple[float, float]]
    ) -> List[go.Scatter]:
        """Build Plotly traces for nodes."""
        traces = []
        
        # Group nodes by type
        by_type = {}
        for node in nodes:
            entity = self.story_graph.get_entity(node) if self.story_graph else None
            node_type = entity.type if entity else "DEFAULT"
            if node_type not in by_type:
                by_type[node_type] = []
            by_type[node_type].append((node, entity))
        
        # Create trace for each type
        for node_type, node_list in by_type.items():
            style = self.NODE_STYLES.get(node_type, self.NODE_STYLES["DEFAULT"])
            
            x_vals = []
            y_vals = []
            texts = []
            hovers = []
            
            for node, entity in node_list:
                pos = positions.get(node, (0, 0))
                x_vals.append(pos[0])
                y_vals.append(pos[1])
                texts.append(entity.name if entity else node)
                
                hover = f"<b>{entity.name if entity else node}</b><br>"
                hover += f"Type: {node_type}<br>"
                if entity:
                    hover += f"Chapter: {entity.first_appearance}"
                hovers.append(hover)
            
            traces.append(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers+text',
                marker=dict(
                    size=style.size,
                    color=style.color,
                    symbol=style.symbol,
                    line=dict(width=2, color='white')
                ),
                text=texts,
                textposition='top center',
                hovertext=hovers,
                hoverinfo='text',
                name=node_type,
                showlegend=True
            ))
        
        return traces
    
    def _build_edge_traces(
        self,
        nodes: List[str],
        positions: Dict[str, Tuple[float, float]],
        max_chapter: Optional[int] = None,
        filter_nodes: Optional[List[str]] = None
    ) -> List[go.Scatter]:
        """Build Plotly traces for edges."""
        traces = []
        
        if not self.story_graph:
            return traces
        
        node_set = set(filter_nodes or nodes)
        
        # Group edges by type
        by_type = {}
        for source, target, data in self.story_graph.graph.edges(data=True):
            if source not in node_set or target not in node_set:
                continue
            
            if max_chapter and data.get('temporal_context', 0) > max_chapter:
                continue
            
            rel_type = data.get('relation_type', 'DEFAULT')
            if rel_type not in by_type:
                by_type[rel_type] = []
            by_type[rel_type].append((source, target, data))
        
        # Create traces
        for rel_type, edges in by_type.items():
            style = self.EDGE_STYLES.get(rel_type, self.EDGE_STYLES["DEFAULT"])
            
            for source, target, data in edges:
                pos_s = positions.get(source, (0, 0))
                pos_t = positions.get(target, (0, 0))
                
                traces.append(go.Scatter(
                    x=[pos_s[0], pos_t[0]],
                    y=[pos_s[1], pos_t[1]],
                    mode='lines',
                    line=dict(
                        color=style["color"],
                        width=style["width"],
                        dash=style["dash"]
                    ),
                    hoverinfo='text',
                    hovertext=f"{source} → {rel_type} → {target}",
                    showlegend=False
                ))
        
        return traces
    
    def _empty_figure(self, message: str, height: int, width: int) -> go.Figure:
        """Create empty figure with message."""
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=height,
            width=width,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig
