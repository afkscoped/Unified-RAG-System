"""
Multi-Hop Relationship Discovery Engine

Discovers complex relationships via graph traversal including:
- Indirect connections (N-hop paths)
- Conflict triangles (dramatic tension)
- Causal chains
- Hidden connections for plot revelations
"""

import networkx as nx
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class RelationshipPath:
    """Represents a multi-hop relationship chain."""
    source: str
    target: str
    path: List[str]  # Node sequence
    relationships: List[str]  # Edge types in sequence
    total_hops: int
    narrative_weight: float  # Combined strength along path
    story_implication: str  # Human-readable meaning
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "path": self.path,
            "relationships": self.relationships,
            "total_hops": self.total_hops,
            "narrative_weight": self.narrative_weight,
            "story_implication": self.story_implication
        }


class MultiHopDiscoveryEngine:
    """
    Discovers complex relationships via graph traversal.
    
    Provides methods to find indirect connections, conflict triangles,
    causal chains, and suggest plot complications.
    """
    
    def __init__(self, story_graph):
        """
        Initialize discovery engine.
        
        Args:
            story_graph: StoryKnowledgeGraph instance
        """
        self.story_graph = story_graph
        self.discovery_cache: Dict[str, List[RelationshipPath]] = {}
        
        logger.info("MultiHopDiscoveryEngine initialized")
        
    def discover_indirect_relationships(
        self,
        character_id: str,
        max_hops: int = 3,
        min_strength: float = 0.3
    ) -> List[RelationshipPath]:
        """
        Find all multi-hop relationships for a character.
        
        Args:
            character_id: Starting character ID
            max_hops: Maximum path length
            min_strength: Minimum combined path strength
            
        Returns:
            List of RelationshipPath objects sorted by narrative weight
        """
        cache_key = f"{character_id}_{max_hops}_{min_strength}"
        if cache_key in self.discovery_cache:
            return self.discovery_cache[cache_key]
        
        discovered_paths = []
        graph = self.story_graph.graph
        
        if character_id not in graph:
            return []
        
        # Use BFS to explore up to max_hops
        for target in graph.nodes():
            if target == character_id:
                continue
            
            try:
                # Find all simple paths (no cycles)
                all_paths = nx.all_simple_paths(
                    graph,
                    character_id,
                    target,
                    cutoff=max_hops
                )
                
                for path in all_paths:
                    if len(path) > 2:  # Only multi-hop (2+ edges)
                        rel_path = self._build_relationship_path(path, graph)
                        if rel_path and rel_path.narrative_weight >= min_strength:
                            discovered_paths.append(rel_path)
                            
            except nx.NetworkXNoPath:
                continue
            except nx.NodeNotFound:
                continue
        
        # Sort by narrative weight
        discovered_paths.sort(key=lambda p: p.narrative_weight, reverse=True)
        
        self.discovery_cache[cache_key] = discovered_paths
        return discovered_paths
    
    def _build_relationship_path(
        self,
        path: List[str],
        graph: nx.MultiDiGraph
    ) -> Optional[RelationshipPath]:
        """Build RelationshipPath from node sequence."""
        rel_types = []
        total_strength = 1.0
        
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                # Get first edge's data (MultiDiGraph can have multiple)
                first_edge = list(edge_data.values())[0]
                rel_types.append(first_edge.get('relation', 'UNKNOWN'))
                total_strength *= first_edge.get('strength', 0.5)
            else:
                return None
        
        return RelationshipPath(
            source=path[0],
            target=path[-1],
            path=path,
            relationships=rel_types,
            total_hops=len(path) - 1,
            narrative_weight=total_strength,
            story_implication=self._interpret_path(path, rel_types, graph)
        )
    
    def _interpret_path(
        self,
        path: List[str],
        relationships: List[str],
        graph: nx.MultiDiGraph
    ) -> str:
        """Generate human-readable interpretation of relationship chain."""
        # Get character names
        node_names = []
        for node_id in path:
            node_data = graph.nodes.get(node_id, {})
            node_names.append(node_data.get('name', node_id))
        
        if len(relationships) == 2:
            # Two-hop patterns with interpretations
            pattern = f"{relationships[0]}->{relationships[1]}"
            
            interpretations = {
                "ALLIES_WITH->CONFLICTS_WITH": 
                    f"{node_names[0]}'s ally {node_names[1]} is enemies with {node_names[2]} (potential loyalty conflict)",
                "CONFLICTS_WITH->ALLIES_WITH": 
                    f"{node_names[0]}'s enemy {node_names[1]} is allied with {node_names[2]} (common threat)",
                "MEMBER_OF->CONFLICTS_WITH": 
                    f"{node_names[0]} belongs to {node_names[1]} which opposes {node_names[2]}",
                "ALLIES_WITH->ALLIES_WITH":
                    f"{node_names[0]} and {node_names[2]} share a mutual friend: {node_names[1]}",
                "CONFLICTS_WITH->CONFLICTS_WITH":
                    f"{node_names[0]} and {node_names[2]} share a common enemy: {node_names[1]}",
            }
            
            return interpretations.get(
                pattern,
                f"{node_names[0]} connected to {node_names[-1]} through {node_names[1]}"
            )
        
        elif len(relationships) == 3:
            return f"Complex chain: {' -> '.join(node_names)}"
        
        else:
            intermediates = ', '.join(node_names[1:-1])
            return f"{len(relationships)}-hop connection through {intermediates}"
    
    def find_conflict_triangles(self) -> List[Dict]:
        """
        Discover dramatic tension triangles.
        
        Pattern: A allies with B, B allies with C, but A conflicts with C
        Classic "enemy of my friend" scenario.
        
        Returns:
            List of triangle dictionaries with tension descriptions
        """
        triangles = []
        graph = self.story_graph.graph
        
        # Get all character nodes
        characters = [
            n for n, d in graph.nodes(data=True)
            if d.get('type') == 'CHARACTER'
        ]
        
        for node_a in characters:
            # Find A's allies
            allies_a = self._get_related_nodes(node_a, "ALLIES_WITH")
            # Find A's enemies
            enemies_a = self._get_related_nodes(node_a, "CONFLICTS_WITH")
            
            for node_b in allies_a:
                if node_b not in graph:
                    continue
                # Find B's allies
                allies_b = self._get_related_nodes(node_b, "ALLIES_WITH")
                
                # Check for overlap with A's enemies
                conflict_nodes = set(allies_b) & set(enemies_a)
                
                for node_c in conflict_nodes:
                    # Avoid duplicate triangles
                    triangle_key = tuple(sorted([node_a, node_b, node_c]))
                    if any(t.get('key') == triangle_key for t in triangles):
                        continue
                    
                    name_a = graph.nodes[node_a].get('name', node_a)
                    name_b = graph.nodes[node_b].get('name', node_b)
                    name_c = graph.nodes[node_c].get('name', node_c)
                    
                    triangles.append({
                        "key": triangle_key,
                        "type": "ally_enemy_triangle",
                        "node_a": name_a,
                        "node_b": name_b,
                        "node_c": name_c,
                        "tension": (
                            f"{name_a} is allied with {name_b}, "
                            f"who is allied with {name_c}, "
                            f"but {name_a} conflicts with {name_c}"
                        ),
                        "narrative_potential": "high"
                    })
        
        return triangles
    
    def _get_related_nodes(self, node: str, relation_type: str) -> List[str]:
        """Get all nodes connected via specific relation type."""
        related = []
        graph = self.story_graph.graph
        
        for neighbor in graph.neighbors(node):
            edge_data = graph.get_edge_data(node, neighbor)
            if edge_data:
                for key, attrs in edge_data.items():
                    if attrs.get('relation') == relation_type:
                        related.append(neighbor)
                        break
        
        return related
    
    def discover_causal_chains(
        self,
        event_id: str,
        max_depth: int = 5
    ) -> List[Dict]:
        """
        Trace causality: What events led to this event?
        
        Follows CAUSED_BY edges backwards from the event.
        
        Args:
            event_id: Event to trace causes for
            max_depth: Maximum chain length
            
        Returns:
            List of causal chain dictionaries
        """
        logger.info(f"Tracing causal chains for event: {event_id} (max_depth={max_depth})")
        causal_chains = []
        graph = self.story_graph.graph
        
        if event_id not in graph:
            logger.warning(f"Event ID {event_id} not found in graph")
            return []
        
        def trace_backwards(current_event: str, chain: List[str], depth: int):
            if depth >= max_depth:
                if len(chain) > 1:
                    logger.debug(f"Reached max depth at {current_event}, chain: {chain}")
                    causal_chains.append(chain.copy())
                return
            
            # Find predecessor events via CAUSED_BY
            predecessors = []
            for pred in graph.predecessors(current_event):
                edge_data = graph.get_edge_data(pred, current_event)
                if edge_data:
                    for key, attrs in edge_data.items():
                        if attrs.get('relation') == 'CAUSED_BY':
                            predecessors.append(pred)
                            break
            
            if not predecessors:
                # Reached a root cause
                if len(chain) > 1:
                    logger.debug(f"Reached root cause at {current_event}, chain: {chain}")
                    causal_chains.append(chain.copy())
                else:
                    logger.debug(f"No predecessors for {current_event}, no chain formed")
                return
            
            for pred in predecessors:
                # Avoid cycles
                if pred in chain:
                    logger.warning(f"Cycle detected in causal chain: {pred} -> ... -> {current_event}")
                    continue
                    
                chain.append(pred)
                trace_backwards(pred, chain, depth + 1)
                chain.pop()
        
        # Start traversal
        trace_backwards(event_id, [event_id], 0)
        
        # Format results
        formatted_chains = []
        for chain in causal_chains:
            event_names = [
                graph.nodes[e].get('name', e) 
                for e in reversed(chain)
            ]
            formatted_chains.append({
                "causal_sequence": event_names,
                "depth": len(chain) - 1,
                "root_cause": event_names[0],
                "final_outcome": event_names[-1],
                "narrative": " led to ".join(event_names)
            })
        
        logger.info(f"Discovered {len(formatted_chains)} causal chains for {event_id}")
        return formatted_chains
    
    def find_hidden_connections(
        self,
        char_a: str,
        char_b: str,
        max_hops: int = 4
    ) -> List[RelationshipPath]:
        """
        Find non-obvious connections between two characters.
        
        Useful for foreshadowing or plot revelations.
        
        Args:
            char_a: First character ID
            char_b: Second character ID
            max_hops: Maximum path length
            
        Returns:
            List of indirect paths between the characters
        """
        all_paths = []
        graph = self.story_graph.graph
        
        if char_a not in graph or char_b not in graph:
            return []
        
        try:
            paths = nx.all_simple_paths(
                graph,
                char_a,
                char_b,
                cutoff=max_hops
            )
            
            for path in paths:
                if len(path) > 2:  # Only indirect connections
                    rel_path = self._build_relationship_path(path, graph)
                    if rel_path:
                        all_paths.append(rel_path)
                        
        except nx.NetworkXNoPath:
            pass
        except nx.NodeNotFound:
            pass
        
        # Sort by narrative weight (decay with distance)
        all_paths.sort(key=lambda p: p.narrative_weight, reverse=True)
        return all_paths
    
    def suggest_plot_complications(self, current_chapter: int) -> List[Dict]:
        """
        Analyze graph to suggest interesting plot developments.
        
        Based on relationship tensions and unexplored connections.
        
        Args:
            current_chapter: Current story chapter
            
        Returns:
            List of plot suggestions with priority and details
        """
        suggestions = []
        graph = self.story_graph.graph
        
        # 1. Find conflict triangles
        triangles = self.find_conflict_triangles()
        for triangle in triangles[:3]:  # Top 3
            suggestions.append({
                "type": "character_tension",
                "priority": "high",
                "suggestion": (
                    f"Explore the tension between {triangle['node_a']}, "
                    f"{triangle['node_b']}, and {triangle['node_c']}"
                ),
                "details": triangle['tension']
            })
        
        # 2. Find characters who haven't interacted yet
        characters = [
            n for n, d in graph.nodes(data=True) 
            if d.get('type') == 'CHARACTER'
        ]
        
        for i, char_a in enumerate(characters):
            for char_b in characters[i + 1:]:
                # Check if they have direct relationship
                has_direct = (
                    graph.has_edge(char_a, char_b) or 
                    graph.has_edge(char_b, char_a)
                )
                
                if not has_direct:
                    # Are they connected indirectly?
                    hidden = self.find_hidden_connections(char_a, char_b, max_hops=3)
                    if hidden:
                        name_a = graph.nodes[char_a].get('name', char_a)
                        name_b = graph.nodes[char_b].get('name', char_b)
                        suggestions.append({
                            "type": "reveal_connection",
                            "priority": "medium",
                            "suggestion": f"Reveal connection between {name_a} and {name_b}",
                            "details": hidden[0].story_implication
                        })
        
        # 3. Find dangling plot threads (events with no consequences)
        events = [
            n for n, d in graph.nodes(data=True) 
            if d.get('type') == 'EVENT'
        ]
        
        for event in events:
            successors = list(graph.successors(event))
            event_data = graph.nodes[event]
            
            if not successors:
                suggestions.append({
                    "type": "unresolved_event",
                    "priority": "low",
                    "suggestion": f"Resolve consequences of '{event_data.get('name', event)}'",
                    "details": "This event has had no follow-up yet"
                })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: priority_order.get(s['priority'], 3))
        
        return suggestions
    
    def clear_cache(self):
        """Clear discovery cache."""
        self.discovery_cache.clear()
