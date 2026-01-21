"""
Causal Inference Engine

Infers causal relationships between events based on:
- Direct linguistic markers (explicit causality)
- Temporal sequence (post hoc ergo propter hoc heuristic)
- Common actors and locations
- Logical preconditions
"""

import networkx as nx
from typing import List, Dict, Optional, Tuple
from loguru import logger
from .story_graph import StoryKnowledgeGraph, StoryRelationship

class CausalInferenceEngine:
    """
    Analyzes story graph to infer missing causal links between events.
    """
    
    def __init__(self, graph: StoryKnowledgeGraph):
        self.story_graph = graph
        
    def infer_causality(self, events: List[Dict]) -> List[StoryRelationship]:
        """
        Infer causal relationships between a list of events.
        
        Args:
            events: List of event dictionaries (must be time-sorted)
            
        Returns:
            List of inferred StoryRelationship objects
        """
        relationships = []
        
        # Sort events by chapter then index if available
        # Assuming events list is already roughly sorted or we rely on chapter
        sorted_events = sorted(events, key=lambda e: (e.get('chapter', 1), e.get('id', '')))
        
        for i in range(len(sorted_events) - 1):
            cause = sorted_events[i]
            
            # Look ahead at subsequent events
            for j in range(i + 1, min(i + 5, len(sorted_events))): # Look ahead window
                effect = sorted_events[j]
                
                # 1. Temporal Check (Basic requirement)
                if not self._is_temporally_valid(cause, effect):
                    continue
                    
                confidence = 0.0
                reasons = []
                
                # 2. Direct Linguistic Link (Strongest)
                # Check if effect description references cause
                if cause['name'].lower() in effect['description'].lower():
                    confidence += 0.6
                    reasons.append("Explicit reference")
                    
                # 3. Common Actors (Medium)
                # If same characters involved, likely related
                common_actors = self._get_common_actors(cause['id'], effect['id'])
                if common_actors:
                    confidence += 0.3 * min(len(common_actors), 3)
                    reasons.append(f"Shared actors: {len(common_actors)}")
                    
                # 4. Location Continuity (Weak)
                if self._get_event_location(cause['id']) == self._get_event_location(effect['id']):
                    confidence += 0.1
                    reasons.append("Same location")
                    
                # Threshold for inference
                if confidence >= 0.5:
                    relationships.append(StoryRelationship(
                        source=cause['id'],
                        target=effect['id'],
                        relation_type="CAUSED_BY",
                        strength=min(confidence, 1.0),
                        temporal_context=effect.get('chapter', 1),
                        metadata={
                            "inference_method": "causal_engine",
                            "reasoning": ", ".join(reasons)
                        }
                    ))
                    
        return relationships
    
    def _is_temporally_valid(self, cause: Dict, effect: Dict) -> bool:
        """Check if cause happens before or same time as effect."""
        c_chap = cause.get('chapter', 1)
        e_chap = effect.get('chapter', 1)
        return c_chap <= e_chap
    
    def _get_common_actors(self, event_a_id: str, event_b_id: str) -> List[str]:
        """Find characters involved in both events."""
        # This requires querying the graph for participants
        # Assuming PARTICIPATED_IN edges exist or can be inferred
        graph = self.story_graph.graph
        
        actors_a = set()
        if event_a_id in graph:
            # Find predecessors (Characters -> PARTICIPATED_IN -> Event)
            # Or successors (Event -> INVOLVES -> Character)?
            # Usually graph schema is Entity -> RELATION -> Entity
            # Let's check neighbors
            for neighbor in graph.neighbors(event_a_id):
                # Check edge type
                pass
            # Also check incoming
            for pred in graph.predecessors(event_a_id):
                edge_data = graph.get_edge_data(pred, event_a_id)
                # Assuming simple check for now or specific relation types
                actors_a.add(pred)
                
        actors_b = set()
        if event_b_id in graph:
            for pred in graph.predecessors(event_b_id):
                actors_b.add(pred)
                
        return list(actors_a.intersection(actors_b))

    def _get_event_location(self, event_id: str) -> Optional[str]:
        """Get location of an event."""
        graph = self.story_graph.graph
        if event_id not in graph:
            return None
            
        # Check outgoing edges for LOCATED_IN
        for neighbor in graph.neighbors(event_id):
            edge_data = graph.get_edge_data(event_id, neighbor)
            for key, attrs in edge_data.items():
                if attrs.get('relation') == 'LOCATED_IN':
                    return neighbor
        return None
