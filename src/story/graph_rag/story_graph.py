"""
Story Knowledge Graph

Core knowledge graph implementation for narrative generation.
Tracks entities, relationships, and temporal states across story chapters.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


@dataclass
class StoryEntity:
    """Represents a story element node in the knowledge graph."""
    id: str
    type: str  # CHARACTER, LOCATION, EVENT, THEME, ARTIFACT, FACTION
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    first_appearance: int = 1  # Chapter number
    last_modified: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "attributes": self.attributes,
            "first_appearance": self.first_appearance,
            "last_modified": self.last_modified.isoformat()
        }


@dataclass
class StoryRelationship:
    """Represents an edge between entities in the knowledge graph."""
    source: str
    target: str
    relation_type: str  # ALLIES_WITH, CONFLICTS_WITH, LOCATED_IN, CAUSED_BY, etc.
    strength: float = 0.5  # 0.0 to 1.0
    temporal_context: int = 1  # Chapter where relationship established
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "strength": self.strength,
            "temporal_context": self.temporal_context,
            "metadata": self.metadata
        }


class StoryKnowledgeGraph:
    """
    Graph RAG implementation for narrative generation.
    
    Uses NetworkX MultiDiGraph to support multiple relationship types
    between the same entities with temporal context.
    """
    
    def __init__(self, embedding_manager=None):
        """
        Initialize the story knowledge graph.
        
        Args:
            embedding_manager: Optional EmbeddingManager for semantic operations
        """
        self.graph = nx.MultiDiGraph()
        self.embedder = embedding_manager
        self.entity_index: Dict[str, StoryEntity] = {}  # Fast lookup
        self.chapter_timeline: Dict[int, List[str]] = {}  # chapter -> event IDs
        
        logger.info("StoryKnowledgeGraph initialized")
        
    def add_entity(self, entity: StoryEntity) -> None:
        """
        Add or update an entity node in the graph.
        
        Args:
            entity: StoryEntity to add
        """
        self.graph.add_node(
            entity.id,
            type=entity.type,
            name=entity.name,
            attributes=entity.attributes,
            embedding=entity.embedding,
            first_appearance=entity.first_appearance,
            last_modified=entity.last_modified.isoformat()
        )
        self.entity_index[entity.id] = entity
        
        # Track events in timeline
        if entity.type == "EVENT":
            chapter = entity.first_appearance
            if chapter not in self.chapter_timeline:
                self.chapter_timeline[chapter] = []
            if entity.id not in self.chapter_timeline[chapter]:
                self.chapter_timeline[chapter].append(entity.id)
        
        logger.debug(f"Added entity: {entity.id} ({entity.type})")
        
    def add_relationship(self, rel: StoryRelationship) -> None:
        """
        Add a relationship edge with temporal metadata.
        
        Args:
            rel: StoryRelationship to add
        """
        self.graph.add_edge(
            rel.source,
            rel.target,
            relation=rel.relation_type,
            strength=rel.strength,
            chapter=rel.temporal_context,
            **rel.metadata
        )
        logger.debug(f"Added relationship: {rel.source} --{rel.relation_type}--> {rel.target}")
        
    def get_entity(self, entity_id: str) -> Optional[StoryEntity]:
        """Get entity by ID."""
        return self.entity_index.get(entity_id)
    
    def get_entity_context(self, entity_id: str, max_hops: int = 2) -> Optional[Dict]:
        """
        Retrieve entity with its relationship neighborhood.
        
        Args:
            entity_id: ID of the central entity
            max_hops: Maximum traversal depth
            
        Returns:
            Dictionary containing entity data and multi-hop neighbors
        """
        if entity_id not in self.graph:
            return None
            
        # Get central entity data
        entity_data = dict(self.graph.nodes[entity_id])
        
        # Multi-hop traversal
        neighbors = {}
        current_frontier = {entity_id}
        visited = {entity_id}
        
        for hop in range(1, max_hops + 1):
            next_frontier = set()
            hop_neighbors = []
            
            for node in current_frontier:
                # Get outgoing edges
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        edge_data = self.graph.get_edge_data(node, neighbor)
                        hop_neighbors.append({
                            "id": neighbor,
                            "data": dict(self.graph.nodes[neighbor]),
                            "edge": edge_data,
                            "from_node": node
                        })
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
                        
                # Get incoming edges
                for predecessor in self.graph.predecessors(node):
                    if predecessor not in visited:
                        edge_data = self.graph.get_edge_data(predecessor, node)
                        hop_neighbors.append({
                            "id": predecessor,
                            "data": dict(self.graph.nodes[predecessor]),
                            "edge": edge_data,
                            "from_node": node,
                            "direction": "incoming"
                        })
                        next_frontier.add(predecessor)
                        visited.add(predecessor)
            
            if hop_neighbors:
                neighbors[f"hop_{hop}"] = hop_neighbors
            current_frontier = next_frontier
        
        return {
            "entity": entity_data,
            "relationships": neighbors,
            "total_connections": len(visited) - 1
        }
    
    def query_by_relation_path(
        self, 
        start: str, 
        path: List[str]
    ) -> List[Dict]:
        """
        Traverse specific relationship path from start entity.
        
        Example: 
            query_by_relation_path("marcus", ["MEMBER_OF", "CONFLICTS_WITH"])
            Returns entities reachable via: Marcus -MEMBER_OF-> X -CONFLICTS_WITH-> Y
            
        Args:
            start: Starting entity ID
            path: List of relationship types to follow
            
        Returns:
            List of entities reachable via the specified path
        """
        if start not in self.graph:
            return []
            
        results = []
        current_nodes = [start]
        traversed_path = []
        
        for relation in path:
            next_nodes = []
            traversed_path.append(relation)
            
            for node in current_nodes:
                for neighbor in self.graph.neighbors(node):
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    # Check if any edge has matching relation type
                    for edge_key, edge_attrs in edge_data.items():
                        if edge_attrs.get('relation') == relation:
                            next_nodes.append(neighbor)
                            results.append({
                                "node": neighbor,
                                "data": dict(self.graph.nodes[neighbor]),
                                "path": traversed_path.copy(),
                                "via": node
                            })
                            
            current_nodes = list(set(next_nodes))  # Deduplicate
            
        return results
    
    def get_temporal_state(self, chapter: int) -> nx.MultiDiGraph:
        """
        Return graph state as it existed at a specific chapter.
        
        Args:
            chapter: Chapter number to get state for
            
        Returns:
            New graph containing only nodes/edges that existed by that chapter
        """
        temporal_graph = nx.MultiDiGraph()
        
        # Add nodes that existed by this chapter
        for node, data in self.graph.nodes(data=True):
            if data.get('first_appearance', 1) <= chapter:
                temporal_graph.add_node(node, **data)
        
        # Add edges that existed by this chapter
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if data.get('chapter', 1) <= chapter:
                # Only add if both nodes exist in temporal graph
                if u in temporal_graph and v in temporal_graph:
                    temporal_graph.add_edge(u, v, key=key, **data)
                
        return temporal_graph
    
    def get_characters(self) -> List[Dict]:
        """Get all character entities."""
        characters = []
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'CHARACTER':
                characters.append({
                    "id": node,
                    "name": data.get('name', node),
                    "attributes": data.get('attributes', {}),
                    "first_appearance": data.get('first_appearance', 1)
                })
        return characters
    
    def get_character_relationships(self, character_id: str) -> List[Dict]:
        """Get all relationships for a specific character."""
        if character_id not in self.graph:
            return []
            
        relationships = []
        
        # Outgoing relationships
        for neighbor in self.graph.neighbors(character_id):
            edge_data = self.graph.get_edge_data(character_id, neighbor)
            for key, attrs in edge_data.items():
                relationships.append({
                    "target": neighbor,
                    "target_name": self.graph.nodes[neighbor].get('name', neighbor),
                    "relation": attrs.get('relation'),
                    "strength": attrs.get('strength', 0.5),
                    "chapter": attrs.get('chapter', 1),
                    "direction": "outgoing"
                })
        
        # Incoming relationships
        for predecessor in self.graph.predecessors(character_id):
            edge_data = self.graph.get_edge_data(predecessor, character_id)
            for key, attrs in edge_data.items():
                relationships.append({
                    "target": predecessor,
                    "target_name": self.graph.nodes[predecessor].get('name', predecessor),
                    "relation": attrs.get('relation'),
                    "strength": attrs.get('strength', 0.5),
                    "chapter": attrs.get('chapter', 1),
                    "direction": "incoming"
                })
                
        return relationships
    
    def detect_inconsistencies(self) -> List[Dict]:
        """
        Find plot holes and contradictions in the graph.
        
        Returns:
            List of detected issues with type, entity, and description
        """
        issues = []
        
        # Check for conflicting relationships
        for node in self.graph.nodes():
            outgoing = list(self.graph.out_edges(node, data=True))
            
            # Track relationship types per target
            relations_by_target: Dict[str, List[str]] = {}
            for _, target, data in outgoing:
                rel_type = data.get('relation', '')
                if target not in relations_by_target:
                    relations_by_target[target] = []
                relations_by_target[target].append(rel_type)
            
            # Check for conflicting relationships with same target
            conflicting_pairs = [
                ("ALLIES_WITH", "CONFLICTS_WITH"),
                ("TRUSTS", "DISTRUSTS"),
                ("LOVES", "HATES"),
            ]
            
            for target, rel_types in relations_by_target.items():
                for rel_a, rel_b in conflicting_pairs:
                    if rel_a in rel_types and rel_b in rel_types:
                        node_name = self.graph.nodes[node].get('name', node)
                        target_name = self.graph.nodes[target].get('name', target)
                        issues.append({
                            "type": "relationship_conflict",
                            "entity": node,
                            "target": target,
                            "description": f"{node_name} has conflicting relationships with {target_name}: both {rel_a} and {rel_b}"
                        })
        
        # Check for orphaned events (no causal predecessors after chapter 1)
        event_nodes = [
            n for n, d in self.graph.nodes(data=True) 
            if d.get('type') == 'EVENT'
        ]
        for event in event_nodes:
            event_data = self.graph.nodes[event]
            chapter = event_data.get('first_appearance', 1)
            
            # Only check events after chapter 1
            if chapter > 1:
                predecessors = list(self.graph.predecessors(event))
                causal_preds = []
                for pred in predecessors:
                    edge_data = self.graph.get_edge_data(pred, event)
                    if edge_data:
                        for key, attrs in edge_data.items():
                            if attrs.get('relation') == 'CAUSED_BY':
                                causal_preds.append(pred)
                                break
                                
                if not causal_preds:
                    issues.append({
                        "type": "orphaned_event",
                        "entity": event,
                        "description": f"Event '{event_data.get('name', event)}' (Ch. {chapter}) has no causal explanation"
                    })
        
        return issues
    
    def get_statistics(self) -> Dict:
        """Get graph statistics."""
        node_types = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'UNKNOWN')
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
        relation_types = {}
        for u, v, data in self.graph.edges(data=True):
            rel_type = data.get('relation', 'UNKNOWN')
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
            
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "relation_types": relation_types,
            "chapters_covered": len(self.chapter_timeline)
        }
    
    def save(self, path: str) -> None:
        """Save graph to file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'entity_index': {k: v.to_dict() for k, v in self.entity_index.items()},
                'chapter_timeline': self.chapter_timeline
            }, f)
        logger.info(f"Graph saved to {path}")
        
    def load(self, path: str) -> None:
        """Load graph from file."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.graph = data['graph']
            self.chapter_timeline = data['chapter_timeline']
            # Reconstruct entity index from graph nodes
            self.entity_index = {}
            for node, node_data in self.graph.nodes(data=True):
                self.entity_index[node] = StoryEntity(
                    id=node,
                    type=node_data.get('type', 'UNKNOWN'),
                    name=node_data.get('name', node),
                    attributes=node_data.get('attributes', {}),
                    first_appearance=node_data.get('first_appearance', 1)
                )
        logger.info(f"Graph loaded from {path}")
