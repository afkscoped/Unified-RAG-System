
import sys
import os
import networkx as nx
from loguru import logger

# Add src to path
sys.path.append(r"c:\Users\raddo\Documents\RAG project\unified-rag-system")

from src.story.graph_rag.story_graph import StoryKnowledgeGraph, StoryEntity, StoryRelationship
from src.story.graph_rag.relationship_discovery import MultiHopDiscoveryEngine

def reproduce_event_chain_bug():
    # 1. Setup Graph
    graph = StoryKnowledgeGraph()
    
    # 2. Add Events
    events = [
        ("event_1", "The King dies"),
        ("event_2", "Civil War begins"),
        ("event_3", "The Capital burns"),
    ]
    
    for eid, name in events:
        entity = StoryEntity(id=eid, type="EVENT", name=name, first_appearance=1)
        graph.add_entity(entity)
        
    # Event 1 -> Event 2 (1 causes 2)
    graph.add_relationship(StoryRelationship(
        source="event_1", target="event_2", relation_type="CAUSED_BY", strength=1.0
    ))
    
    # Event 2 -> Event 3 (2 causes 3)
    graph.add_relationship(StoryRelationship(
        source="event_2", target="event_3", relation_type="CAUSED_BY", strength=1.0
    ))
    
    # 4. Run Discovery
    engine = MultiHopDiscoveryEngine(graph)
    # We want to find what caused event_3. Should trace back to event_2 and event_1.
    chains = engine.discover_causal_chains("event_3")
    
    print(f"Found {len(chains)} chains for event_3")
    for chain in chains:
        print(chain)

if __name__ == "__main__":
    reproduce_event_chain_bug()
