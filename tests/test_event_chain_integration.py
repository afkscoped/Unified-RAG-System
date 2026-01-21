
import sys
import os
import networkx as nx
from loguru import logger

# Add src to path
sys.path.append(r"c:\Users\raddo\Documents\RAG project\unified-rag-system")

from src.story.graph_rag.story_graph import StoryKnowledgeGraph, StoryEntity, StoryRelationship
from src.story.graph_rag.entity_extractor import NarrativeEntityExtractor
from src.story.graph_rag.relationship_discovery import MultiHopDiscoveryEngine

def test_full_chain_pipeline():
    # 1. Initialize components
    extractor = NarrativeEntityExtractor()
    graph = StoryKnowledgeGraph()
    discovery = MultiHopDiscoveryEngine(graph)
    
    # 2. Process text
    text = "The assassination of the King caused the Civil War. The war led to the burning of the Capital."
    print(f"Processing text: {text}")
    
    entities, relationships = extractor.extract_entities(text)
    events = extractor.extract_events(text)
    
    print(f"Extracted {len(entities)} entities, {len(events)} events, {len(relationships)} relationships")
    
    # 3. Populate Graph
    # Add entities
    for entity in entities:
        story_entity = StoryEntity(
            id=entity['id'],
            type=entity['type'],
            name=entity['name'],
            first_appearance=entity['chapter']
        )
        graph.add_entity(story_entity)
        
    # Add events (ensure they are in the graph)
    for event in events:
        story_event = StoryEntity(
            id=event['id'],
            type='EVENT',
            name=event['name'],
            attributes={'description': event['description'], 'action': event['action']},
            first_appearance=event['chapter']
        )
        graph.add_entity(story_event)
        
    # Add relationships
    for rel in relationships:
        story_rel = StoryRelationship(
            source=rel['source_id'],
            target=rel['target_id'],
            relation_type=rel['relation_type'],
            strength=rel['strength'],
            temporal_context=rel['chapter']
        )
        graph.add_relationship(story_rel)
        
    # 4. Run Discovery
    # We expect: assassination -> Civil War -> burning of Capital
    # Find the 'burning' event id
    burning_events = [e for e in events if 'burning' in e['name'].lower()]
    if not burning_events:
        print("Error: Could not find 'burning' event")
        return
        
    target_event_id = burning_events[0]['id']
    print(f"Tracing causes for: {target_event_id} ({burning_events[0]['name']})")
    
    chains = discovery.discover_causal_chains(target_event_id)
    
    print(f"Found {len(chains)} causal chains")
    for chain in chains:
        print(f"Chain: {chain['narrative']}")
        print(f"Sequence: {chain['causal_sequence']}")

if __name__ == "__main__":
    test_full_chain_pipeline()
