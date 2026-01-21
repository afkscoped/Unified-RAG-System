
import sys
import os
from loguru import logger

# Add src to path
sys.path.append(r"c:\Users\raddo\Documents\RAG project\unified-rag-system")

from src.story.graph_rag.story_graph import StoryKnowledgeGraph, StoryEntity, StoryRelationship
from src.story.graph_rag.causal_engine import CausalInferenceEngine

def test_causal_inference():
    # 1. Setup Graph & Engine
    graph = StoryKnowledgeGraph()
    engine = CausalInferenceEngine(graph)
    
    # 2. Define Events
    # Event A: Setup
    # Event B: Explicit reference to A (Linguistic)
    # Event C: Shared actor with B (Common Actor)
    # Event D: Same location as C (Location)
    
    events = [
        {
            "id": "event_a",
            "name": "The King's Speech",
            "description": "The King gives a speech about unity.",
            "chapter": 1,
            "action": "speech"
        },
        {
            "id": "event_b",
            "name": "The Riot",
            "description": "A riot starts because of the King's speech.",
            "chapter": 1,
            "action": "riot"
        },
        {
            "id": "event_c",
            "name": "Marcus is arrested",
            "description": "Marcus is taken into custody by the guards.",
            "chapter": 2,
            "action": "arrested"
        },
        {
            "id": "event_d",
            "name": "Prison Break",
            "description": "Explosion at the prison wall.",
            "chapter": 2,
            "action": "break"
        }
    ]
    
    # Add events to graph
    for e in events:
        graph.add_entity(StoryEntity(
            id=e['id'], type='EVENT', name=e['name'], 
            attributes={'description': e['description']},
            first_appearance=e['chapter']
        ))
        
    # Add helper entities for context
    # Marcus involved in Event B (implied) and Event C
    # Let's add explicit relationships to graph for Common Actor inference
    graph.add_entity(StoryEntity(id="char_marcus", type="CHARACTER", name="Marcus"))
    
    # Marcus in Event B (The Riot) -> maybe Marcus led it?
    # Usually events don't have direct edges to characters in this simple test unless we add them.
    # The CausalInferenceEngine._get_common_actors looks for neighbors.
    # Let's add: Marcus -> PARTICIPATED_IN -> Event B
    graph.add_relationship(StoryRelationship(source="char_marcus", target="event_b", relation_type="PARTICIPATED_IN"))
    graph.add_relationship(StoryRelationship(source="char_marcus", target="event_c", relation_type="PARTICIPATED_IN"))
    
    # Add Location context
    # Event C at Prison, Event D at Prison
    graph.add_entity(StoryEntity(id="loc_prison", type="LOCATION", name="City Prison"))
    graph.add_relationship(StoryRelationship(source="event_c", target="loc_prison", relation_type="LOCATED_IN"))
    graph.add_relationship(StoryRelationship(source="event_d", target="loc_prison", relation_type="LOCATED_IN"))
    
    # 3. Run Inference
    print("Running Causal Inference...")
    relationships = engine.infer_causality(events)
    
    print(f"Inferred {len(relationships)} relationships:")
    for rel in relationships:
        print(f"  {rel.source} -> {rel.target} ({rel.relation_type})")
        print(f"    Confidence: {rel.strength}")
        print(f"    Reasoning: {rel.metadata.get('reasoning')}")

if __name__ == "__main__":
    test_causal_inference()
