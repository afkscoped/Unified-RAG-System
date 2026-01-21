
import sys
import os
from loguru import logger

# Add src to path
sys.path.append(r"c:\Users\raddo\Documents\RAG project\unified-rag-system")

from src.story.graph_rag.entity_extractor import NarrativeEntityExtractor

def reproduce_extraction_failure():
    extractor = NarrativeEntityExtractor()
    
    text = "The assassination of the King caused the Civil War. The war led to the burning of the Capital."
    
    print(f"Analyzing text: {text}")
    
    doc = extractor.nlp(text)
    for token in doc:
        print(f"{token.text:<15} {token.pos_:<6} {token.dep_:<10} {token.head.text:<15} {token.lemma_:<15}")

    entities, relationships = extractor.extract_entities(text)
    
    print(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
    
    causal_rels = [r for r in relationships if r['relation_type'] == 'CAUSED_BY']
    print(f"Found {len(causal_rels)} CAUSES/CAUSED_BY relationships")
    
    for r in causal_rels:
        print(f"  {r['source']} -> {r['target']}")

    # Check events
    events = extractor.extract_events(text)
    print(f"Found {len(events)} events")
    for e in events:
        print(f"  {e['name']}: {e['description']}")

if __name__ == "__main__":
    reproduce_extraction_failure()
