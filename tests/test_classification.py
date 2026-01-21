
import sys
import os
sys.path.append(r"c:\Users\raddo\Documents\RAG project\unified-rag-system")

from src.story.graph_rag.entity_extractor import NarrativeEntityExtractor

def test_classification():
    extractor = NarrativeEntityExtractor()
    
    verbs = ["cause", "caused", "lead", "led", "result", "resulted"]
    
    print("Checking verb classification:")
    for v in verbs:
        rel = extractor._classify_relation(v)
        print(f"  '{v}' -> {rel}")
        
    print("\nChecking causal verbs set:")
    print(extractor.causal_verbs)

if __name__ == "__main__":
    test_classification()
