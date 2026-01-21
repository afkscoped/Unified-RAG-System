
import sys
import os
from loguru import logger

# Add src to path
sys.path.append(r"c:\Users\raddo\Documents\RAG project\unified-rag-system")

from src.story.graph_rag.story_graph import StoryKnowledgeGraph, StoryEntity, StoryRelationship
from src.story.analysis.consistency_checker import ConsistencyChecker, ConsistencyReport, ViolationType

def reproduce_plot_holes():
    # 1. Setup Graph with established facts
    graph = StoryKnowledgeGraph()
    checker = ConsistencyChecker(story_graph=graph)
    
    # Established Fact: Marcus hates the King
    graph.add_entity(StoryEntity(id="char_marcus", type="CHARACTER", name="Marcus"))
    graph.add_entity(StoryEntity(id="char_king", type="CHARACTER", name="The King"))
    graph.add_relationship(StoryRelationship(
        source="char_marcus", target="char_king", 
        relation_type="CONFLICTS_WITH", strength=0.9
    ))
    
    # Established Fact: The Castle is in the Capital
    graph.add_entity(StoryEntity(id="loc_castle", type="LOCATION", name="The Castle"))
    graph.add_entity(StoryEntity(id="loc_capital", type="LOCATION", name="The Capital"))
    graph.add_relationship(StoryRelationship(
        source="loc_castle", target="loc_capital", 
        relation_type="LOCATED_IN"
    ))
    
    # 2. Test Content with Inconsistencies
    
    # Case A: Relationship Violation
    # Marcus acts friendly towards the King
    text_a = "Marcus bowed deeply and smiled at the King. 'My old friend,' he said warmly, 'I am here to help you.'"
    print(f"\nAnalyzing Text A (Relationship Violation): \"{text_a}\"")
    
    report_a = checker.check_graph_rag(
        text=text_a, 
        context="", 
        entities=[{"name": "Marcus"}, {"name": "The King"}], 
        previous_segments=[],
        chapter=2
    )
    
    print(f"Violations found: {len(report_a.violations)}")
    for v in report_a.violations:
        print(f" - [{v.type}] {v.description}")
        
    print(f"Prevented issues: {len(report_a.prevented_by_graph)}")
    for p in report_a.prevented_by_graph:
        print(f" - {p}")

    # Case B: Location Impossibility (Not currently well implemented in existing code)
    # Marcus is in the Castle but also in the Forest
    # The current checker only checks if a character is in two places in the *text* or graph vs text?
    # Let's see what it does.
    text_b = "Marcus stood on the battlements of the Castle. At the same moment, he was walking through the dark Forest."
    print(f"\nAnalyzing Text B (Location Impossibility): \"{text_b}\"")
    
    report_b = checker.check_graph_rag(
        text=text_b,
        context="",
        entities=[{"name": "Marcus"}, {"name": "The Castle"}, {"name": "The Forest"}],
        previous_segments=[],
        chapter=2
    )
    
    print(f"Violations found: {len(report_b.violations)}")
    for v in report_b.violations:
        print(f" - [{v.type}] {v.description}")

    # Case C: Temporal/Causal (Effect before Cause)
    # This requires the timeline analysis which is likely missing
    text_c = "The King died from the poison. Later that day, Marcus poured the poison into the King's cup."
    print(f"\nAnalyzing Text C (Causal Violation): \"{text_c}\"")
    
    report_c = checker.check_graph_rag(
        text=text_c,
        context="",
        entities=[],
        previous_segments=[],
        chapter=2
    )
    
    print(f"Violations found: {len(report_c.violations)}")
    for v in report_c.violations:
        print(f" - [{v.type}] {v.description}")

if __name__ == "__main__":
    reproduce_plot_holes()
