"""
Test script for Multi-Dimensional Coherence Analyzer

Verifies all 5 coherence dimensions:
1. Semantic - Context similarity
2. Lexical - N-gram overlap
3. Discourse - Entity continuity
4. Temporal - Time marker consistency
5. Voice - Narrative voice consistency (POV, sentence length, vocabulary)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.story.analysis.coherence_analyzer import CoherenceAnalyzer, CoherenceBreakdown


def test_coherence_analyzer_initialization():
    """Test that CoherenceAnalyzer initializes with correct weights."""
    print("\n" + "="*60)
    print("TEST 1: Coherence Analyzer Initialization")
    print("="*60)
    
    analyzer = CoherenceAnalyzer()
    
    # Check weights include all 5 dimensions
    expected_weights = {"semantic", "lexical", "discourse", "temporal", "voice"}
    actual_weights = set(analyzer.weights.keys())
    
    assert actual_weights == expected_weights, f"Missing weights: {expected_weights - actual_weights}"
    assert abs(sum(analyzer.weights.values()) - 1.0) < 0.01, "Weights should sum to 1.0"
    
    print(f"[OK] All 5 dimensions present: {list(analyzer.weights.keys())}")
    print(f"[OK] Weights sum to: {sum(analyzer.weights.values()):.2f}")
    print("PASSED\n")


def test_coherence_breakdown_dataclass():
    """Test CoherenceBreakdown dataclass has all dimensions."""
    print("\n" + "="*60)
    print("TEST 2: CoherenceBreakdown Dataclass")
    print("="*60)
    
    breakdown = CoherenceBreakdown(
        semantic=0.8,
        lexical=0.7,
        discourse=0.75,
        temporal=0.9,
        voice=0.85,
        composite=0.8
    )
    
    result = breakdown.to_dict()
    
    assert "semantic" in result, "Missing semantic"
    assert "lexical" in result, "Missing lexical"
    assert "discourse" in result, "Missing discourse"
    assert "temporal" in result, "Missing temporal"
    assert "voice" in result, "Missing voice"
    assert "composite" in result, "Missing composite"
    
    print(f"[OK] All fields present in to_dict(): {list(result.keys())}")
    print(f"[OK] Values: semantic={result['semantic']}, voice={result['voice']}")
    print("PASSED\n")


def test_semantic_coherence():
    """Test semantic coherence calculation."""
    print("\n" + "="*60)
    print("TEST 3: Semantic Coherence")
    print("="*60)
    
    analyzer = CoherenceAnalyzer()
    
    context = "The brave knight rode through the dark forest seeking the dragon."
    generated_similar = "The courageous warrior traveled through shadowy woods hunting the beast."
    generated_different = "The stock market crashed yesterday due to economic concerns."
    
    # Similar text should have higher semantic coherence
    score_similar = analyzer.calculate_semantic_coherence(generated_similar, context)
    score_different = analyzer.calculate_semantic_coherence(generated_different, context)
    
    print(f"  Similar text score: {score_similar:.3f}")
    print(f"  Different text score: {score_different:.3f}")
    
    # Similar should score higher (if embeddings work) or both be fallback
    assert 0 <= score_similar <= 1, "Score should be 0-1"
    assert 0 <= score_different <= 1, "Score should be 0-1"
    
    print("[OK] Semantic coherence returns valid scores")
    print("PASSED\n")


def test_lexical_coherence():
    """Test lexical coherence (n-gram overlap)."""
    print("\n" + "="*60)
    print("TEST 4: Lexical Coherence")
    print("="*60)
    
    analyzer = CoherenceAnalyzer()
    
    context = "The knight drew his sword and charged at the dragon."
    generated_overlap = "The knight raised his sword high as he faced the dragon."
    generated_no_overlap = "Completely unrelated text about something else entirely."
    
    score_overlap = analyzer.calculate_lexical_coherence(generated_overlap, context, [])
    score_no_overlap = analyzer.calculate_lexical_coherence(generated_no_overlap, context, [])
    
    print(f"  Overlapping text score: {score_overlap:.3f}")
    print(f"  Non-overlapping text score: {score_no_overlap:.3f}")
    
    assert score_overlap > score_no_overlap, "Overlapping text should score higher"
    print("[OK] Lexical coherence correctly identifies overlap")
    print("PASSED\n")


def test_discourse_coherence():
    """Test discourse coherence (entity continuity)."""
    print("\n" + "="*60)
    print("TEST 5: Discourse Coherence")
    print("="*60)
    
    analyzer = CoherenceAnalyzer()
    
    previous = ["Marcus the knight entered the castle.", "He met Princess Elena there."]
    generated_continuous = "Marcus smiled at Elena as they walked through the halls."
    generated_new_entities = "John and Mary went to the store to buy groceries."
    
    score_continuous = analyzer.calculate_discourse_coherence(generated_continuous, "", previous)
    score_new = analyzer.calculate_discourse_coherence(generated_new_entities, "", previous)
    
    print(f"  Continuous entities score: {score_continuous:.3f}")
    print(f"  New entities score: {score_new:.3f}")
    
    assert score_continuous >= score_new, "Continuing entities should score higher or equal"
    print("[OK] Discourse coherence tracks entity continuity")
    print("PASSED\n")


def test_temporal_coherence():
    """Test temporal coherence (time marker consistency)."""
    print("\n" + "="*60)
    print("TEST 6: Temporal Coherence")
    print("="*60)
    
    analyzer = CoherenceAnalyzer()
    
    previous = ["Yesterday, Marcus arrived at the village.", "That morning, he met the elder."]
    generated_consistent = "Later that day, he set off on his journey."
    generated_inconsistent = "Tomorrow will be a new day for exploration."
    
    # Method signature: calculate_temporal_coherence(generated_text, chapter, previous_segments)
    score_consistent = analyzer.calculate_temporal_coherence(generated_consistent, 1, previous)
    score_inconsistent = analyzer.calculate_temporal_coherence(generated_inconsistent, 1, previous)
    
    print(f"  Temporally consistent score: {score_consistent:.3f}")
    print(f"  Temporally inconsistent score: {score_inconsistent:.3f}")
    
    assert 0 <= score_consistent <= 1, "Score should be 0-1"
    assert 0 <= score_inconsistent <= 1, "Score should be 0-1"
    print("[OK] Temporal coherence returns valid scores")
    print("PASSED\n")


def test_voice_coherence():
    """Test voice coherence (POV, sentence length, vocabulary)."""
    print("\n" + "="*60)
    print("TEST 7: Voice Coherence (NEW)")
    print("="*60)
    
    analyzer = CoherenceAnalyzer()
    
    # First person narrative
    previous_first = [
        "I walked through the forest alone. My heart was pounding.",
        "I could hear the wolves howling in the distance. I gripped my sword tighter."
    ]
    generated_first = "I continued forward, my resolve strengthening with each step."
    generated_third = "He walked through the forest. His heart was pounding as he moved."
    
    score_first_match = analyzer.calculate_voice_coherence(generated_first, "", previous_first)
    score_third_mismatch = analyzer.calculate_voice_coherence(generated_third, "", previous_first)
    
    print(f"  First person match score: {score_first_match:.3f}")
    print(f"  Third person mismatch score: {score_third_mismatch:.3f}")
    
    assert score_first_match > score_third_mismatch, "Matching POV should score higher"
    print("[OK] Voice coherence detects POV consistency")
    
    # Test with no previous segments (should return 1.0)
    score_no_ref = analyzer.calculate_voice_coherence("Some text here.", "", [])
    assert score_no_ref == 1.0, "No reference should return 1.0"
    print("[OK] Voice coherence handles empty reference correctly")
    print("PASSED\n")


def test_full_analyze():
    """Test the full analyze method returns all dimensions."""
    print("\n" + "="*60)
    print("TEST 8: Full Analysis Integration")
    print("="*60)
    
    analyzer = CoherenceAnalyzer()
    
    context = "The ancient kingdom was under threat from dark forces."
    generated = "The kingdom's defenders rallied to protect their homeland from evil."
    previous = ["War had come to the peaceful lands.", "Heroes emerged from unexpected places."]
    
    result = analyzer.analyze(generated, context, previous)
    
    assert isinstance(result, CoherenceBreakdown), "Should return CoherenceBreakdown"
    
    print(f"  Semantic: {result.semantic:.3f}")
    print(f"  Lexical: {result.lexical:.3f}")
    print(f"  Discourse: {result.discourse:.3f}")
    print(f"  Temporal: {result.temporal:.3f}")
    print(f"  Voice: {result.voice:.3f}")
    print(f"  Composite: {result.composite:.3f}")
    
    # Check composite is weighted average
    expected_composite = (
        analyzer.weights["semantic"] * result.semantic +
        analyzer.weights["lexical"] * result.lexical +
        analyzer.weights["discourse"] * result.discourse +
        analyzer.weights["temporal"] * result.temporal +
        analyzer.weights["voice"] * result.voice
    )
    
    assert abs(result.composite - expected_composite) < 0.01, "Composite should match weighted average"
    print("[OK] Composite score matches weighted calculation")
    print("PASSED\n")


def run_all_tests():
    """Run all coherence tests."""
    print("\n" + "#"*60)
    print("# MULTI-DIMENSIONAL COHERENCE ANALYZER TESTS")
    print("#"*60)
    
    tests = [
        test_coherence_analyzer_initialization,
        test_coherence_breakdown_dataclass,
        test_semantic_coherence,
        test_lexical_coherence,
        test_discourse_coherence,
        test_temporal_coherence,
        test_voice_coherence,
        test_full_analyze
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] FAILED: {test.__name__}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] ERROR in {test.__name__}: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
