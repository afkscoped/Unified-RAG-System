"""
Tests for Story Comparison Engine

Integration tests for:
- Comparative generation
- Metrics calculation
- Character state updates
- Story history tracking
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.story.comparison_engine import StoryComparisonEngine, GenerationMetrics


class TestGenerationMetrics:
    """Tests for GenerationMetrics dataclass."""
    
    def test_create_metrics(self):
        """Test creating metrics."""
        metrics = GenerationMetrics(
            response_time=1.5,
            consistency_score=0.9,
            coherence_score=0.8,
            retrieval_count=5,
            tokens_generated=100,
            source_diversity=0.7
        )
        
        assert metrics.response_time == 1.5
        assert metrics.consistency_score == 0.9
        assert metrics.coherence_score == 0.8
        assert metrics.retrieval_count == 5
        assert metrics.tokens_generated == 100
        assert metrics.source_diversity == 0.7
    
    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = GenerationMetrics()
        data = metrics.to_dict()
        
        assert isinstance(data, dict)
        assert "response_time" in data
        assert "consistency_score" in data


class TestStoryComparisonEngine:
    """Tests for StoryComparisonEngine."""
    
    @pytest.fixture
    def mock_llm_router(self):
        """Create mock LLM router."""
        router = Mock()
        router.generate = Mock(return_value="Generated story text here.")
        return router
    
    @pytest.fixture
    def engine(self, mock_llm_router):
        """Create comparison engine with mocked dependencies."""
        return StoryComparisonEngine(
            unified_rag_system=None,
            llm_router=mock_llm_router,
            embedding_manager=None
        )
    
    def test_initialization(self, engine):
        """Test engine initializes correctly."""
        assert engine.story_graph is not None
        assert engine.entity_extractor is not None
        assert engine.arc_tracker is not None
        assert engine.relationship_discovery is not None
        assert engine.current_chapter == 1
        assert len(engine.story_history) == 0
    
    def test_generate_comparative_returns_all_modes(self, engine, mock_llm_router):
        """Test generate_comparative returns results for all three modes."""
        results = engine.generate_comparative(
            prompt="Elena discovers a hidden chamber",
            chapter=1
        )
        
        assert "unified" in results
        assert "graph" in results
        assert "hybrid" in results
        
        # Each should have text and metrics
        for mode in ["unified", "graph", "hybrid"]:
            assert "text" in results[mode]
            assert "metrics" in results[mode]
            assert "method" in results[mode]
    
    def test_story_history_updated(self, engine, mock_llm_router):
        """Test that story history is updated after generation."""
        initial_history_len = len(engine.story_history)
        
        engine.generate_comparative("Test prompt", chapter=1)
        
        assert len(engine.story_history) == initial_history_len + 1
        assert engine.story_history[-1]["prompt"] == "Test prompt"
        assert engine.story_history[-1]["chapter"] == 1
    
    def test_current_chapter_updates(self, engine, mock_llm_router):
        """Test current chapter tracks latest generation."""
        engine.generate_comparative("Chapter 1 content", chapter=1)
        assert engine.current_chapter == 1
        
        engine.generate_comparative("Chapter 3 content", chapter=3)
        assert engine.current_chapter == 3
    
    def test_get_statistics(self, engine):
        """Test statistics retrieval."""
        stats = engine.get_statistics()
        
        assert "graph_stats" in stats
        assert "total_segments" in stats
        assert "current_chapter" in stats
        assert "characters_tracked" in stats
    
    def test_get_story_graph(self, engine):
        """Test graph accessor."""
        graph = engine.get_story_graph()
        assert graph is not None
    
    def test_get_arc_tracker(self, engine):
        """Test arc tracker accessor."""
        tracker = engine.get_arc_tracker()
        assert tracker is not None
    
    def test_get_story_history(self, engine):
        """Test history accessor."""
        history = engine.get_story_history()
        assert isinstance(history, list)
    
    def test_consistency_score_calculation(self, engine):
        """Test consistency score based on graph issues."""
        # No issues = full score
        score = engine._check_consistency("Clean text")
        assert score == 1.0
        
        # Add conflicting relationships to trigger issues
        from src.story.graph_rag.story_graph import StoryEntity, StoryRelationship
        
        engine.story_graph.add_entity(StoryEntity(id="a", type="CHARACTER", name="A"))
        engine.story_graph.add_entity(StoryEntity(id="b", type="CHARACTER", name="B"))
        engine.story_graph.add_relationship(StoryRelationship(
            source="a", target="b", relation_type="ALLIES_WITH"
        ))
        engine.story_graph.add_relationship(StoryRelationship(
            source="a", target="b", relation_type="CONFLICTS_WITH"
        ))
        
        # Now should have lower score due to conflict
        score = engine._check_consistency("Text with conflict in graph")
        assert score < 1.0
    
    def test_diversity_calculation(self, engine):
        """Test source diversity calculation."""
        # Empty sources
        assert engine._calc_diversity([]) == 0.0
        
        # All unique
        sources = [
            {"content": "Source 1"},
            {"content": "Source 2"},
            {"content": "Source 3"}
        ]
        assert engine._calc_diversity(sources) == 1.0
        
        # Duplicates
        sources_with_dups = [
            {"content": "Same content here"},
            {"content": "Same content here"},
            {"content": "Different"}
        ]
        diversity = engine._calc_diversity(sources_with_dups)
        assert 0 < diversity < 1.0


class TestCharacterStateExtraction:
    """Tests for character state extraction from generated text."""
    
    @pytest.fixture
    def engine(self):
        """Create engine for testing."""
        mock_router = Mock()
        mock_router.generate = Mock(return_value="Story text")
        return StoryComparisonEngine(llm_router=mock_router)
    
    def test_infer_emotion(self, engine):
        """Test emotion inference from text."""
        hopeful_text = "Elena smiled with hope as she entered the room."
        assert engine._infer_emotion(hopeful_text, "Elena") == "hopeful"
        
        angry_text = "Marcus shouted furiously at the intruders."
        assert engine._infer_emotion(angry_text, "Marcus") == "angry"
        
        neutral_text = "The character walked into the room."
        assert engine._infer_emotion(neutral_text, "Unknown") == "neutral"
    
    def test_extract_goals(self, engine):
        """Test goal extraction from text."""
        text = "Elena wanted to find the hidden treasure. She also needed to escape the castle."
        goals = engine._extract_goals(text, "Elena")
        
        assert len(goals) > 0
        assert any("find" in g for g in goals)
    
    def test_infer_arc_phase(self, engine):
        """Test arc phase inference by chapter."""
        assert engine._infer_arc_phase(1) == "setup"
        assert engine._infer_arc_phase(5) == "rising_action"
        assert engine._infer_arc_phase(10) == "crisis"
        assert engine._infer_arc_phase(14) == "climax"
        assert engine._infer_arc_phase(20) == "resolution"
