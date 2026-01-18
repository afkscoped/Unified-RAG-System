"""
Tests for Story Knowledge Graph

Unit tests for:
- Entity creation and addition
- Relationship creation
- Multi-hop traversal
- Temporal state snapshots
- Inconsistency detection
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.story.graph_rag.story_graph import (
    StoryKnowledgeGraph,
    StoryEntity,
    StoryRelationship
)


class TestStoryEntity:
    """Tests for StoryEntity dataclass."""
    
    def test_create_entity(self):
        """Test creating a story entity."""
        entity = StoryEntity(
            id="char_elena",
            type="CHARACTER",
            name="Elena",
            attributes={"title": "Princess"},
            first_appearance=1
        )
        
        assert entity.id == "char_elena"
        assert entity.type == "CHARACTER"
        assert entity.name == "Elena"
        assert entity.attributes["title"] == "Princess"
        assert entity.first_appearance == 1
    
    def test_entity_to_dict(self):
        """Test entity serialization."""
        entity = StoryEntity(
            id="loc_castle",
            type="LOCATION",
            name="Castle Blackwood"
        )
        
        data = entity.to_dict()
        
        assert data["id"] == "loc_castle"
        assert data["type"] == "LOCATION"
        assert data["name"] == "Castle Blackwood"
        assert "last_modified" in data


class TestStoryRelationship:
    """Tests for StoryRelationship dataclass."""
    
    def test_create_relationship(self):
        """Test creating a relationship."""
        rel = StoryRelationship(
            source="char_elena",
            target="char_marcus",
            relation_type="ALLIES_WITH",
            strength=0.8,
            temporal_context=2
        )
        
        assert rel.source == "char_elena"
        assert rel.target == "char_marcus"
        assert rel.relation_type == "ALLIES_WITH"
        assert rel.strength == 0.8
        assert rel.temporal_context == 2
    
    def test_relationship_to_dict(self):
        """Test relationship serialization."""
        rel = StoryRelationship(
            source="char_a",
            target="char_b",
            relation_type="CONFLICTS_WITH"
        )
        
        data = rel.to_dict()
        
        assert data["source"] == "char_a"
        assert data["target"] == "char_b"
        assert data["relation_type"] == "CONFLICTS_WITH"


class TestStoryKnowledgeGraph:
    """Tests for StoryKnowledgeGraph."""
    
    @pytest.fixture
    def graph(self):
        """Create a fresh graph for each test."""
        return StoryKnowledgeGraph()
    
    @pytest.fixture
    def populated_graph(self, graph):
        """Create a graph with sample data."""
        # Add characters
        graph.add_entity(StoryEntity(
            id="char_elena",
            type="CHARACTER",
            name="Elena",
            first_appearance=1
        ))
        graph.add_entity(StoryEntity(
            id="char_marcus",
            type="CHARACTER",
            name="Marcus",
            first_appearance=1
        ))
        graph.add_entity(StoryEntity(
            id="char_victor",
            type="CHARACTER",
            name="Victor",
            first_appearance=2
        ))
        
        # Add location
        graph.add_entity(StoryEntity(
            id="loc_castle",
            type="LOCATION",
            name="Castle Blackwood",
            first_appearance=1
        ))
        
        # Add relationships
        graph.add_relationship(StoryRelationship(
            source="char_elena",
            target="char_marcus",
            relation_type="ALLIES_WITH",
            strength=0.8,
            temporal_context=1
        ))
        graph.add_relationship(StoryRelationship(
            source="char_marcus",
            target="char_victor",
            relation_type="CONFLICTS_WITH",
            strength=0.7,
            temporal_context=2
        ))
        
        return graph
    
    def test_add_entity(self, graph):
        """Test adding an entity to the graph."""
        entity = StoryEntity(
            id="char_test",
            type="CHARACTER",
            name="Test Character"
        )
        
        graph.add_entity(entity)
        
        assert "char_test" in graph.graph
        assert graph.entity_index["char_test"] == entity
    
    def test_add_relationship(self, graph):
        """Test adding a relationship."""
        # Add entities first
        graph.add_entity(StoryEntity(id="a", type="CHARACTER", name="A"))
        graph.add_entity(StoryEntity(id="b", type="CHARACTER", name="B"))
        
        rel = StoryRelationship(
            source="a",
            target="b",
            relation_type="ALLIES_WITH"
        )
        graph.add_relationship(rel)
        
        assert graph.graph.has_edge("a", "b")
    
    def test_get_entity(self, populated_graph):
        """Test retrieving an entity."""
        entity = populated_graph.get_entity("char_elena")
        
        assert entity is not None
        assert entity.name == "Elena"
    
    def test_get_entity_context(self, populated_graph):
        """Test multi-hop context retrieval."""
        context = populated_graph.get_entity_context("char_elena", max_hops=2)
        
        assert context is not None
        assert "entity" in context
        assert "relationships" in context
        assert context["total_connections"] > 0
    
    def test_query_by_relation_path(self, populated_graph):
        """Test path-based query."""
        # Elena -ALLIES_WITH-> Marcus -CONFLICTS_WITH-> Victor
        results = populated_graph.query_by_relation_path(
            "char_elena",
            ["ALLIES_WITH", "CONFLICTS_WITH"]
        )
        
        assert len(results) > 0
        assert any(r["node"] == "char_victor" for r in results)
    
    def test_get_temporal_state(self, populated_graph):
        """Test temporal state snapshot."""
        # Get state at chapter 1 (before Victor)
        temporal = populated_graph.get_temporal_state(chapter=1)
        
        assert "char_elena" in temporal
        assert "char_marcus" in temporal
        # Victor appeared in chapter 2, should not be in chapter 1 state
        assert "char_victor" not in temporal
    
    def test_get_characters(self, populated_graph):
        """Test getting all characters."""
        characters = populated_graph.get_characters()
        
        assert len(characters) == 3
        names = [c["name"] for c in characters]
        assert "Elena" in names
        assert "Marcus" in names
        assert "Victor" in names
    
    def test_detect_inconsistencies_conflict(self, graph):
        """Test detection of conflicting relationships."""
        # Setup conflicting relationships
        graph.add_entity(StoryEntity(id="a", type="CHARACTER", name="A"))
        graph.add_entity(StoryEntity(id="b", type="CHARACTER", name="B"))
        
        graph.add_relationship(StoryRelationship(
            source="a", target="b", relation_type="ALLIES_WITH"
        ))
        graph.add_relationship(StoryRelationship(
            source="a", target="b", relation_type="CONFLICTS_WITH"
        ))
        
        issues = graph.detect_inconsistencies()
        
        assert len(issues) > 0
        assert any(i["type"] == "relationship_conflict" for i in issues)
    
    def test_get_statistics(self, populated_graph):
        """Test graph statistics."""
        stats = populated_graph.get_statistics()
        
        assert stats["total_nodes"] == 4
        assert stats["total_edges"] == 2
        assert "CHARACTER" in stats["node_types"]
        assert stats["node_types"]["CHARACTER"] == 3
