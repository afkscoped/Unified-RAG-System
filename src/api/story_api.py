"""
Story Mode API

FastAPI endpoints for story generation functionality.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from loguru import logger


# Create router
story_router = APIRouter()


# Pydantic models for request/response
class StoryGenerateRequest(BaseModel):
    """Request model for story generation."""
    prompt: str = Field(..., description="Story prompt/continuation request")
    chapter: int = Field(default=1, ge=1, description="Chapter number")
    mode: str = Field(
        default="compare", 
        description="Generation mode: 'compare', 'unified', 'graph', 'hybrid'"
    )


class GenerationMetricsResponse(BaseModel):
    """Metrics for a generation result."""
    response_time: float
    consistency_score: float
    coherence_score: float
    retrieval_count: int
    tokens_generated: int
    source_diversity: float


class StoryGenerateResponse(BaseModel):
    """Response model for story generation."""
    text: str
    method: str
    metrics: GenerationMetricsResponse
    sources: Optional[List[Dict]] = None


class CompareResponse(BaseModel):
    """Response model for comparative generation."""
    unified: StoryGenerateResponse
    graph: StoryGenerateResponse
    hybrid: StoryGenerateResponse


class CharacterArcResponse(BaseModel):
    """Character arc summary response."""
    character_id: str
    total_chapters: int
    current_state: Dict
    initial_state: Dict
    arc_trajectory: str
    key_transitions: List[Dict]
    relationship_evolution: Dict


class PlotSuggestionResponse(BaseModel):
    """Plot suggestion response."""
    type: str
    priority: str
    suggestion: str
    details: str


# Global engine reference (will be set by main app)
_comparison_engine = None


def set_comparison_engine(engine):
    """Set the comparison engine instance."""
    global _comparison_engine
    _comparison_engine = engine


def get_comparison_engine():
    """Get the comparison engine. Raises if not configured."""
    if _comparison_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Story engine not initialized. Please configure the system first."
        )
    return _comparison_engine


@story_router.post("/generate", response_model=StoryGenerateResponse)
async def generate_story(request: StoryGenerateRequest):
    """
    Generate a story segment using specified mode.
    
    Modes:
    - unified: Use Unified RAG (vector search)
    - graph: Use Graph RAG (knowledge graph)
    - hybrid: Use combined approach
    - compare: Generate with all three (returns comparison)
    """
    try:
        engine = get_comparison_engine()
        
        if request.mode == "compare":
            results = engine.generate_comparative(request.prompt, request.chapter)
            # Return hybrid result for single response
            return _format_single_response(results["hybrid"])
        elif request.mode in ("unified", "graph", "hybrid"):
            results = engine.generate_comparative(request.prompt, request.chapter)
            return _format_single_response(results[request.mode])
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {request.mode}. Use 'compare', 'unified', 'graph', or 'hybrid'."
            )
            
    except Exception as e:
        logger.error(f"Story generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@story_router.post("/compare", response_model=CompareResponse)
async def compare_generation(request: StoryGenerateRequest):
    """
    Generate story segment using all three approaches for comparison.
    
    Returns results from Unified RAG, Graph RAG, and Hybrid Fusion
    with metrics for each.
    """
    try:
        engine = get_comparison_engine()
        results = engine.generate_comparative(request.prompt, request.chapter)
        
        return {
            "unified": _format_single_response(results["unified"]),
            "graph": _format_single_response(results["graph"]),
            "hybrid": _format_single_response(results["hybrid"])
        }
        
    except Exception as e:
        logger.error(f"Comparison generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@story_router.get("/characters")
async def list_characters():
    """Get all characters in the knowledge graph."""
    try:
        engine = get_comparison_engine()
        graph = engine.get_story_graph()
        characters = graph.get_characters()
        
        return {
            "characters": characters,
            "total": len(characters)
        }
        
    except Exception as e:
        logger.error(f"Failed to get characters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@story_router.get("/arc/{character_id}", response_model=CharacterArcResponse)
async def get_character_arc(character_id: str):
    """Get character arc summary for a specific character."""
    try:
        engine = get_comparison_engine()
        arc_tracker = engine.get_arc_tracker()
        
        summary = arc_tracker.get_character_arc_summary(character_id)
        
        if not summary:
            raise HTTPException(
                status_code=404,
                detail=f"Character '{character_id}' not found or has no arc data."
            )
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get character arc: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@story_router.get("/relationships/{character_id}")
async def get_character_relationships(character_id: str):
    """Get all relationships for a specific character."""
    try:
        engine = get_comparison_engine()
        graph = engine.get_story_graph()
        
        relationships = graph.get_character_relationships(character_id)
        
        return {
            "character_id": character_id,
            "relationships": relationships,
            "total": len(relationships)
        }
        
    except Exception as e:
        logger.error(f"Failed to get relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@story_router.get("/suggestions", response_model=List[PlotSuggestionResponse])
async def get_plot_suggestions(chapter: int = 1):
    """Get AI-generated plot development suggestions."""
    try:
        engine = get_comparison_engine()
        discovery = engine.relationship_discovery
        
        suggestions = discovery.suggest_plot_complications(chapter)
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Failed to get suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@story_router.get("/history")
async def get_story_history():
    """Get story generation history."""
    try:
        engine = get_comparison_engine()
        history = engine.get_story_history()
        
        return {
            "history": history,
            "total_segments": len(history)
        }
        
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@story_router.get("/graph/stats")
async def get_graph_statistics():
    """Get knowledge graph statistics."""
    try:
        engine = get_comparison_engine()
        graph = engine.get_story_graph()
        
        return graph.get_statistics()
        
    except Exception as e:
        logger.error(f"Failed to get graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@story_router.get("/graph/inconsistencies")
async def detect_inconsistencies():
    """Detect plot holes and contradictions in the knowledge graph."""
    try:
        engine = get_comparison_engine()
        graph = engine.get_story_graph()
        
        issues = graph.detect_inconsistencies()
        
        return {
            "issues": issues,
            "total": len(issues),
            "consistency_score": max(0.0, 1.0 - len(issues) * 0.1)
        }
        
    except Exception as e:
        logger.error(f"Failed to detect inconsistencies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@story_router.post("/reset")
async def reset_story():
    """Reset story state (clear graph, history, etc.)."""
    try:
        engine = get_comparison_engine()
        
        # Clear graph - recreate
        engine.story_graph = type(engine.story_graph)(engine.embedding_manager)
        engine.arc_tracker = type(engine.arc_tracker)(engine.story_graph)
        engine.relationship_discovery = type(engine.relationship_discovery)(engine.story_graph)
        engine.story_history = []
        engine.current_chapter = 1
        
        if engine.unified_adapter:
            engine.unified_adapter.clear_context()
        
        return {"message": "Story state reset successfully"}
        
    except Exception as e:
        logger.error(f"Failed to reset story: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _format_single_response(result: Dict) -> Dict:
    """Format a single generation result for response."""
    metrics = result.get("metrics")
    if hasattr(metrics, "to_dict"):
        metrics_dict = metrics.to_dict()
    elif isinstance(metrics, dict):
        metrics_dict = metrics
    else:
        metrics_dict = {
            "response_time": 0.0,
            "consistency_score": 0.5,
            "coherence_score": 0.5,
            "retrieval_count": 0,
            "tokens_generated": 0,
            "source_diversity": 0.0
        }
    
    return {
        "text": result.get("text", ""),
        "method": result.get("method", "Unknown"),
        "metrics": metrics_dict,
        "sources": result.get("sources", [])
    }
