"""
Story Comparison Engine

Runs Unified RAG, Graph RAG, and Hybrid approaches side-by-side
for direct comparison with quantitative metrics.

Integrates:
- CoherenceAnalyzer for multi-dimensional coherence scoring
- ConsistencyChecker for violation detection
- PlotSuggestionEngine for actionable plot suggestions
- GraphVisualizer for interactive graph visualization
- PlotTimelineVisualizer for timeline visualization
- FeedbackManager for adaptive learning
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from loguru import logger

from src.story.graph_rag.story_graph import StoryKnowledgeGraph, StoryEntity, StoryRelationship
from src.story.graph_rag.entity_extractor import NarrativeEntityExtractor
from src.story.graph_rag.arc_tracker import DynamicArcTracker, CharacterState
from src.story.graph_rag.relationship_discovery import MultiHopDiscoveryEngine
from src.story.unified_rag.story_adapter import StoryUnifiedRAG
from src.story.fusion.hybrid_generator import HybridFusionEngine

# New analysis and visualization components
from src.story.analysis.coherence_analyzer import CoherenceAnalyzer
from src.story.analysis.consistency_checker import ConsistencyChecker
from src.story.analysis.plot_suggestion_engine import PlotSuggestionEngine
from src.story.visualization.graph_visualizer import GraphVisualizer
from src.story.visualization.plot_timeline import PlotPointExtractor, PlotTimelineVisualizer
from src.story.feedback.feedback_manager import StoryFeedbackManager


@dataclass
class GenerationMetrics:
    """Metrics for comparing generation approaches."""
    response_time: float = 0.0
    consistency_score: float = 0.0  # 0-1, based on contradiction detection
    coherence_score: float = 0.0    # 0-1, based on embedding similarity
    retrieval_count: int = 0
    tokens_generated: int = 0
    source_diversity: float = 0.0   # Unique sources / total sources
    
    # Extended metrics
    coherence_breakdown: Optional[Dict] = None
    consistency_violations: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "response_time": self.response_time,
            "consistency_score": self.consistency_score,
            "coherence_score": self.coherence_score,
            "retrieval_count": self.retrieval_count,
            "tokens_generated": self.tokens_generated,
            "source_diversity": self.source_diversity,
            "coherence_breakdown": self.coherence_breakdown,
            "consistency_violations": self.consistency_violations
        }


class StoryComparisonEngine:
    """
    Runs all three RAG approaches and compares results.
    
    Provides side-by-side generation with metrics for:
    - Unified RAG (vector-based hybrid search)
    - Graph RAG (relationship-aware knowledge graph)
    - Hybrid Fusion (combined approach)
    
    Includes:
    - Multi-dimensional coherence analysis
    - Detailed consistency checking
    - Plot suggestion generation
    - Graph visualization
    - Timeline visualization
    - Feedback-driven learning
    """
    
    def __init__(self, unified_rag_system=None, llm_router=None, embedding_manager=None):
        """
        Initialize comparison engine.
        
        Args:
            unified_rag_system: Optional existing UnifiedRAGSystem
            llm_router: LLMRouter for generation
            embedding_manager: EmbeddingManager for coherence calculation
        """
        # Core components
        self.unified_rag = unified_rag_system
        self.llm_router = llm_router
        self.embedding_manager = embedding_manager
        
        # Graph RAG components
        self.story_graph = StoryKnowledgeGraph(embedding_manager)
        self.entity_extractor = NarrativeEntityExtractor()
        self.arc_tracker = DynamicArcTracker(self.story_graph)
        self.relationship_discovery = MultiHopDiscoveryEngine(self.story_graph)
        
        # Adapters
        self.unified_adapter = None
        if unified_rag_system:
            self.unified_adapter = StoryUnifiedRAG(unified_rag_system)
        
        self.hybrid_generator = None
        if unified_rag_system and llm_router:
            self.hybrid_generator = HybridFusionEngine(
                self.unified_adapter,
                self.story_graph,
                self.arc_tracker,
                llm_router
            )
        
        # NEW: Analysis and visualization components
        self.coherence_analyzer = CoherenceAnalyzer(embedding_manager)
        self.consistency_checker = ConsistencyChecker(self.story_graph, self.arc_tracker)
        self.plot_suggestion_engine = PlotSuggestionEngine(self.story_graph, self.arc_tracker)
        self.graph_visualizer = GraphVisualizer(self.story_graph)
        self.plot_extractor = PlotPointExtractor()
        self.timeline_visualizer = PlotTimelineVisualizer(self.plot_extractor)
        self.feedback_manager = StoryFeedbackManager()
        
        # Story state
        self.current_chapter = 1
        self.story_history: List[Dict] = []
        
        logger.info("StoryComparisonEngine initialized with all analyzers")
    
    def generate_comparative(
        self,
        prompt: str,
        chapter: int
    ) -> Dict[str, Any]:
        """
        Generate story segment using all three approaches.
        
        Args:
            prompt: User's story prompt
            chapter: Current chapter number
            
        Returns:
            Dictionary with results from all approaches
        """
        self.current_chapter = chapter
        
        # === UNIFIED RAG GENERATION ===
        start = time.time()
        unified_result = self._generate_unified(prompt, chapter)
        unified_time = time.time() - start
        
        # === GRAPH RAG GENERATION ===
        start = time.time()
        graph_result = self._generate_graph(prompt, chapter)
        graph_time = time.time() - start
        
        # === HYBRID FUSION ===
        start = time.time()
        hybrid_result = self._generate_hybrid(
            prompt, chapter, unified_result, graph_result
        )
        hybrid_time = time.time() - start
        
        # Build results
        results = {
            "unified": {
                "text": unified_result.get("text", ""),
                "metrics": GenerationMetrics(
                    response_time=unified_time,
                    consistency_score=self._check_consistency(unified_result.get("text", "")),
                    coherence_score=unified_result.get("coherence", 0.5),
                    retrieval_count=len(unified_result.get("sources", [])),
                    tokens_generated=len(unified_result.get("text", "").split()),
                    source_diversity=self._calc_diversity(unified_result.get("sources", []))
                ),
                "sources": unified_result.get("sources", []),
                "method": unified_result.get("method", "Unified RAG")
            },
            "graph": {
                "text": graph_result.get("text", ""),
                "metrics": GenerationMetrics(
                    response_time=graph_time,
                    consistency_score=self._check_consistency(graph_result.get("text", "")),
                    coherence_score=graph_result.get("coherence", 0.5),
                    retrieval_count=len(graph_result.get("entities", [])),
                    tokens_generated=len(graph_result.get("text", "").split()),
                    source_diversity=self._calc_diversity(graph_result.get("entities", []))
                ),
                "entities": graph_result.get("entities", []),
                "relationships": graph_result.get("relationships", []),
                "character_arcs": graph_result.get("character_arcs", []),
                "plot_suggestions": graph_result.get("plot_suggestions", []),
                "method": graph_result.get("method", "Graph RAG")
            },
            "hybrid": {
                "text": hybrid_result.get("text", ""),
                "metrics": GenerationMetrics(
                    response_time=hybrid_time,
                    consistency_score=self._check_consistency(hybrid_result.get("text", "")),
                    coherence_score=hybrid_result.get("coherence", 0.5),
                    retrieval_count=hybrid_result.get("total_sources", 0),
                    tokens_generated=len(hybrid_result.get("text", "").split()),
                    source_diversity=hybrid_result.get("diversity", 0.5)
                ),
                "fusion_strategy": "Weighted combination of semantic and graph context",
                "method": hybrid_result.get("method", "Hybrid Fusion")
            }
        }
        
        # Update story state with best result (hybrid)
        self._update_story_state(hybrid_result.get("text", ""), chapter)
        
        # Store in history
        self.story_history.append({
            "chapter": chapter,
            "prompt": prompt,
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
        return results
    
    def _generate_unified(self, prompt: str, chapter: int) -> Dict[str, Any]:
        """Generate using Unified RAG approach."""
        if not self.unified_adapter:
            return self._generate_fallback(prompt, chapter, "Unified RAG")
        
        try:
            return self.unified_adapter.generate_story_segment(prompt, chapter)
        except Exception as e:
            logger.error(f"Unified RAG generation failed: {e}")
            return self._generate_fallback(prompt, chapter, "Unified RAG")
    
    def _generate_graph(self, prompt: str, chapter: int) -> Dict[str, Any]:
        """Generate using Graph RAG approach."""
        # Extract entities from prompt
        entities, relationships = self.entity_extractor.extract_entities(prompt, chapter)
        
        # Get character arc states
        character_arcs = []
        for entity in entities:
            if entity.get('type') == 'CHARACTER':
                state = self.arc_tracker.get_current_state_for_generation(
                    entity['id'], chapter
                )
                if state:
                    character_arcs.append({
                        "name": entity['name'],
                        "id": entity['id'],
                        "emotional_state": state.emotional_state,
                        "goals": state.goals,
                        "arc_phase": state.arc_phase,
                        "predicted_next": self.arc_tracker.predict_next_arc_phase(entity['id'])
                    })
        
        # Get multi-hop relationships
        indirect_relationships = []
        for entity in entities:
            if entity.get('type') == 'CHARACTER':
                paths = self.relationship_discovery.discover_indirect_relationships(
                    entity['id'], max_hops=3
                )
                indirect_relationships.extend([p.to_dict() for p in paths[:3]])
        
        # Get graph context
        graph_context = []
        for entity in entities:
            ctx = self.story_graph.get_entity_context(entity['id'], max_hops=2)
            if ctx:
                graph_context.append(ctx)
        
        # Get plot suggestions
        plot_suggestions = self.relationship_discovery.suggest_plot_complications(chapter)
        
        # Build context for generation
        context_parts = []
        
        if character_arcs:
            context_parts.append("CHARACTER STATES:")
            for arc in character_arcs[:3]:
                context_parts.append(
                    f"- {arc['name']}: {arc['emotional_state']}, "
                    f"goals: {', '.join(arc['goals'][:2]) if arc['goals'] else 'none'}, "
                    f"phase: {arc['arc_phase']}"
                )
        
        if indirect_relationships:
            context_parts.append("\nHIDDEN CONNECTIONS:")
            for rel in indirect_relationships[:3]:
                context_parts.append(f"- {rel.get('story_implication', '')}")
        
        if plot_suggestions:
            context_parts.append("\nPOTENTIAL DEVELOPMENTS:")
            for sug in plot_suggestions[:2]:
                context_parts.append(f"- {sug.get('suggestion', '')}")
        
        context_str = "\n".join(context_parts) if context_parts else ""
        
        # Generate with LLM
        full_prompt = f"""You are writing Chapter {chapter} of an ongoing story.

{context_str}

MAINTAIN CONSISTENCY with established character states and relationships.

USER REQUEST: {prompt}

Write the next story segment:"""
        
        if self.llm_router:
            try:
                response = self.llm_router.generate(
                    full_prompt,
                    system_prompt="You are a fiction writer creating consistent narratives. "
                                  "Respect established character arcs and relationships."
                )
            except Exception as e:
                logger.error(f"Graph RAG LLM generation failed: {e}")
                response = "[Generation failed]"
        else:
            response = "[LLM not configured]"
        
        # Calculate coherence
        coherence = 0.5
        if self.embedding_manager and context_str:
            try:
                text_emb = self.embedding_manager.encode_single(response[:1000])
                ctx_emb = self.embedding_manager.encode_single(context_str[:1000])
                coherence = float(np.dot(text_emb, ctx_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(ctx_emb)
                ))
            except Exception:
                pass
        
        return {
            "text": response,
            "entities": entities,
            "relationships": graph_context,
            "character_arcs": character_arcs,
            "plot_suggestions": plot_suggestions,
            "coherence": coherence,
            "method": "Graph RAG - Knowledge Graph Multi-Hop Traversal"
        }
    
    def _generate_hybrid(
        self,
        prompt: str,
        chapter: int,
        unified_result: Dict,
        graph_result: Dict
    ) -> Dict[str, Any]:
        """Generate using hybrid fusion approach."""
        if self.hybrid_generator:
            try:
                return self.hybrid_generator.generate(
                    prompt, chapter, unified_result, graph_result
                )
            except Exception as e:
                logger.error(f"Hybrid fusion failed: {e}")
        
        # Fallback: combine contexts manually
        context = f"""SEMANTIC CONTEXT:
{unified_result.get('sources', [{}])[0].get('content', '')[:500] if unified_result.get('sources') else ''}

RELATIONSHIP CONTEXT:
Characters: {[e.get('name', '') for e in graph_result.get('entities', [])[:3]]}
"""
        
        full_prompt = f"""{context}

USER REQUEST: {prompt}

Generate a story segment that combines semantic richness with relationship consistency:"""
        
        if self.llm_router:
            try:
                response = self.llm_router.generate(full_prompt)
            except Exception as e:
                logger.error(f"Hybrid fallback generation failed: {e}")
                response = "[Generation failed]"
        else:
            response = "[LLM not configured]"
        
        return {
            "text": response,
            "coherence": (unified_result.get('coherence', 0.5) + graph_result.get('coherence', 0.5)) / 2,
            "total_sources": len(unified_result.get('sources', [])) + len(graph_result.get('entities', [])),
            "diversity": 0.5,
            "method": "Hybrid Fusion - Combined Approaches"
        }
    
    def _generate_fallback(self, prompt: str, chapter: int, method: str) -> Dict[str, Any]:
        """Fallback generation when primary method unavailable."""
        if self.llm_router:
            try:
                response = self.llm_router.generate(
                    f"Chapter {chapter}: {prompt}\n\nWrite a story segment:",
                    system_prompt="You are a fiction writer."
                )
            except Exception:
                response = "[Generation unavailable]"
        else:
            response = "[LLM not configured]"
        
        return {
            "text": response,
            "sources": [],
            "coherence": 0.5,
            "method": f"{method} (fallback)"
        }
    
    def _update_story_state(self, text: str, chapter: int) -> None:
        """Update story state with new generated text."""
        # Extract entities and relationships
        entities, relationships = self.entity_extractor.extract_entities(text, chapter)
        
        # Add to graph
        for entity in entities:
            story_entity = StoryEntity(
                id=entity['id'],
                type=entity['type'],
                name=entity['name'],
                attributes=entity.get('attributes', {}),
                first_appearance=chapter
            )
            self.story_graph.add_entity(story_entity)
        
        for rel in relationships:
            story_rel = StoryRelationship(
                source=rel.get('source_id', rel.get('source', '')),
                target=rel.get('target_id', rel.get('target', '')),
                relation_type=rel.get('relation_type', 'INTERACTS_WITH'),
                strength=rel.get('strength', 0.5),
                temporal_context=chapter,
                metadata={"context": rel.get('context', '')}
            )
            self.story_graph.add_relationship(story_rel)
        
        # Update character states
        for entity in entities:
            if entity['type'] == 'CHARACTER':
                self._update_character_state(text, entity, chapter)
        
        # Update unified adapter context
        if self.unified_adapter:
            self.unified_adapter.add_story_segment(text, chapter)
    
    def _update_character_state(self, text: str, entity: Dict, chapter: int) -> None:
        """Update character state based on new text."""
        traits = self.entity_extractor.extract_character_traits(text, entity['name'])
        
        # Infer emotional state
        emotional_state = self._infer_emotion(text, entity['name'])
        
        # Create character state
        state = CharacterState(
            chapter=chapter,
            emotional_state=emotional_state,
            goals=self._extract_goals(text, entity['name']),
            relationships={},
            location="unknown",
            knowledge=[],
            arc_phase=self._infer_arc_phase(chapter)
        )
        
        self.arc_tracker.record_character_state(entity['id'], state)
    
    def _infer_emotion(self, text: str, character_name: str) -> str:
        """Simple emotion inference from text."""
        text_lower = text.lower()
        name_lower = character_name.lower()
        
        # Check for emotion keywords near character name
        if name_lower not in text_lower:
            return "neutral"
        
        emotions = {
            "hopeful": ["smiled", "hoped", "dreamed", "excited", "happy"],
            "desperate": ["cried", "pleaded", "begged", "desperate", "terrified"],
            "angry": ["shouted", "furious", "rage", "angry", "furiously"],
            "conflicted": ["hesitated", "uncertain", "torn", "confused", "wavered"]
        }
        
        for emotion, keywords in emotions.items():
            if any(kw in text_lower for kw in keywords):
                return emotion
        
        return "neutral"
    
    def _extract_goals(self, text: str, character_name: str) -> List[str]:
        """Extract character goals from text."""
        goals = []
        goal_patterns = ["wanted to", "needed to", "hoped to", "planned to", "must", "had to"]
        
        for sentence in text.split('.'):
            if character_name.lower() in sentence.lower():
                for pattern in goal_patterns:
                    if pattern in sentence.lower():
                        # Extract goal phrase
                        parts = sentence.lower().split(pattern)
                        if len(parts) > 1:
                            goal = parts[1].strip()[:50]
                            if goal:
                                goals.append(goal)
        
        return goals[:3]  # Limit to 3 goals
    
    def _infer_arc_phase(self, chapter: int) -> str:
        """Simple chapter-based arc phase."""
        if chapter <= 3:
            return "setup"
        elif chapter <= 8:
            return "rising_action"
        elif chapter <= 12:
            return "crisis"
        elif chapter <= 15:
            return "climax"
        else:
            return "resolution"
    
    def _check_consistency(self, text: str) -> float:
        """Check consistency score based on graph."""
        issues = self.story_graph.detect_inconsistencies()
        return max(0.0, 1.0 - (len(issues) * 0.1))
    
    def _calc_diversity(self, sources: List) -> float:
        """Calculate source diversity."""
        if not sources:
            return 0.0
        
        unique = set()
        for s in sources:
            if isinstance(s, dict):
                unique.add(s.get('content', '')[:50] if 'content' in s else s.get('id', str(s)))
            else:
                unique.add(str(s)[:50])
        
        return len(unique) / len(sources)
    
    def get_story_graph(self) -> StoryKnowledgeGraph:
        """Get the story knowledge graph."""
        return self.story_graph
    
    def get_arc_tracker(self) -> DynamicArcTracker:
        """Get the arc tracker."""
        return self.arc_tracker
    
    def get_story_history(self) -> List[Dict]:
        """Get story generation history."""
        return self.story_history
    
    def get_statistics(self) -> Dict:
        """Get overall statistics."""
        return {
            "graph_stats": self.story_graph.get_statistics(),
            "total_segments": len(self.story_history),
            "current_chapter": self.current_chapter,
            "characters_tracked": len(self.arc_tracker.character_timelines)
        }
    
    # === NEW: Getter methods for analyzers and visualizers ===
    
    def get_coherence_analyzer(self) -> CoherenceAnalyzer:
        """Get the coherence analyzer."""
        return self.coherence_analyzer
    
    def get_consistency_checker(self) -> ConsistencyChecker:
        """Get the consistency checker."""
        return self.consistency_checker
    
    def get_plot_suggestion_engine(self) -> PlotSuggestionEngine:
        """Get the plot suggestion engine."""
        return self.plot_suggestion_engine
    
    def get_graph_visualizer(self) -> GraphVisualizer:
        """Get the graph visualizer."""
        return self.graph_visualizer
    
    def get_timeline_visualizer(self) -> PlotTimelineVisualizer:
        """Get the timeline visualizer."""
        return self.timeline_visualizer
    
    def get_feedback_manager(self) -> StoryFeedbackManager:
        """Get the feedback manager."""
        return self.feedback_manager
    
    def analyze_coherence(
        self,
        generated_text: str,
        context: str,
        chapter: int = 1
    ) -> Dict:
        """Analyze coherence for a generated text."""
        previous_segments = [h.get("results", {}).get("hybrid", {}).get("text", "") 
                           for h in self.story_history[-3:]]
        entities = [e.to_dict() for e in self.story_graph.entities.values()]
        
        breakdown = self.coherence_analyzer.analyze(
            generated_text, context, previous_segments, entities, chapter
        )
        return breakdown.to_dict()
    
    def check_consistency(
        self,
        text: str,
        approach: str,
        chapter: int = 1
    ) -> Dict:
        """Check consistency for a specific approach."""
        entities = [{"name": e.name, "type": e.type} for e in self.story_graph.entities.values()]
        previous_segments = [h.get("results", {}).get("hybrid", {}).get("text", "") 
                           for h in self.story_history[-3:]]
        
        if approach == "unified":
            report = self.consistency_checker.check_unified_rag(text, "", entities, previous_segments)
        else:
            report = self.consistency_checker.check_graph_rag(text, "", entities, previous_segments, chapter)
        
        return report.to_dict()
    
    def get_plot_suggestions(self, limit: int = 5) -> List[Dict]:
        """Get prioritized plot suggestions."""
        suggestions = self.plot_suggestion_engine.generate_suggestions(
            self.current_chapter, limit
        )
        return [s.to_dict() for s in suggestions]
    
    def create_graph_visualization(self, view_type: str = "full") -> Any:
        """Create graph visualization."""
        if view_type == "characters":
            return self.graph_visualizer.create_character_network()
        elif view_type == "events":
            return self.graph_visualizer.create_event_chain()
        elif view_type == "locations":
            return self.graph_visualizer.create_location_map()
        else:
            return self.graph_visualizer.create_full_graph()
    
    def create_timeline_visualization(self) -> Any:
        """Create plot timeline visualization."""
        return self.timeline_visualizer.create_timeline()
    
    def get_timeline_analysis(self) -> Dict:
        """Get timeline structure analysis."""
        return {
            "structure": self.timeline_visualizer.analyze_structure(),
            "plot_holes": self.timeline_visualizer.detect_plot_holes()
        }
    
    def submit_feedback(
        self,
        approach: str,
        prompt: str,
        overall: int = 3,
        consistency: int = 3,
        creativity: int = 3,
        character_authenticity: int = 3,
        plot_coherence: int = 3,
        selected_best: bool = False
    ) -> Dict:
        """Submit feedback for a generation."""
        feedback = self.feedback_manager.submit_feedback(
            chapter=self.current_chapter,
            approach=approach,
            prompt=prompt,
            overall=overall,
            consistency=consistency,
            creativity=creativity,
            character_authenticity=character_authenticity,
            plot_coherence=plot_coherence,
            selected_best=selected_best
        )
        return feedback.to_dict()
    
    def get_feedback_analytics(self) -> Dict:
        """Get feedback analytics."""
        return {
            "performance_by_approach": self.feedback_manager.get_performance_by_approach(),
            "dimension_analysis": self.feedback_manager.get_dimension_analysis(),
            "trends": self.feedback_manager.get_performance_trends(),
            "recommendations": self.feedback_manager.suggest_improvements(),
            "weights": self.feedback_manager.get_weights().to_dict()
        }
    
    def get_recommended_approach(self) -> str:
        """Get recommended approach based on feedback."""
        return self.feedback_manager.get_recommended_approach()
