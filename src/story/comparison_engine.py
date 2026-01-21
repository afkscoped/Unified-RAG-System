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
from src.story.analysis.trope_detector import TropeDetector
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
        self.trope_detector = TropeDetector(self.story_graph, self.arc_tracker)
        
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
        
        # Get feedback-driven weights for fusion
        weights = self.feedback_manager.get_weights()
        
        # === UNIFIED RAG GENERATION ===
        start = time.time()
        unified_result = self._generate_unified(prompt, chapter)
        unified_time = time.time() - start
        unified_text = unified_result.get("text", "")
        
        # === GRAPH RAG GENERATION ===
        start = time.time()
        graph_result = self._generate_graph(prompt, chapter)
        graph_time = time.time() - start
        graph_text = graph_result.get("text", "")
        
        # === HYBRID FUSION ===
        start = time.time()
        hybrid_result = self._generate_hybrid(
            prompt, chapter, unified_result, graph_result
        )
        hybrid_time = time.time() - start
        hybrid_text = hybrid_result.get("text", "")
        
        # Calculate approach-specific metrics
        unified_coherence = self._calculate_coherence(unified_text, prompt, "unified")
        graph_coherence = self._calculate_coherence(graph_text, prompt, "graph")
        hybrid_coherence = self._calculate_coherence(hybrid_text, prompt, "hybrid")
        
        unified_consistency = self._calculate_consistency(unified_text, "unified", chapter)
        graph_consistency = self._calculate_consistency(graph_text, "graph", chapter)
        hybrid_consistency = self._calculate_consistency(hybrid_text, "hybrid", chapter)
        
        # Build results
        results = {
            "unified": {
                "text": unified_text,
                "metrics": GenerationMetrics(
                    response_time=unified_time,
                    consistency_score=unified_consistency,
                    coherence_score=unified_coherence,
                    retrieval_count=len(unified_result.get("sources", [])),
                    tokens_generated=len(unified_text.split()),
                    source_diversity=self._calc_diversity(unified_result.get("sources", []))
                ),
                "sources": unified_result.get("sources", []),
                "method": unified_result.get("method", "Unified RAG")
            },
            "graph": {
                "text": graph_text,
                "metrics": GenerationMetrics(
                    response_time=graph_time,
                    consistency_score=graph_consistency,
                    coherence_score=graph_coherence,
                    retrieval_count=len(graph_result.get("entities", [])),
                    tokens_generated=len(graph_text.split()),
                    source_diversity=self._calc_diversity(graph_result.get("entities", []))
                ),
                "entities": graph_result.get("entities", []),
                "relationships": graph_result.get("relationships", []),
                "character_arcs": graph_result.get("character_arcs", []),
                "plot_suggestions": graph_result.get("plot_suggestions", []),
                "method": graph_result.get("method", "Graph RAG")
            },
            "hybrid": {
                "text": hybrid_text,
                "metrics": GenerationMetrics(
                    response_time=hybrid_time,
                    consistency_score=hybrid_consistency,
                    coherence_score=hybrid_coherence,
                    retrieval_count=hybrid_result.get("total_sources", 0),
                    tokens_generated=len(hybrid_text.split()),
                    source_diversity=hybrid_result.get("diversity", 0.5)
                ),
                "fusion_strategy": "Weighted combination of semantic and graph context",
                "method": hybrid_result.get("method", "Hybrid Fusion")
            }
        }
        
        # Update story state with hybrid result
        self._update_story_state(hybrid_text, chapter)
        
        # Run trope detection on all approaches
        for approach, text in [("unified", unified_text), ("graph", graph_text), ("hybrid", hybrid_text)]:
            if text and len(text) > 50:
                entities = [{"name": e.name, "type": e.type} for e in self.story_graph.entities.values()]
                self.trope_detector.analyze_segment(text, chapter, entities)
        
        # Extract plot points
        self.plot_extractor.extract_from_text(hybrid_text, chapter,
            [{"name": e.name, "type": e.type} for e in self.story_graph.entities.values()])
        
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
        """Legacy consistency check - use _calculate_consistency for approach-specific."""
        issues = self.story_graph.detect_inconsistencies()
        return max(0.0, 1.0 - (len(issues) * 0.1))
    
    def _calculate_coherence(self, text: str, context: str, approach: str) -> float:
        """
        Calculate approach-specific coherence score.
        Different approaches have inherently different coherence profiles.
        """
        if not text or len(text) < 20:
            return 0.3
        
        base_coherence = 0.5
        
        # Approach-specific adjustments
        if approach == "unified":
            # Unified RAG: good at semantic similarity but may miss context
            # Check for contextual keywords
            context_words = set(context.lower().split())
            text_words = set(text.lower().split())
            overlap = len(context_words & text_words)
            base_coherence = min(0.8, 0.4 + (overlap * 0.05))
            
        elif approach == "graph":
            # Graph RAG: tracks entities and relationships
            # Higher coherence if it mentions known entities
            entity_names = [e.name.lower() for e in self.story_graph.entities.values()]
            mentions = sum(1 for name in entity_names if name in text.lower())
            base_coherence = min(0.9, 0.5 + (mentions * 0.08))
            
        elif approach == "hybrid":
            # Hybrid should have balanced coherence
            context_words = set(context.lower().split())
            text_words = set(text.lower().split())
            overlap = len(context_words & text_words)
            entity_names = [e.name.lower() for e in self.story_graph.entities.values()]
            mentions = sum(1 for name in entity_names if name in text.lower())
            base_coherence = min(0.95, 0.5 + (overlap * 0.03) + (mentions * 0.05))
        
        # Add some variance based on text quality indicators
        if len(text) > 500:
            base_coherence += 0.05
        if any(p in text.lower() for p in ["however", "therefore", "meanwhile", "suddenly"]):
            base_coherence += 0.03
            
        return min(1.0, max(0.1, base_coherence))
    
    def _calculate_consistency(self, text: str, approach: str, chapter: int) -> float:
        """
        Calculate approach-specific consistency score.
        Graph-based approaches should have higher consistency due to tracking.
        """
        if not text or len(text) < 20:
            return 0.5
        
        # Base consistency from graph issues
        issues = self.story_graph.detect_inconsistencies()
        base_consistency = max(0.0, 1.0 - (len(issues) * 0.1))
        
        # Approach-specific adjustments
        if approach == "unified":
            # Unified RAG has no entity tracking, lower consistency
            # Penalize for not mentioning known characters
            characters = [e.name.lower() for e in self.story_graph.entities.values() if e.type == "CHARACTER"]
            if characters:
                mentions = sum(1 for c in characters if c in text.lower())
                char_consistency = mentions / max(1, len(characters))
                base_consistency = (base_consistency * 0.5) + (char_consistency * 0.3) + 0.1
            else:
                base_consistency = max(0.4, base_consistency - 0.15)
                
        elif approach == "graph":
            # Graph RAG is context-aware, higher consistency
            base_consistency = min(0.98, base_consistency + 0.15)
            # Bonus for using graph relationships
            if any(rel in text.lower() for rel in ["ally", "enemy", "mentor", "friend", "foe"]):
                base_consistency = min(0.98, base_consistency + 0.05)
                
        elif approach == "hybrid":
            # Hybrid gets good consistency from graph portion
            base_consistency = min(0.95, base_consistency + 0.08)
        
        return min(1.0, max(0.1, base_consistency))
    
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
    
    # === NEW: Trope Detection & Enhanced Visualization ===
    
    def get_trope_detector(self) -> TropeDetector:
        """Get the trope detector."""
        return self.trope_detector
    
    def analyze_tropes(self, text: str, chapter: int) -> List[Dict]:
        """Analyze text for narrative tropes."""
        entities = [{"name": e.name, "type": e.type} for e in self.story_graph.entities.values()]
        new_tropes = self.trope_detector.analyze_segment(text, chapter, entities)
        return [t.to_dict() for t in new_tropes]
    
    def get_thematic_summary(self) -> Dict:
        """Get thematic summary of detected tropes."""
        return self.trope_detector.get_thematic_summary()
    
    def create_arc_visualization(self) -> Any:
        """Create character arc visualization."""
        return self.timeline_visualizer.create_character_arc_chart(self.arc_tracker)
    
    def bootstrap_sample_story(self) -> Dict:
        """
        Bootstrap with sample story data for demo purposes.
        Populates the graph with sample characters, relationships, and events.
        """
        # Sample characters
        sample_entities = [
            StoryEntity(id="elena", type="CHARACTER", name="Elena", 
                       attributes={"role": "protagonist"}, first_appearance=1),
            StoryEntity(id="marcus", type="CHARACTER", name="Marcus",
                       attributes={"role": "mentor"}, first_appearance=1),
            StoryEntity(id="raven", type="CHARACTER", name="Raven",
                       attributes={"role": "antagonist"}, first_appearance=2),
            StoryEntity(id="castle", type="LOCATION", name="Shadowkeep Castle",
                       attributes={"type": "fortress"}, first_appearance=1),
            StoryEntity(id="ancient_scroll", type="ARTIFACT", name="Ancient Scroll",
                       attributes={"power": "prophecy"}, first_appearance=1),
        ]
        
        for entity in sample_entities:
            self.story_graph.add_entity(entity)
        
        # Sample relationships
        sample_rels = [
            StoryRelationship(source="elena", target="marcus", relation_type="MENTORED_BY", 
                            strength=0.9, temporal_context=1),
            StoryRelationship(source="elena", target="raven", relation_type="CONFLICTS_WITH",
                            strength=0.8, temporal_context=2),
            StoryRelationship(source="marcus", target="raven", relation_type="FAMILY",
                            strength=0.7, temporal_context=2, metadata={"detail": "siblings"}),
            StoryRelationship(source="elena", target="castle", relation_type="LOCATED_IN",
                            strength=0.5, temporal_context=1),
        ]
        
        for rel in sample_rels:
            self.story_graph.add_relationship(rel)
        
        # Sample character states
        sample_states = [
            ("elena", CharacterState(chapter=1, emotional_state="hopeful", 
                                     goals=["find the truth", "master her powers"],
                                     relationships={"marcus": 0.9}, location="castle",
                                     knowledge=["scroll exists"], arc_phase="setup")),
            ("elena", CharacterState(chapter=2, emotional_state="conflicted",
                                     goals=["defeat raven", "save marcus"],
                                     relationships={"marcus": 0.9, "raven": -0.8}, location="forest",
                                     knowledge=["raven is marcus sister"], arc_phase="rising_action")),
            ("marcus", CharacterState(chapter=1, emotional_state="neutral",
                                      goals=["train elena", "protect scroll"],
                                      relationships={"elena": 0.8}, location="castle",
                                      knowledge=["ancient prophecy"], arc_phase="setup")),
        ]
        
        for char_id, state in sample_states:
            self.arc_tracker.record_character_state(char_id, state)
        
        # Sample story segment for trope detection
        sample_text = """Elena had always known she was destined for something greater. 
        When the ancient scroll was discovered in the hidden chamber, her mentor Marcus 
        revealed the truth about her powers. She was the chosen one, called to adventure 
        by forces beyond her understanding. But dark forces were gathering. Raven, the 
        fallen warrior who had once been noble, now sought to corrupt the power for herself.
        The betrayal cut deep when Elena learned Raven was Marcus's own sister."""
        
        self.trope_detector.analyze_segment(sample_text, 1, 
            [{"name": e.name, "type": e.type} for e in sample_entities if e.type == "CHARACTER"])
        
        # Extract plot points
        self.plot_extractor.extract_from_text(sample_text, 1, 
            [{"name": e.name, "type": e.type} for e in sample_entities])
        
        self.current_chapter = 2
        
        return {
            "status": "success",
            "entities_added": len(sample_entities),
            "relationships_added": len(sample_rels),
            "character_states_added": len(sample_states),
            "tropes_detected": len(self.trope_detector.get_all_tropes()),
            "message": "Sample story data loaded! Try the visualization features."
        }
    
    # === NEW: Advanced Story Analysis Features ===
    
    def detect_plot_holes(self) -> Dict:
        """
        Comprehensive plot hole detection across the story.
        """
        plot_holes = []
        
        # 1. Graph-based inconsistencies
        graph_issues = self.story_graph.detect_inconsistencies()
        for issue in graph_issues:
            plot_holes.append({
                "type": issue["type"],
                "severity": "high",
                "description": issue["description"],
                "chapter": None,
                "suggestion": "Review and clarify this inconsistency"
            })
        
        # 2. Disappeared characters
        characters = self.story_graph.get_characters()
        for char in characters:
            last_mention = self._find_last_mention(char["name"])
            if last_mention and self.current_chapter - last_mention > 2:
                plot_holes.append({
                    "type": "disappeared_character",
                    "severity": "medium",
                    "description": f"{char['name']} hasn't appeared since Chapter {last_mention}",
                    "chapter": last_mention,
                    "suggestion": f"Bring {char['name']} back or explain their absence"
                })
        
        # 3. Chekhov's Gun violations
        artifacts = self.story_graph.get_entities_by_type("ARTIFACT")
        for artifact in artifacts:
            if self.current_chapter - artifact.first_appearance >= 2:
                relations = list(self.story_graph.graph.edges(artifact.id, data=True))
                if len(relations) < 2:
                    plot_holes.append({
                        "type": "chekhovs_gun",
                        "severity": "medium",
                        "description": f"'{artifact.name}' introduced but not utilized",
                        "chapter": artifact.first_appearance,
                        "suggestion": f"Use '{artifact.name}' in the plot"
                    })
        
        # 4. Character arc issues
        for char_id, states in self.arc_tracker.character_timelines.items():
            if len(states) >= 2:
                sorted_states = sorted(states, key=lambda s: s.chapter)
                for i in range(1, len(sorted_states)):
                    prev, curr = sorted_states[i-1], sorted_states[i]
                    emotion_change = self._emotion_distance(prev.emotional_state, curr.emotional_state)
                    if emotion_change > 0.7:
                        char_name = self.story_graph.get_entity(char_id)
                        char_name = char_name.name if char_name else char_id
                        plot_holes.append({
                            "type": "abrupt_change",
                            "severity": "low",
                            "description": f"{char_name} changed from {prev.emotional_state} to {curr.emotional_state} suddenly",
                            "chapter": curr.chapter,
                            "suggestion": "Add transitional scenes"
                        })
        
        high_count = len([p for p in plot_holes if p["severity"] == "high"])
        med_count = len([p for p in plot_holes if p["severity"] == "medium"])
        score = 100 - (high_count * 20) - (med_count * 5)
        health = "Excellent 🌟" if score >= 90 else "Good 👍" if score >= 70 else "Needs Work 🔧" if score >= 50 else "Critical ⚠️"
        
        return {
            "total_issues": len(plot_holes),
            "by_severity": {"high": high_count, "medium": med_count, "low": len([p for p in plot_holes if p["severity"] == "low"])},
            "issues": plot_holes,
            "story_health": health
        }
    
    def _find_last_mention(self, character_name: str) -> Optional[int]:
        """Find the last chapter where a character was mentioned."""
        last_chapter = None
        for entry in self.story_history:
            for approach in ["unified", "graph", "hybrid"]:
                text = entry.get("results", {}).get(approach, {}).get("text", "")
                if character_name.lower() in text.lower():
                    last_chapter = entry["chapter"]
        return last_chapter
    
    def _emotion_distance(self, emotion1: str, emotion2: str) -> float:
        """Calculate distance between emotional states."""
        values = {"hopeful": 0.8, "excited": 0.9, "happy": 0.85, "neutral": 0.5,
                  "uncertain": 0.4, "conflicted": 0.35, "angry": 0.3, "desperate": 0.15, "terrified": 0.1}
        return abs(values.get(emotion1.lower(), 0.5) - values.get(emotion2.lower(), 0.5))
    
    def track_foreshadowing(self) -> Dict:
        """Track foreshadowing elements and their payoffs."""
        patterns = ["prophecy", "warning", "omen", "secret", "promised", "destined", "mysterious", "ancient", "hidden"]
        plants = []
        
        for entry in self.story_history:
            text = entry.get("results", {}).get("hybrid", {}).get("text", "")
            for pattern in patterns:
                if pattern in text.lower():
                    for sentence in text.split('.'):
                        if pattern in sentence.lower():
                            plants.append({"chapter": entry["chapter"], "pattern": pattern, "hint": sentence.strip()[:80], "resolved": False})
                            break
        
        # Check resolutions
        for plant in plants:
            for entry in self.story_history:
                if entry["chapter"] > plant["chapter"]:
                    text = entry.get("results", {}).get("hybrid", {}).get("text", "")
                    if plant["pattern"] in text.lower():
                        plant["resolved"] = True
                        break
        
        pending = [p for p in plants if not p["resolved"]]
        return {
            "total": len(plants), "resolved": len([p for p in plants if p["resolved"]]),
            "pending": len(pending), "pending_items": pending[:5],
            "suggestions": [f"Resolve '{p['pattern']}' from Ch.{p['chapter']}" for p in pending[:3]]
        }
    
    def analyze_narrative_pacing(self) -> Dict:
        """Analyze narrative pacing and tension curve."""
        pacing_data = []
        tension_words = ["battle", "fight", "danger", "fear", "desperate", "urgent", "chase", "conflict"]
        calm_words = ["peaceful", "quiet", "rest", "calm", "gentle", "slowly"]
        
        for entry in self.story_history:
            text = entry.get("results", {}).get("hybrid", {}).get("text", "")
            text_lower = text.lower()
            tension = sum(1 for w in tension_words if w in text_lower)
            calm = sum(1 for w in calm_words if w in text_lower)
            net = (tension - calm) / max(1, tension + calm)
            normalized = (net + 1) / 2
            
            pacing_data.append({
                "chapter": entry["chapter"],
                "tension_level": round(normalized, 2),
                "word_count": len(text.split()),
                "pacing": "intense" if normalized > 0.7 else "slow" if normalized < 0.3 else "balanced"
            })
        
        issues = []
        for i in range(1, len(pacing_data)):
            if pacing_data[i-1]["pacing"] == pacing_data[i]["pacing"] and pacing_data[i]["pacing"] in ["slow", "intense"]:
                issues.append(f"Ch.{pacing_data[i-1]['chapter']}-{pacing_data[i]['chapter']}: consecutive {pacing_data[i]['pacing']} pacing")
        
        return {"chapters": pacing_data, "pacing_issues": issues, "average_tension": np.mean([p["tension_level"] for p in pacing_data]) if pacing_data else 0}
    
    def get_feedback_influence(self) -> Dict:
        """Get how feedback has influenced generation weights."""
        weights = self.feedback_manager.get_weights()
        performance = self.feedback_manager.get_performance_by_approach()
        best = max(performance, key=performance.get) if performance else "hybrid"
        
        return {
            "current_weights": weights.to_dict(),
            "performance_scores": performance,
            "recommended_approach": best,
            "feedback_count": len(self.feedback_manager.feedback_history),
            "influence": f"Based on feedback, {best.title()} performs best. Weights favor successful approaches." if performance else "Provide ratings to influence generation."
        }
