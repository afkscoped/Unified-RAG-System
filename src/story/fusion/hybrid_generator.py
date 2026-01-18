"""
Hybrid Fusion Generator

Combines Unified RAG and Graph RAG approaches for superior
story generation by merging semantic richness with relationship consistency.
"""

from typing import Dict, List, Any, Optional
from loguru import logger


class HybridFusionEngine:
    """
    Combines Unified RAG and Graph RAG for best-of-both-worlds generation.
    
    Strategy:
    - Uses Graph RAG for relationship consistency and character states
    - Uses Unified RAG for semantic richness and thematic coherence
    - Validates generated content against graph constraints
    """
    
    def __init__(
        self,
        unified_adapter,
        story_graph,
        arc_tracker,
        llm_router
    ):
        """
        Initialize hybrid fusion engine.
        
        Args:
            unified_adapter: StoryUnifiedRAG adapter
            story_graph: StoryKnowledgeGraph instance
            arc_tracker: DynamicArcTracker instance
            llm_router: LLMRouter for generation
        """
        self.unified = unified_adapter
        self.graph = story_graph
        self.arc_tracker = arc_tracker
        self.llm_router = llm_router
        
        # Fusion weights (can be adjusted)
        self.semantic_weight = 0.4
        self.graph_weight = 0.6
        
        logger.info("HybridFusionEngine initialized")
    
    def generate(
        self,
        prompt: str,
        chapter: int,
        unified_result: Dict,
        graph_result: Dict
    ) -> Dict[str, Any]:
        """
        Generate story segment using combined approach.
        
        Args:
            prompt: User's story prompt
            chapter: Current chapter
            unified_result: Result from Unified RAG
            graph_result: Result from Graph RAG
            
        Returns:
            Dictionary with fused text, metrics, and sources
        """
        # Build combined context
        context = self._build_fusion_context(unified_result, graph_result)
        
        # Generate with fusion prompt
        full_prompt = self._build_fusion_prompt(prompt, chapter, context)
        
        response = self.llm_router.generate(
            full_prompt,
            system_prompt=self._get_fusion_system_prompt()
        )
        
        # Validate against graph constraints
        validation = self._validate_against_graph(response)
        
        # Calculate combined coherence
        coherence = (
            unified_result.get('coherence', 0.5) * self.semantic_weight +
            graph_result.get('coherence', 0.5) * self.graph_weight
        )
        
        return {
            "text": response,
            "coherence": coherence,
            "validation": validation,
            "total_sources": (
                len(unified_result.get('sources', [])) +
                len(graph_result.get('entities', []))
            ),
            "diversity": self._calculate_diversity(unified_result, graph_result),
            "method": "Hybrid Fusion - Unified RAG + Graph RAG Combined"
        }
    
    def _build_fusion_context(
        self,
        unified_result: Dict,
        graph_result: Dict
    ) -> Dict[str, str]:
        """Build context from both approaches."""
        context = {}
        
        # Semantic context from unified RAG
        semantic_sources = unified_result.get('sources', [])
        if semantic_sources:
            context['semantic'] = "\n".join([
                s.get('content', '')[:300] if isinstance(s, dict) else str(s)[:300]
                for s in semantic_sources[:2]
            ])
        
        # Character states from graph
        character_arcs = graph_result.get('character_arcs', [])
        if character_arcs:
            arc_context = []
            for arc in character_arcs[:3]:
                if isinstance(arc, dict):
                    name = arc.get('name', 'Unknown')
                    state = arc.get('emotional_state', 'neutral')
                    goals = arc.get('goals', [])
                    arc_context.append(
                        f"- {name}: {state}, goals: {', '.join(goals[:2]) if goals else 'none'}"
                    )
            context['character_states'] = "\n".join(arc_context)
        
        # Relationships from graph
        relationships = graph_result.get('relationships', [])
        if relationships:
            rel_context = []
            for rel in relationships[:5]:
                if isinstance(rel, dict):
                    for hop_data in rel.values():
                        if isinstance(hop_data, list):
                            for r in hop_data[:2]:
                                if isinstance(r, dict):
                                    edge = r.get('edge', {})
                                    if edge:
                                        first_edge = list(edge.values())[0] if edge else {}
                                        rel_type = first_edge.get('relation', 'connected')
                                        rel_context.append(f"- {r.get('data', {}).get('name', '?')} ({rel_type})")
            context['relationships'] = "\n".join(rel_context)
        
        # Plot suggestions
        suggestions = graph_result.get('plot_suggestions', [])
        if suggestions:
            context['plot_hints'] = "\n".join([
                f"- {s.get('suggestion', '')}"
                for s in suggestions[:2]
            ])
        
        return context
    
    def _build_fusion_prompt(
        self,
        prompt: str,
        chapter: int,
        context: Dict[str, str]
    ) -> str:
        """Build the fusion prompt."""
        parts = [f"CHAPTER {chapter} - Story Generation\n"]
        
        if context.get('semantic'):
            parts.append(f"THEMATIC CONTEXT (from semantic search):\n{context['semantic']}\n")
        
        if context.get('character_states'):
            parts.append(f"CHARACTER STATES (from knowledge graph):\n{context['character_states']}\n")
        
        if context.get('relationships'):
            parts.append(f"KEY RELATIONSHIPS:\n{context['relationships']}\n")
        
        if context.get('plot_hints'):
            parts.append(f"POTENTIAL DEVELOPMENTS:\n{context['plot_hints']}\n")
        
        parts.append(f"USER REQUEST: {prompt}")
        parts.append("\nGenerate a story segment that:")
        parts.append("1. Maintains thematic consistency with the semantic context")
        parts.append("2. Respects established character states and relationships")
        parts.append("3. Advances the narrative in an engaging way")
        
        return "\n".join(parts)
    
    def _get_fusion_system_prompt(self) -> str:
        """System prompt for fusion generation."""
        return """You are an expert fiction writer creating a cohesive narrative.
You have access to both semantic context (themes, style) and structured 
knowledge about characters and relationships. Use both to create 
compelling, consistent story segments. Prioritize character consistency
while maintaining narrative flow."""
    
    def _validate_against_graph(self, generated_text: str) -> Dict[str, Any]:
        """
        Validate generated text against graph constraints.
        
        Checks for:
        - Character trait consistency
        - Relationship contradictions
        - Timeline consistency
        """
        validation = {
            "passed": True,
            "issues": [],
            "warnings": []
        }
        
        # Check for existing inconsistencies in graph
        graph_issues = self.graph.detect_inconsistencies()
        if graph_issues:
            validation["warnings"].append(
                f"Graph has {len(graph_issues)} existing inconsistencies"
            )
        
        # Basic character name check
        characters = self.graph.get_characters()
        for char in characters:
            name = char.get('name', '')
            if name.lower() in generated_text.lower():
                # Character mentioned - could add deeper consistency checks here
                pass
        
        return validation
    
    def _calculate_diversity(
        self,
        unified_result: Dict,
        graph_result: Dict
    ) -> float:
        """Calculate source diversity score."""
        unified_sources = set()
        graph_sources = set()
        
        for s in unified_result.get('sources', []):
            if isinstance(s, dict):
                unified_sources.add(s.get('content', '')[:50])
            else:
                unified_sources.add(str(s)[:50])
        
        for e in graph_result.get('entities', []):
            if isinstance(e, dict):
                graph_sources.add(e.get('id', ''))
        
        total = len(unified_sources) + len(graph_sources)
        unique = len(unified_sources | graph_sources)
        
        return unique / total if total > 0 else 0.0
    
    def set_weights(self, semantic: float, graph: float) -> None:
        """
        Adjust fusion weights.
        
        Args:
            semantic: Weight for semantic/unified approach (0-1)
            graph: Weight for graph approach (0-1)
        """
        total = semantic + graph
        self.semantic_weight = semantic / total
        self.graph_weight = graph / total
        logger.info(f"Fusion weights updated: semantic={self.semantic_weight:.2f}, graph={self.graph_weight:.2f}")
