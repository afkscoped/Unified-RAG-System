"""
Story Unified RAG Adapter

Adapts the existing Unified RAG system for story generation.
Provides story-specific prompting and chapter-aware context filtering.
"""

from typing import Dict, List, Optional, Any
from loguru import logger


class StoryUnifiedRAG:
    """
    Adapter to use existing Unified RAG system for story generation.
    
    Wraps the UnifiedRAGSystem with story-specific functionality:
    - Chapter-aware context filtering
    - Narrative-focused prompts
    - Coherence scoring
    """
    
    def __init__(self, unified_rag_system):
        """
        Initialize story adapter.
        
        Args:
            unified_rag_system: Instance of UnifiedRAGSystem
        """
        self.rag = unified_rag_system
        self.story_context: List[str] = []  # Accumulated story segments
        
        logger.info("StoryUnifiedRAG adapter initialized")
    
    def add_story_segment(self, segment: str, chapter: int) -> None:
        """
        Add a generated story segment to context.
        
        Args:
            segment: Generated story text
            chapter: Chapter number
        """
        self.story_context.append(f"[Chapter {chapter}]\n{segment}")
        logger.debug(f"Added segment for chapter {chapter}")
    
    def get_story_context(self, max_segments: int = 5) -> str:
        """
        Get recent story context for generation.
        
        Args:
            max_segments: Maximum segments to include
            
        Returns:
            Concatenated recent story segments
        """
        recent = self.story_context[-max_segments:]
        return "\n\n".join(recent)
    
    def generate_story_segment(
        self,
        prompt: str,
        chapter: int,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Generate story segment using Unified RAG.
        
        Args:
            prompt: User's story prompt
            chapter: Current chapter number
            top_k: Number of sources to retrieve
            
        Returns:
            Dictionary with text, sources, and coherence score
        """
        # Get relevant context from existing documents
        if hasattr(self.rag, 'hybrid_search') and self.rag.hybrid_search:
            try:
                sources = self.rag.hybrid_search.search_hybrid(prompt, k=top_k)
            except Exception:
                sources = []
        else:
            sources = []
        
        # Build context from sources
        source_context = "\n\n".join([
            s.content[:500] for s in sources[:3]
        ]) if sources else ""
        
        # Get story context
        story_context = self.get_story_context()
        
        # Build story generation prompt
        full_prompt = self._build_story_prompt(
            prompt,
            chapter,
            source_context,
            story_context
        )
        
        # Generate with LLM
        if hasattr(self.rag, 'llm_router') and self.rag.llm_router:
            response = self.rag.llm_router.generate(
                full_prompt,
                system_prompt=self._get_story_system_prompt()
            )
        else:
            response = "[LLM not configured]"
        
        # Calculate coherence
        coherence = self._calculate_coherence(response, story_context)
        
        return {
            "text": response,
            "sources": [{"content": s.content, "score": s.score} for s in sources[:3]],
            "coherence": coherence,
            "method": "Unified RAG - Hybrid BM25 + FAISS Vector Search"
        }
    
    def _build_story_prompt(
        self,
        prompt: str,
        chapter: int,
        source_context: str,
        story_context: str
    ) -> str:
        """Build the full prompt for story generation."""
        parts = []
        
        if story_context:
            parts.append(f"PREVIOUS STORY CONTEXT:\n{story_context[:2000]}")
        
        if source_context:
            parts.append(f"REFERENCE MATERIAL:\n{source_context}")
        
        parts.append(f"CHAPTER: {chapter}")
        parts.append(f"USER REQUEST: {prompt}")
        parts.append("Continue the story naturally, maintaining consistency with previous events:")
        
        return "\n\n".join(parts)
    
    def _get_story_system_prompt(self) -> str:
        """Get system prompt for story generation."""
        return """You are a creative fiction writer. Generate engaging, 
consistent narrative segments that flow naturally from the previous context.
Maintain character voices, setting details, and plot continuity.
Write in a vivid, immersive style."""
    
    def _calculate_coherence(
        self,
        generated_text: str,
        context: str
    ) -> float:
        """
        Calculate coherence score between generated text and context.
        
        Uses embedding similarity if available.
        """
        if not context or not generated_text:
            return 0.5
        
        try:
            if hasattr(self.rag, 'embedding_manager') and self.rag.embedding_manager:
                import numpy as np
                
                text_emb = self.rag.embedding_manager.encode_single(generated_text[:1000])
                ctx_emb = self.rag.embedding_manager.encode_single(context[:1000])
                
                # Cosine similarity
                similarity = np.dot(text_emb, ctx_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(ctx_emb)
                )
                return float(similarity)
        except Exception as e:
            logger.warning(f"Coherence calculation failed: {e}")
        
        return 0.5
    
    def clear_context(self) -> None:
        """Clear accumulated story context."""
        self.story_context = []
        logger.info("Story context cleared")
