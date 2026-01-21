"""
Multi-Dimensional Coherence Analyzer

Evaluates story coherence across multiple dimensions:
- Semantic: Embedding similarity between context and output
- Lexical: N-gram overlap and ROUGE scores
- Discourse: Entity continuity and coreference
- Temporal: Timeline and event sequencing consistency
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
import re
import numpy as np
from loguru import logger


@dataclass
class CoherenceBreakdown:
    """Breakdown of coherence scores by dimension."""
    semantic: float = 0.0
    lexical: float = 0.0
    discourse: float = 0.0
    temporal: float = 0.0
    voice: float = 0.0
    composite: float = 0.0
    
    # Detailed sub-scores
    details: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "semantic": self.semantic,
            "lexical": self.lexical,
            "discourse": self.discourse,
            "temporal": self.temporal,
            "voice": self.voice,
            "composite": self.composite,
            "details": self.details
        }


class CoherenceAnalyzer:
    """
    Multi-dimensional coherence scoring for story generation.
    
    Provides clear differentiation in quality scores by evaluating
    multiple aspects of narrative coherence.
    """
    
    def __init__(
        self,
        embedding_manager=None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize coherence analyzer.
        
        Args:
            embedding_manager: For semantic similarity calculations
            weights: Custom weights for each dimension (default: equal)
        """
        self.embedding_manager = embedding_manager
        self.weights = weights or {
            "semantic": 0.25,
            "lexical": 0.20,
            "discourse": 0.25,
            "temporal": 0.15,
            "voice": 0.15
        }
        
        logger.info("CoherenceAnalyzer initialized")
    
    def analyze(
        self,
        generated_text: str,
        context: str,
        previous_segments: Optional[List[str]] = None,
        entities: Optional[List[Dict]] = None,
        chapter: int = 1
    ) -> CoherenceBreakdown:
        """
        Perform full coherence analysis.
        
        Args:
            generated_text: The generated story segment
            context: The context/prompt used for generation
            previous_segments: Previous story segments for continuity
            entities: Known entities from the story graph
            chapter: Current chapter number
            
        Returns:
            CoherenceBreakdown with all dimension scores
        """
        previous_segments = previous_segments or []
        entities = entities or []
        
        # Calculate each dimension
        semantic = self.calculate_semantic_coherence(generated_text, context)
        lexical = self.calculate_lexical_coherence(generated_text, context, previous_segments)
        discourse = self.calculate_discourse_coherence(generated_text, entities, previous_segments)
        temporal = self.calculate_temporal_coherence(generated_text, chapter, previous_segments)
        voice = self.calculate_voice_coherence(generated_text, context, previous_segments)
        
        # Calculate weighted composite
        composite = (
            self.weights["semantic"] * semantic +
            self.weights["lexical"] * lexical +
            self.weights["discourse"] * discourse +
            self.weights["temporal"] * temporal +
            self.weights["voice"] * voice
        )
        
        return CoherenceBreakdown(
            semantic=semantic,
            lexical=lexical,
            discourse=discourse,
            temporal=temporal,
            voice=voice,
            composite=composite,
            details={
                "semantic_context_similarity": semantic,
                "lexical_ngram_overlap": lexical,
                "entity_continuity": discourse,
                "temporal_consistency": temporal,
                "voice_consistency": voice
            }
        )
    
    def calculate_semantic_coherence(
        self,
        generated_text: str,
        context: str
    ) -> float:
        """
        Calculate semantic coherence via embedding similarity.
        
        Uses multiple comparison approaches for more variance:
        1. Overall text similarity
        2. Key phrase similarity
        3. Sentence-level similarity variance
        """
        if not self.embedding_manager or not generated_text or not context:
            return 0.5
        
        try:
            # Overall similarity
            gen_emb = self.embedding_manager.encode_single(generated_text[:2000])
            ctx_emb = self.embedding_manager.encode_single(context[:2000])
            overall_sim = self._cosine_similarity(gen_emb, ctx_emb)
            
            # Sentence-level similarity (more granular)
            gen_sentences = self._split_sentences(generated_text)[:10]
            ctx_sentences = self._split_sentences(context)[:10]
            
            if gen_sentences and ctx_sentences:
                gen_sent_embs = [self.embedding_manager.encode_single(s) for s in gen_sentences[:5]]
                ctx_sent_embs = [self.embedding_manager.encode_single(s) for s in ctx_sentences[:5]]
                
                # Max similarity for each generated sentence
                sentence_sims = []
                for g_emb in gen_sent_embs:
                    max_sim = max(self._cosine_similarity(g_emb, c_emb) for c_emb in ctx_sent_embs)
                    sentence_sims.append(max_sim)
                
                sentence_avg = np.mean(sentence_sims) if sentence_sims else 0.5
            else:
                sentence_avg = overall_sim
            
            # Combine with variance penalty
            variance = np.var(sentence_sims) if len(sentence_sims) > 1 else 0
            consistency_bonus = max(0, 0.1 - variance)  # Reward consistent similarity
            
            return min(1.0, overall_sim * 0.6 + sentence_avg * 0.3 + consistency_bonus)
            
        except Exception as e:
            logger.warning(f"Semantic coherence calculation failed: {e}")
            return 0.5
    
    def calculate_lexical_coherence(
        self,
        generated_text: str,
        context: str,
        previous_segments: List[str]
    ) -> float:
        """
        Calculate lexical coherence via n-gram overlap.
        
        Measures:
        1. Unigram overlap (vocabulary consistency)
        2. Bigram overlap (phrase consistency)
        3. ROUGE-L approximation (longest common subsequence)
        """
        if not generated_text:
            return 0.0
        
        # Combine context with previous segments
        reference_text = context + " " + " ".join(previous_segments[-3:])
        
        # Tokenize
        gen_tokens = self._tokenize(generated_text.lower())
        ref_tokens = self._tokenize(reference_text.lower())
        
        if not gen_tokens or not ref_tokens:
            return 0.5
        
        # Unigram overlap
        gen_set = set(gen_tokens)
        ref_set = set(ref_tokens)
        unigram_overlap = len(gen_set & ref_set) / max(len(gen_set), 1)
        
        # Bigram overlap
        gen_bigrams = set(zip(gen_tokens[:-1], gen_tokens[1:]))
        ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:]))
        bigram_overlap = len(gen_bigrams & ref_bigrams) / max(len(gen_bigrams), 1) if gen_bigrams else 0
        
        # ROUGE-L approximation (LCS-based)
        lcs_score = self._rouge_l_score(gen_tokens[:100], ref_tokens[:100])
        
        # Weighted combination
        return unigram_overlap * 0.3 + bigram_overlap * 0.3 + lcs_score * 0.4
    
    def calculate_discourse_coherence(
        self,
        generated_text: str,
        entities: List[Dict],
        previous_segments: List[str]
    ) -> float:
        """
        Calculate discourse coherence via entity continuity.
        
        Measures:
        1. Entity mention consistency (known entities referenced)
        2. New entity introduction rate (not too many new entities)
        3. Pronoun resolution potential
        """
        if not generated_text:
            return 0.0
        
        text_lower = generated_text.lower()
        
        # Count known entity mentions
        known_mentioned = 0
        for entity in entities:
            name = entity.get('name', '').lower()
            if name and name in text_lower:
                known_mentioned += 1
        
        entity_mention_rate = known_mentioned / max(len(entities), 1) if entities else 0.5
        
        # Detect new entities (capitalized words not in entity list)
        known_names = set(e.get('name', '').lower() for e in entities)
        potential_new = set(re.findall(r'\b[A-Z][a-z]+\b', generated_text))
        new_entities = potential_new - known_names - {'The', 'A', 'An', 'He', 'She', 'It', 'They', 'We', 'I'}
        
        # Penalize too many new entities (confusing)
        new_entity_penalty = max(0, 1 - len(new_entities) * 0.15)
        
        # Check reference to previous content
        prev_text = " ".join(previous_segments[-2:]).lower() if previous_segments else ""
        if prev_text:
            # Find shared entity references
            prev_entities = set(re.findall(r'\b[A-Z][a-z]+\b', " ".join(previous_segments[-2:])))
            shared = len(potential_new & prev_entities) / max(len(potential_new), 1) if potential_new else 0.5
        else:
            shared = 0.5
        
        return entity_mention_rate * 0.4 + new_entity_penalty * 0.3 + shared * 0.3
    
    def calculate_temporal_coherence(
        self,
        generated_text: str,
        chapter: int,
        previous_segments: List[str]
    ) -> float:
        """
        Calculate temporal coherence via timeline consistency.
        
        Measures:
        1. Tense consistency
        2. Temporal marker appropriateness
        3. Event sequencing logic
        """
        if not generated_text:
            return 0.0
        
        # Analyze tense distribution
        past_markers = len(re.findall(r'\b(was|were|had|did|went|said|came|saw|took|made)\b', 
                                       generated_text.lower()))
        present_markers = len(re.findall(r'\b(is|are|has|does|goes|says|comes|sees|takes|makes)\b',
                                          generated_text.lower()))
        
        total_markers = past_markers + present_markers
        if total_markers > 0:
            # Narrative should be predominantly one tense
            tense_consistency = max(past_markers, present_markers) / total_markers
        else:
            tense_consistency = 0.7  # Neutral if no clear markers
        
        # Check temporal sequence words
        sequence_words = ['then', 'next', 'after', 'before', 'suddenly', 'finally', 
                         'meanwhile', 'later', 'earlier', 'soon', 'eventually']
        sequence_count = sum(1 for w in sequence_words if w in generated_text.lower())
        sequence_score = min(1.0, sequence_count / 3)  # Expect 3+ for good flow
        
        # Check for temporal contradictions (simplified)
        contradiction_patterns = [
            (r'before.*after.*before', -0.2),
            (r'yesterday.*tomorrow.*yesterday', -0.2),
            (r'first.*last.*first', -0.1),
        ]
        
        penalty = 0
        for pattern, pen in contradiction_patterns:
            if re.search(pattern, generated_text.lower()):
                penalty += pen
        
        return max(0, min(1.0, tense_consistency * 0.5 + sequence_score * 0.3 + 0.2 + penalty))
    
    def calculate_voice_coherence(
        self,
        generated_text: str,
        context: str,
        previous_segments: List[str]
    ) -> float:
        """
        Calculate narrative voice consistency.
        
        Measures:
        1. Sentence length distribution similarity (style)
        2. POV consistency (pronoun usage)
        3. Vocabulary complexity (unique words ratio)
        """
        if not generated_text:
            return 0.0
            
        # Establish reference text
        reference_texts = previous_segments[-3:] if previous_segments else []
        # If no previous segments, use context or default to high coherence (self-consistent)
        if not reference_texts:
            return 1.0 
            
        ref_text = " ".join(reference_texts)
        
        # 1. Sentence Length Similarity
        gen_sentences = self._split_sentences(generated_text)
        ref_sentences = self._split_sentences(ref_text)
        
        if not gen_sentences or not ref_sentences:
            return 0.5
            
        gen_lens = [len(s.split()) for s in gen_sentences]
        ref_lens = [len(s.split()) for s in ref_sentences]
        
        avg_gen = np.mean(gen_lens) if gen_lens else 0
        avg_ref = np.mean(ref_lens) if ref_lens else 0
        
        # Normalized difference
        len_sim = max(0, 1 - abs(avg_gen - avg_ref) / max(avg_ref, 1))
        
        # 2. POV Consistency
        def get_pov_vector(text):
            tokens = self._tokenize(text.lower())
            first_person = sum(1 for t in tokens if t in {'i', 'me', 'my', 'mine', 'we', 'us', 'our'})
            third_person = sum(1 for t in tokens if t in {'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their'})
            total = first_person + third_person + 1
            return first_person / total
            
        gen_pov = get_pov_vector(generated_text)
        ref_pov = get_pov_vector(ref_text)
        
        pov_sim = 1.0 - abs(gen_pov - ref_pov)
        
        # 3. Vocabulary Complexity (Type-Token Ratio)
        def get_ttr(text):
            tokens = self._tokenize(text.lower())
            if not tokens: return 0
            return len(set(tokens)) / len(tokens)
            
        gen_ttr = get_ttr(generated_text)
        ref_ttr = get_ttr(ref_text)
        
        complexity_sim = 1.0 - min(abs(gen_ttr - ref_ttr) * 2, 1.0)
        
        return len_sim * 0.3 + pov_sim * 0.5 + complexity_sim * 0.2

    def compare_approaches(
        self,
        unified_text: str,
        graph_text: str,
        hybrid_text: str,
        context: str,
        **kwargs
    ) -> Dict[str, CoherenceBreakdown]:
        """
        Compare coherence across all three approaches.
        
        Returns dict with breakdowns for each approach.
        """
        return {
            "unified": self.analyze(unified_text, context, **kwargs),
            "graph": self.analyze(graph_text, context, **kwargs),
            "hybrid": self.analyze(hybrid_text, context, **kwargs)
        }
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b\w+\b', text)
    
    def _rouge_l_score(self, gen_tokens: List[str], ref_tokens: List[str]) -> float:
        """Calculate ROUGE-L score (LCS-based)."""
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        # Dynamic programming LCS
        m, n = len(gen_tokens), len(ref_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if gen_tokens[i-1] == ref_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        precision = lcs_length / m if m > 0 else 0
        recall = lcs_length / n if n > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
