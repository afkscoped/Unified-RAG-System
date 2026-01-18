"""
Adaptive Weight Manager

Dynamically adjusts semantic vs lexical search weights
based on query characteristics and user feedback.
"""

from collections import defaultdict
from typing import Tuple, List, Dict
import numpy as np
from loguru import logger


class AdaptiveWeightManager:
    """
    Manages hybrid search weights adaptively.
    
    Adjusts semantic vs lexical weights based on:
    - Query classification (keyword, conceptual, mixed)
    - Historical user feedback
    """
    
    QUERY_TYPES = ["keyword", "conceptual", "mixed"]
    
    # Keywords that suggest conceptual queries
    CONCEPTUAL_INDICATORS = [
        "explain", "what is", "how does", "why", "describe",
        "compare", "difference", "relationship", "concept",
        "theory", "principle", "understand", "meaning"
    ]
    
    # Base weights for each query type
    BASE_WEIGHTS = {
        "keyword": (0.3, 0.7),     # Favor lexical for short/specific
        "conceptual": (0.7, 0.3), # Favor semantic for abstract
        "mixed": (0.5, 0.5)       # Balanced
    }
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        min_feedback_samples: int = 5
    ):
        """
        Initialize weight manager.
        
        Args:
            learning_rate: Rate of weight adjustment (0-1)
            min_feedback_samples: Min samples before adjusting weights
        """
        self.learning_rate = learning_rate
        self.min_feedback_samples = min_feedback_samples
        
        # Feedback storage: query_type -> list of (rating, semantic_weight)
        self.feedback: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        
        # Current adjusted weights
        self.adjusted_weights = dict(self.BASE_WEIGHTS)
        
    def classify_query(self, query: str) -> str:
        """
        Classify query type based on characteristics.
        
        Args:
            query: The search query
            
        Returns:
            Query type: "keyword", "conceptual", or "mixed"
        """
        words = query.lower().split()
        word_count = len(words)
        
        # Short queries are likely keyword searches
        if word_count <= 3:
            return "keyword"
        
        # Check for conceptual indicators
        query_lower = query.lower()
        has_conceptual = any(
            indicator in query_lower 
            for indicator in self.CONCEPTUAL_INDICATORS
        )
        
        if has_conceptual:
            return "conceptual"
        
        # Long queries without clear indicators
        if word_count >= 8:
            return "conceptual"
            
        return "mixed"
    
    def get_weights(
        self,
        query: str,
        override_type: str = None
    ) -> Tuple[float, float]:
        """
        Get semantic and lexical weights for a query.
        
        Args:
            query: The search query
            override_type: Optional query type override
            
        Returns:
            Tuple of (semantic_weight, lexical_weight)
        """
        query_type = override_type or self.classify_query(query)
        
        weights = self.adjusted_weights.get(query_type, (0.5, 0.5))
        
        logger.debug(
            f"Query type: {query_type}, "
            f"weights: sem={weights[0]:.2f}, lex={weights[1]:.2f}"
        )
        
        return weights
    
    def record_feedback(
        self,
        query: str,
        rating: float,
        semantic_weight: float
    ):
        """
        Record user feedback for a query.
        
        Args:
            query: The original query
            rating: User rating (1-5)
            semantic_weight: The semantic weight used
        """
        query_type = self.classify_query(query)
        normalized_rating = rating / 5.0  # Normalize to 0-1
        
        self.feedback[query_type].append((normalized_rating, semantic_weight))
        
        # Adjust weights if enough samples
        if len(self.feedback[query_type]) >= self.min_feedback_samples:
            self._update_weights(query_type)
            
    def _update_weights(self, query_type: str):
        """Update weights based on feedback."""
        samples = self.feedback[query_type][-20:]  # Use last 20 samples
        
        if not samples:
            return
            
        # Analyze feedback
        avg_rating = np.mean([r for r, _ in samples])
        
        current_sem, current_lex = self.adjusted_weights[query_type]
        
        # If ratings are low, try swapping emphasis
        if avg_rating < 0.6:
            # Calculate correlation between semantic weight and rating
            high_sem_ratings = [r for r, w in samples if w > 0.5]
            low_sem_ratings = [r for r, w in samples if w <= 0.5]
            
            # Adjust based on which worked better
            if high_sem_ratings and low_sem_ratings:
                if np.mean(high_sem_ratings) > np.mean(low_sem_ratings):
                    new_sem = min(1.0, current_sem + self.learning_rate)
                else:
                    new_sem = max(0.0, current_sem - self.learning_rate)
            else:
                # Swap if consistently poor
                new_sem = current_lex
                
            new_lex = 1.0 - new_sem
            self.adjusted_weights[query_type] = (new_sem, new_lex)
            
            logger.info(
                f"Adjusted {query_type} weights: "
                f"sem={new_sem:.2f}, lex={new_lex:.2f} "
                f"(avg rating: {avg_rating:.2f})"
            )
    
    def reset(self):
        """Reset to base weights."""
        self.feedback.clear()
        self.adjusted_weights = dict(self.BASE_WEIGHTS)
        logger.info("Weight manager reset to defaults")
        
    def get_stats(self) -> Dict:
        """Get feedback statistics."""
        stats = {
            "current_weights": dict(self.adjusted_weights),
            "feedback_counts": {k: len(v) for k, v in self.feedback.items()},
            "avg_ratings": {}
        }
        
        for qtype, samples in self.feedback.items():
            if samples:
                stats["avg_ratings"][qtype] = np.mean([r for r, _ in samples])
                
        return stats

