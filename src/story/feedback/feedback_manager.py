"""
Story Feedback Manager

Multi-dimensional feedback system for story generation with:
- Per-dimension ratings (quality, consistency, creativity, etc.)
- Approach-specific feedback
- Adaptive weight adjustment
- Performance trend tracking
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from loguru import logger


@dataclass
class StoryFeedback:
    """Single feedback entry."""
    timestamp: str
    chapter: int
    approach: str  # 'unified', 'graph', 'hybrid'
    prompt: str
    
    # Multi-dimensional ratings (1-5)
    overall: int = 3
    consistency: int = 3
    creativity: int = 3
    character_authenticity: int = 3
    plot_coherence: int = 3
    
    # Optional text feedback
    comment: str = ""
    
    # Which approach was selected as best
    selected_best: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "chapter": self.chapter,
            "approach": self.approach,
            "prompt": self.prompt[:200],
            "ratings": {
                "overall": self.overall,
                "consistency": self.consistency,
                "creativity": self.creativity,
                "character_authenticity": self.character_authenticity,
                "plot_coherence": self.plot_coherence
            },
            "comment": self.comment,
            "selected_best": self.selected_best
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "StoryFeedback":
        ratings = data.get("ratings", {})
        return cls(
            timestamp=data.get("timestamp", ""),
            chapter=data.get("chapter", 1),
            approach=data.get("approach", "hybrid"),
            prompt=data.get("prompt", ""),
            overall=ratings.get("overall", 3),
            consistency=ratings.get("consistency", 3),
            creativity=ratings.get("creativity", 3),
            character_authenticity=ratings.get("character_authenticity", 3),
            plot_coherence=ratings.get("plot_coherence", 3),
            comment=data.get("comment", ""),
            selected_best=data.get("selected_best", False)
        )
    
    def get_average_rating(self) -> float:
        """Calculate average across all dimensions."""
        return (self.overall + self.consistency + self.creativity + 
                self.character_authenticity + self.plot_coherence) / 5


@dataclass
class ApproachWeights:
    """Weights for each approach based on feedback."""
    unified: float = 0.33
    graph: float = 0.33
    hybrid: float = 0.34
    
    def to_dict(self) -> Dict:
        return {
            "unified": self.unified,
            "graph": self.graph,
            "hybrid": self.hybrid
        }
    
    def normalize(self) -> None:
        """Ensure weights sum to 1.0."""
        total = self.unified + self.graph + self.hybrid
        if total > 0:
            self.unified /= total
            self.graph /= total
            self.hybrid /= total


class StoryFeedbackManager:
    """
    Manages story generation feedback and adaptive improvements.
    
    Features:
    - Multi-dimensional rating collection
    - Approach comparison tracking
    - Adaptive weight learning
    - Performance analytics
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize feedback manager.
        
        Args:
            storage_path: Path to feedback JSON file
        """
        self.storage_path = storage_path or "data/feedback.json"
        self.feedback_history: List[StoryFeedback] = []
        self.approach_weights = ApproachWeights()
        
        # Load existing feedback
        self._load_feedback()
        
        logger.info(f"StoryFeedbackManager initialized with {len(self.feedback_history)} entries")
    
    def submit_feedback(
        self,
        chapter: int,
        approach: str,
        prompt: str,
        overall: int = 3,
        consistency: int = 3,
        creativity: int = 3,
        character_authenticity: int = 3,
        plot_coherence: int = 3,
        comment: str = "",
        selected_best: bool = False
    ) -> StoryFeedback:
        """
        Submit feedback for a generation.
        
        Args:
            chapter: Chapter number
            approach: 'unified', 'graph', or 'hybrid'
            prompt: The generation prompt
            overall: Overall quality (1-5)
            consistency: Narrative consistency (1-5)
            creativity: Creative quality (1-5)
            character_authenticity: Character believability (1-5)
            plot_coherence: Plot logic (1-5)
            comment: Optional text feedback
            selected_best: Whether this was selected as best approach
            
        Returns:
            The created StoryFeedback entry
        """
        feedback = StoryFeedback(
            timestamp=datetime.now().isoformat(),
            chapter=chapter,
            approach=approach,
            prompt=prompt,
            overall=max(1, min(5, overall)),
            consistency=max(1, min(5, consistency)),
            creativity=max(1, min(5, creativity)),
            character_authenticity=max(1, min(5, character_authenticity)),
            plot_coherence=max(1, min(5, plot_coherence)),
            comment=comment,
            selected_best=selected_best
        )
        
        self.feedback_history.append(feedback)
        
        # Update adaptive weights
        self._update_weights(feedback)
        
        # Save to disk
        self._save_feedback()
        
        logger.info(f"Feedback submitted for {approach} (chapter {chapter}): avg={feedback.get_average_rating():.1f}")
        
        return feedback
    
    def get_recommended_approach(self) -> str:
        """Get recommended approach based on feedback history."""
        if not self.feedback_history:
            return "hybrid"  # Default
        
        # Find approach with highest weight
        weights = self.approach_weights
        if weights.unified >= weights.graph and weights.unified >= weights.hybrid:
            return "unified"
        elif weights.graph >= weights.hybrid:
            return "graph"
        else:
            return "hybrid"
    
    def get_performance_by_approach(self) -> Dict[str, Dict]:
        """Get average performance for each approach."""
        stats = {
            "unified": {"count": 0, "ratings": []},
            "graph": {"count": 0, "ratings": []},
            "hybrid": {"count": 0, "ratings": []}
        }
        
        for fb in self.feedback_history:
            if fb.approach in stats:
                stats[fb.approach]["count"] += 1
                stats[fb.approach]["ratings"].append(fb.get_average_rating())
        
        result = {}
        for approach, data in stats.items():
            if data["ratings"]:
                result[approach] = {
                    "count": data["count"],
                    "average_rating": sum(data["ratings"]) / len(data["ratings"]),
                    "best_selections": sum(1 for fb in self.feedback_history 
                                          if fb.approach == approach and fb.selected_best)
                }
            else:
                result[approach] = {"count": 0, "average_rating": 0, "best_selections": 0}
        
        return result
    
    def get_dimension_analysis(self) -> Dict[str, Dict]:
        """Analyze which dimensions each approach excels at."""
        dimensions = ["overall", "consistency", "creativity", 
                     "character_authenticity", "plot_coherence"]
        
        result = {}
        for dim in dimensions:
            result[dim] = {}
            for approach in ["unified", "graph", "hybrid"]:
                ratings = [getattr(fb, dim) for fb in self.feedback_history 
                          if fb.approach == approach]
                result[dim][approach] = sum(ratings) / len(ratings) if ratings else 0
        
        return result
    
    def get_performance_trends(self, last_n: int = 10) -> Dict[str, List[float]]:
        """Get performance trends over recent generations."""
        trends = {
            "unified": [],
            "graph": [],
            "hybrid": []
        }
        
        recent = self.feedback_history[-last_n:] if len(self.feedback_history) > last_n else self.feedback_history
        
        for fb in recent:
            if fb.approach in trends:
                trends[fb.approach].append(fb.get_average_rating())
        
        return trends
    
    def get_chapter_performance(self) -> Dict[int, Dict]:
        """Get average performance by chapter."""
        by_chapter = {}
        
        for fb in self.feedback_history:
            if fb.chapter not in by_chapter:
                by_chapter[fb.chapter] = {"ratings": [], "count": 0}
            by_chapter[fb.chapter]["ratings"].append(fb.get_average_rating())
            by_chapter[fb.chapter]["count"] += 1
        
        result = {}
        for chapter, data in by_chapter.items():
            result[chapter] = {
                "average": sum(data["ratings"]) / len(data["ratings"]),
                "count": data["count"]
            }
        
        return result
    
    def suggest_improvements(self) -> List[Dict]:
        """Suggest improvements based on feedback patterns."""
        suggestions = []
        
        dim_analysis = self.get_dimension_analysis()
        
        # Find weakest dimensions
        for dim, scores in dim_analysis.items():
            if all(v > 0 for v in scores.values()):
                avg = sum(scores.values()) / len(scores)
                if avg < 3.0:
                    suggestions.append({
                        "type": "weak_dimension",
                        "dimension": dim,
                        "average_score": avg,
                        "suggestion": f"Focus on improving {dim.replace('_', ' ')}"
                    })
        
        # Compare approaches
        perf = self.get_performance_by_approach()
        best_approach = max(perf.keys(), key=lambda k: perf[k].get("average_rating", 0))
        if perf[best_approach]["average_rating"] > 0:
            suggestions.append({
                "type": "approach_recommendation",
                "approach": best_approach,
                "reason": f"{best_approach.title()} has highest average rating ({perf[best_approach]['average_rating']:.1f})"
            })
        
        return suggestions
    
    def get_weights(self) -> ApproachWeights:
        """Get current approach weights."""
        return self.approach_weights
    
    def _update_weights(self, feedback: StoryFeedback) -> None:
        """Update approach weights based on new feedback."""
        # Learning rate
        lr = 0.1
        
        # Calculate reward based on rating
        rating = feedback.get_average_rating()
        reward = (rating - 3) / 2  # -1 to 1
        
        # Update weight for this approach
        if feedback.approach == "unified":
            self.approach_weights.unified += lr * reward
        elif feedback.approach == "graph":
            self.approach_weights.graph += lr * reward
        else:
            self.approach_weights.hybrid += lr * reward
        
        # Bonus for selected_best
        if feedback.selected_best:
            if feedback.approach == "unified":
                self.approach_weights.unified += 0.05
            elif feedback.approach == "graph":
                self.approach_weights.graph += 0.05
            else:
                self.approach_weights.hybrid += 0.05
        
        # Ensure min weight
        self.approach_weights.unified = max(0.1, self.approach_weights.unified)
        self.approach_weights.graph = max(0.1, self.approach_weights.graph)
        self.approach_weights.hybrid = max(0.1, self.approach_weights.hybrid)
        
        # Normalize
        self.approach_weights.normalize()
    
    def _load_feedback(self) -> None:
        """Load feedback from disk."""
        path = Path(self.storage_path)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                self.feedback_history = [StoryFeedback.from_dict(d) for d in data.get("feedback", [])]
                
                weights = data.get("weights", {})
                self.approach_weights.unified = weights.get("unified", 0.33)
                self.approach_weights.graph = weights.get("graph", 0.33)
                self.approach_weights.hybrid = weights.get("hybrid", 0.34)
                
            except Exception as e:
                logger.warning(f"Failed to load feedback: {e}")
    
    def _save_feedback(self) -> None:
        """Save feedback to disk."""
        path = Path(self.storage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = {
                "feedback": [fb.to_dict() for fb in self.feedback_history],
                "weights": self.approach_weights.to_dict(),
                "last_updated": datetime.now().isoformat()
            }
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
    
    def clear_history(self) -> None:
        """Clear all feedback history."""
        self.feedback_history = []
        self.approach_weights = ApproachWeights()
        self._save_feedback()
        logger.info("Feedback history cleared")
