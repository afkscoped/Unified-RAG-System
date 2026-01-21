"""
Trope Detector Module

Detects common narrative tropes and patterns in story content.
Provides thematic analysis to help writers understand their narrative structure.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import re


class TropeCategory(Enum):
    """Categories of narrative tropes."""
    CHARACTER_ARC = "character_arc"
    RELATIONSHIP = "relationship"
    PLOT_STRUCTURE = "plot_structure"
    THEME = "theme"
    CONFLICT = "conflict"


@dataclass
class DetectedTrope:
    """A detected narrative trope."""
    name: str
    category: TropeCategory
    confidence: float  # 0-1
    description: str
    evidence: List[str]  # Text snippets that support detection
    chapter_first_seen: int
    involved_entities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "category": self.category.value,
            "confidence": self.confidence,
            "description": self.description,
            "evidence": self.evidence[:3],  # Limit for display
            "chapter_first_seen": self.chapter_first_seen,
            "involved_entities": self.involved_entities
        }


class TropeDetector:
    """
    Detects narrative tropes and patterns in story content.
    
    Analyzes:
    - Character arc patterns (Hero's Journey, Fall from Grace, Redemption)
    - Relationship tropes (Love Triangle, Rivals to Lovers, Betrayal)
    - Plot structures (Three Act, Quest, Mystery)
    - Thematic elements (Good vs Evil, Coming of Age)
    """
    
    # Trope patterns with keywords and context
    TROPE_PATTERNS = {
        # Character Arcs
        "hero_journey": {
            "category": TropeCategory.CHARACTER_ARC,
            "description": "Classic Hero's Journey - ordinary person called to adventure",
            "keywords": ["call", "destiny", "chosen", "adventure", "quest", "reluctant", 
                        "mentor", "threshold", "transformation"],
            "anti_keywords": [],
            "min_matches": 3
        },
        "fall_from_grace": {
            "category": TropeCategory.CHARACTER_ARC,
            "description": "A once-noble character descends into moral corruption",
            "keywords": ["corrupted", "fallen", "betrayed", "power", "consumed", 
                        "darkness", "lost", "former", "once was"],
            "anti_keywords": ["redeemed", "saved"],
            "min_matches": 2
        },
        "redemption_arc": {
            "category": TropeCategory.CHARACTER_ARC,
            "description": "A flawed character seeks to make amends",
            "keywords": ["redemption", "forgiveness", "atone", "second chance", 
                        "make amends", "changed", "reformed", "regret"],
            "anti_keywords": [],
            "min_matches": 2
        },
        "coming_of_age": {
            "category": TropeCategory.CHARACTER_ARC,
            "description": "Young protagonist grows into maturity",
            "keywords": ["young", "grow", "learn", "innocent", "first time", 
                        "discover", "realize", "understand", "child", "youth"],
            "anti_keywords": [],
            "min_matches": 3
        },
        
        # Relationship Tropes
        "love_triangle": {
            "category": TropeCategory.RELATIONSHIP,
            "description": "Three characters entangled in romantic tension",
            "keywords": ["jealous", "choose", "both", "torn", "between", 
                        "rival", "heart", "love"],
            "anti_keywords": [],
            "min_matches": 2
        },
        "rivals_to_lovers": {
            "category": TropeCategory.RELATIONSHIP,
            "description": "Former rivals develop romantic feelings",
            "keywords": ["enemy", "rival", "hate", "respect", "understand", 
                        "different", "changed", "feeling"],
            "anti_keywords": [],
            "min_matches": 3
        },
        "betrayal": {
            "category": TropeCategory.RELATIONSHIP,
            "description": "A trusted ally reveals their true treacherous nature",
            "keywords": ["betrayed", "traitor", "trusted", "deceived", "secret", 
                        "revealed", "all along", "never"],
            "anti_keywords": [],
            "min_matches": 2
        },
        "found_family": {
            "category": TropeCategory.RELATIONSHIP,
            "description": "Unrelated characters form deep familial bonds",
            "keywords": ["family", "belong", "home", "together", "protect", 
                        "care", "bond", "like a"],
            "anti_keywords": ["blood", "real family"],
            "min_matches": 2
        },
        
        # Plot Structures
        "quest_narrative": {
            "category": TropeCategory.PLOT_STRUCTURE,
            "description": "Characters embark on a journey to achieve a goal",
            "keywords": ["find", "seek", "journey", "destination", "artifact", 
                        "ancient", "map", "travel", "reach"],
            "anti_keywords": [],
            "min_matches": 3
        },
        "mystery_reveal": {
            "category": TropeCategory.PLOT_STRUCTURE,
            "description": "Hidden truth gradually uncovered through investigation",
            "keywords": ["mystery", "clue", "discover", "hidden", "secret", 
                        "truth", "investigation", "revealed"],
            "anti_keywords": [],
            "min_matches": 3
        },
        "race_against_time": {
            "category": TropeCategory.PLOT_STRUCTURE,
            "description": "Characters must complete objective before deadline",
            "keywords": ["time", "deadline", "hurry", "before", "too late", 
                        "running out", "quickly", "must"],
            "anti_keywords": [],
            "min_matches": 2
        },
        
        # Conflict Tropes
        "good_vs_evil": {
            "category": TropeCategory.CONFLICT,
            "description": "Clear moral battle between virtuous and villainous forces",
            "keywords": ["evil", "dark", "light", "good", "pure", "corrupt", 
                        "save", "destroy", "hero", "villain"],
            "anti_keywords": ["grey", "complex"],
            "min_matches": 3
        },
        "internal_conflict": {
            "category": TropeCategory.CONFLICT,
            "description": "Character struggles with inner demons or choices",
            "keywords": ["torn", "doubt", "struggle", "inside", "heart", 
                        "mind", "choose", "conflict", "battle within"],
            "anti_keywords": [],
            "min_matches": 2
        },
        "underdog_victory": {
            "category": TropeCategory.CONFLICT,
            "description": "Weaker party overcomes seemingly insurmountable odds",
            "keywords": ["impossible", "outnumbered", "weak", "strong", "against all odds",
                        "underestimate", "surprise", "victory"],
            "anti_keywords": [],
            "min_matches": 2
        },
        
        # Thematic Elements
        "sacrifice_theme": {
            "category": TropeCategory.THEME,
            "description": "Characters make significant personal sacrifices",
            "keywords": ["sacrifice", "give up", "cost", "price", "worth it",
                        "for them", "in their place", "instead"],
            "anti_keywords": [],
            "min_matches": 2
        },
        "power_corrupts": {
            "category": TropeCategory.THEME,
            "description": "Acquisition of power leads to moral decay",
            "keywords": ["power", "corrupt", "control", "rule", "domination",
                        "throne", "crown", "absolute"],
            "anti_keywords": [],
            "min_matches": 2
        },
    }
    
    def __init__(self, story_graph=None, arc_tracker=None):
        """
        Initialize trope detector.
        
        Args:
            story_graph: StoryKnowledgeGraph for entity analysis
            arc_tracker: DynamicArcTracker for character arc analysis
        """
        self.story_graph = story_graph
        self.arc_tracker = arc_tracker
        self.detected_tropes: List[DetectedTrope] = []
        self.story_segments: List[Tuple[int, str]] = []  # (chapter, text)
        logger.info("TropeDetector initialized")
    
    def analyze_segment(
        self,
        text: str,
        chapter: int,
        entities: Optional[List[Dict]] = None
    ) -> List[DetectedTrope]:
        """
        Analyze a story segment for tropes.
        
        Args:
            text: Story text to analyze
            chapter: Current chapter number
            entities: Known entities for context
            
        Returns:
            List of newly detected tropes
        """
        self.story_segments.append((chapter, text))
        entities = entities or []
        character_names = [e.get('name', '') for e in entities 
                         if e.get('type') == 'CHARACTER']
        
        new_tropes = []
        text_lower = text.lower()
        
        for trope_id, pattern in self.TROPE_PATTERNS.items():
            # Skip if already detected
            if any(t.name == trope_id for t in self.detected_tropes):
                continue
            
            # Count keyword matches
            matches = []
            for keyword in pattern["keywords"]:
                if keyword in text_lower:
                    # Find sentence containing keyword
                    for sentence in text.split('.'):
                        if keyword in sentence.lower():
                            matches.append(sentence.strip()[:100])
                            break
            
            # Check anti-keywords (disqualifiers)
            has_anti = any(anti in text_lower for anti in pattern.get("anti_keywords", []))
            
            if len(matches) >= pattern["min_matches"] and not has_anti:
                # Calculate confidence
                confidence = min(1.0, len(matches) / (pattern["min_matches"] + 2))
                
                # Find involved characters
                involved = [name for name in character_names if name.lower() in text_lower]
                
                trope = DetectedTrope(
                    name=trope_id,
                    category=pattern["category"],
                    confidence=confidence,
                    description=pattern["description"],
                    evidence=matches[:3],
                    chapter_first_seen=chapter,
                    involved_entities=involved[:3]
                )
                
                new_tropes.append(trope)
                self.detected_tropes.append(trope)
                logger.info(f"Detected trope: {trope_id} (confidence: {confidence:.2f})")
        
        return new_tropes
    
    def get_all_tropes(self) -> List[DetectedTrope]:
        """Get all detected tropes."""
        return sorted(self.detected_tropes, 
                     key=lambda t: t.confidence, reverse=True)
    
    def get_tropes_by_category(self, category: TropeCategory) -> List[DetectedTrope]:
        """Get tropes filtered by category."""
        return [t for t in self.detected_tropes if t.category == category]
    
    def get_thematic_summary(self) -> Dict:
        """
        Generate a thematic summary of the story.
        
        Returns:
            Dictionary with category counts, dominant themes, and suggestions
        """
        if not self.detected_tropes:
            return {
                "status": "insufficient_data",
                "message": "Generate more story content to detect themes",
                "categories": {},
                "dominant_theme": None,
                "narrative_style": "unknown"
            }
        
        # Count by category
        category_counts = {}
        for trope in self.detected_tropes:
            cat = trope.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Find dominant category
        dominant_cat = max(category_counts, key=category_counts.get)
        
        # Determine narrative style
        narrative_style = self._infer_narrative_style()
        
        # Get top tropes
        top_tropes = sorted(self.detected_tropes, 
                           key=lambda t: t.confidence, reverse=True)[:5]
        
        return {
            "status": "analyzed",
            "total_tropes": len(self.detected_tropes),
            "categories": category_counts,
            "dominant_theme": dominant_cat,
            "narrative_style": narrative_style,
            "top_tropes": [t.to_dict() for t in top_tropes],
            "suggestions": self._generate_suggestions()
        }
    
    def _infer_narrative_style(self) -> str:
        """Infer the overall narrative style from detected tropes."""
        trope_names = [t.name for t in self.detected_tropes]
        
        if "hero_journey" in trope_names or "quest_narrative" in trope_names:
            return "Epic Adventure"
        elif "mystery_reveal" in trope_names:
            return "Mystery/Thriller"
        elif any(t in trope_names for t in ["love_triangle", "rivals_to_lovers"]):
            return "Romance/Drama"
        elif "good_vs_evil" in trope_names:
            return "Classic Fantasy"
        elif "internal_conflict" in trope_names:
            return "Character Study"
        elif "coming_of_age" in trope_names:
            return "Coming-of-Age"
        else:
            return "Mixed Genre"
    
    def _generate_suggestions(self) -> List[str]:
        """Generate suggestions based on detected patterns."""
        suggestions = []
        trope_names = [t.name for t in self.detected_tropes]
        
        # Suggest complementary tropes
        if "hero_journey" in trope_names and "sacrifice_theme" not in trope_names:
            suggestions.append("Consider adding a sacrifice moment to deepen the hero's journey")
        
        if "betrayal" in trope_names and "redemption_arc" not in trope_names:
            suggestions.append("The betrayer could have a redemption arc for added complexity")
        
        if "good_vs_evil" in trope_names:
            suggestions.append("Add moral grey areas to make the conflict more nuanced")
        
        if not any(t.category == TropeCategory.RELATIONSHIP for t in self.detected_tropes):
            suggestions.append("Consider developing character relationships for emotional depth")
        
        if len(self.detected_tropes) < 3:
            suggestions.append("Continue developing the narrative to reveal more thematic patterns")
        
        return suggestions[:3]
    
    def clear(self) -> None:
        """Clear all detected tropes."""
        self.detected_tropes = []
        self.story_segments = []
