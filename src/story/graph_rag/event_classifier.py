"""
Event Classification System

Classifies narrative events into structural types:
- Setup
- Complication
- Reversal
- Climax
- Resolution
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class EventType(Enum):
    SETUP = "setup"
    COMPLICATION = "complication"
    REVERSAL = "reversal"
    CLIMAX = "climax"
    RESOLUTION = "resolution"
    UNKNOWN = "unknown"

@dataclass
class EventClassification:
    event_id: str
    event_type: EventType
    confidence: float
    reasoning: List[str]

class EventClassifier:
    """
    Classifies events based on their narrative role using keywords,
    position in story, and graph context.
    """
    
    def __init__(self):
        self.keywords = {
            EventType.SETUP: {
                "introduced", "began", "started", "arrived", "met", "learned",
                "born", "established", "created"
            },
            EventType.COMPLICATION: {
                "attacked", "captured", "stole", "betrayed", "lost", "failed",
                "trapped", "ambushed", "threatened", "obstacle", "problem"
            },
            EventType.REVERSAL: {
                "discovered", "revealed", "realized", "unexpectedly", "suddenly",
                "changed", "turned", "twist", "secret"
            },
            EventType.CLIMAX: {
                "defeated", "killed", "destroyed", "won", "confronted", "battle",
                "final", "ultimate", "decisive", "showdown"
            },
            EventType.RESOLUTION: {
                "returned", "celebrated", "married", "peace", "ended", "concluded",
                "resolved", "home", "safe", "aftermath"
            }
        }
        
    def classify_event(self, event_data: Dict, total_chapters: int = 1) -> EventClassification:
        """
        Classify a single event.
        
        Args:
            event_data: Dictionary containing event details (description, action, chapter)
            total_chapters: Total number of chapters in the story (for relative timing)
            
        Returns:
            EventClassification object
        """
        description = event_data.get('description', '').lower()
        action = event_data.get('action', '').lower()
        chapter = event_data.get('chapter', 1)
        
        scores = {t: 0.0 for t in EventType if t != EventType.UNKNOWN}
        reasoning = []
        
        # 1. Keyword analysis
        for event_type, keywords in self.keywords.items():
            matches = [k for k in keywords if k in description or k == action]
            if matches:
                scores[event_type] += len(matches) * 0.5
                reasoning.append(f"Matched keywords for {event_type.value}: {matches}")
                
        # 2. Structural position analysis
        relative_pos = chapter / max(1, total_chapters)
        
        if relative_pos < 0.2:
            scores[EventType.SETUP] += 0.3
            reasoning.append("Early story position suggests Setup")
        elif 0.2 <= relative_pos < 0.8:
            scores[EventType.COMPLICATION] += 0.2
            scores[EventType.REVERSAL] += 0.1
        elif relative_pos >= 0.8:
            scores[EventType.CLIMAX] += 0.3
            scores[EventType.RESOLUTION] += 0.2
            reasoning.append("Late story position suggests Climax/Resolution")
            
        # Determine winner
        best_type = max(scores.items(), key=lambda x: x[1])
        
        if best_type[1] > 0.3:
            final_type = best_type[0]
            confidence = min(1.0, best_type[1])
        else:
            final_type = EventType.UNKNOWN
            confidence = 0.0
            
        return EventClassification(
            event_id=event_data.get('id', 'unknown'),
            event_type=final_type,
            confidence=confidence,
            reasoning=reasoning
        )
