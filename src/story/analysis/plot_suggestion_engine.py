"""
Plot Suggestion Engine

Generates smart, actionable plot suggestions based on graph analysis.
Detects narrative opportunities and problems.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class SuggestionPriority(Enum):
    """Priority levels for plot suggestions."""
    URGENT = "urgent"       # Should address soon
    HIGH = "high"           # Important for story quality
    MEDIUM = "medium"       # Good opportunity
    LOW = "low"            # Optional enhancement


class SuggestionType(Enum):
    """Types of plot suggestions."""
    UNRESOLVED_CONFLICT = "unresolved_conflict"
    CHARACTER_ARC_GAP = "character_arc_gap"
    CHEKHOV_GUN = "chekhov_gun"
    UNDERUTILIZED_LOCATION = "underutilized_location"
    RELATIONSHIP_OPPORTUNITY = "relationship_opportunity"
    FORESHADOWING = "foreshadowing"
    TENSION_ESCALATION = "tension_escalation"


@dataclass
class PlotSuggestion:
    """A single plot suggestion."""
    type: SuggestionType
    priority: SuggestionPriority
    title: str
    description: str
    actionable_prompt: str  # Ready-to-use prompt
    involved_entities: List[str] = field(default_factory=list)
    chapter_target: Optional[int] = None
    
    # Scoring breakdown
    urgency_score: float = 0.5
    impact_score: float = 0.5
    complexity_score: float = 0.5
    
    def to_dict(self) -> Dict:
        return {
            "type": self.type.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "actionable_prompt": self.actionable_prompt,
            "involved_entities": self.involved_entities,
            "chapter_target": self.chapter_target,
            "scores": {
                "urgency": self.urgency_score,
                "impact": self.impact_score,
                "complexity": self.complexity_score
            }
        }


class PlotSuggestionEngine:
    """
    Generates actionable plot suggestions from story graph analysis.
    
    Analyzes:
    - Unresolved conflicts
    - Character development gaps
    - Chekhov's gun violations
    - Underutilized elements
    - Relationship opportunities
    """
    
    def __init__(self, story_graph=None, arc_tracker=None):
        """
        Initialize plot suggestion engine.
        
        Args:
            story_graph: StoryKnowledgeGraph for relationship analysis
            arc_tracker: DynamicArcTracker for character arc analysis
        """
        self.story_graph = story_graph
        self.arc_tracker = arc_tracker
        
        # Track which suggestions have been used
        self.used_suggestions: List[str] = []
        
        logger.info("PlotSuggestionEngine initialized")
    
    def generate_suggestions(
        self,
        current_chapter: int,
        limit: int = 5
    ) -> List[PlotSuggestion]:
        """
        Generate prioritized plot suggestions for the current chapter.
        
        Args:
            current_chapter: Current chapter number
            limit: Maximum number of suggestions to return
            
        Returns:
            List of prioritized PlotSuggestions
        """
        suggestions = []
        
        # Detect various opportunities
        suggestions.extend(self._detect_unresolved_conflicts(current_chapter))
        suggestions.extend(self._detect_character_arc_gaps(current_chapter))
        suggestions.extend(self._detect_chekhov_violations(current_chapter))
        suggestions.extend(self._detect_underutilized_locations(current_chapter))
        suggestions.extend(self._detect_relationship_opportunities(current_chapter))
        
        # Score and prioritize
        scored = self._score_suggestions(suggestions, current_chapter)
        
        # Sort by priority and score
        priority_order = {
            SuggestionPriority.URGENT: 0,
            SuggestionPriority.HIGH: 1,
            SuggestionPriority.MEDIUM: 2,
            SuggestionPriority.LOW: 3
        }
        
        scored.sort(key=lambda s: (
            priority_order.get(s.priority, 4),
            -s.urgency_score,
            -s.impact_score
        ))
        
        # Filter out already used
        fresh = [s for s in scored if s.title not in self.used_suggestions]
        
        return fresh[:limit]
    
    def mark_used(self, suggestion_title: str) -> None:
        """Mark a suggestion as used."""
        self.used_suggestions.append(suggestion_title)
    
    def _detect_unresolved_conflicts(self, chapter: int) -> List[PlotSuggestion]:
        """Detect unresolved conflicts from the graph."""
        suggestions = []
        
        if not self.story_graph:
            return suggestions
        
        # Find CONFLICTS_WITH edges without resolution
        for edge in self.story_graph.graph.edges(data=True):
            if edge[2].get('relation_type') == 'CONFLICTS_WITH':
                source_id, target_id = edge[0], edge[1]
                
                source = self.story_graph.get_entity(source_id)
                target = self.story_graph.get_entity(target_id)
                
                if not source or not target:
                    continue
                
                # Check if conflict is old enough to need resolution
                conflict_chapter = edge[2].get('temporal_context', 1)
                chapters_since = chapter - conflict_chapter
                
                if chapters_since >= 2:
                    urgency = min(1.0, chapters_since / 5)
                    
                    suggestions.append(PlotSuggestion(
                        type=SuggestionType.UNRESOLVED_CONFLICT,
                        priority=SuggestionPriority.HIGH if chapters_since >= 3 else SuggestionPriority.MEDIUM,
                        title=f"Resolve {source.name} vs {target.name} conflict",
                        description=f"The conflict between {source.name} and {target.name} has been "
                                   f"ongoing for {chapters_since} chapters without resolution.",
                        actionable_prompt=f"{source.name} and {target.name} finally confront each other "
                                         f"about their ongoing conflict. The tension reaches a breaking point "
                                         f"when [describe triggering event]. This confrontation will "
                                         f"[resolve/escalate] their relationship.",
                        involved_entities=[source.name, target.name],
                        urgency_score=urgency,
                        impact_score=0.8
                    ))
        
        return suggestions
    
    def _detect_character_arc_gaps(self, chapter: int) -> List[PlotSuggestion]:
        """Detect characters who need development."""
        suggestions = []
        
        if not self.arc_tracker:
            return suggestions
        
        for char_id, timeline in self.arc_tracker.character_timelines.items():
            if not timeline:
                continue
            
            char_name = char_id.replace('char_', '').replace('_', ' ').title()
            latest = timeline[-1]
            
            # Check if character hasn't had recent development
            latest_chapter = latest.chapter
            chapters_since = chapter - latest_chapter
            
            if chapters_since >= 2:
                # Character needs attention
                predicted = self.arc_tracker.predict_next_arc_phase(char_id)
                
                suggestions.append(PlotSuggestion(
                    type=SuggestionType.CHARACTER_ARC_GAP,
                    priority=SuggestionPriority.HIGH if chapters_since >= 4 else SuggestionPriority.MEDIUM,
                    title=f"Develop {char_name}'s character arc",
                    description=f"{char_name} hasn't had significant development in {chapters_since} "
                               f"chapters. Current emotional state: {latest.emotional_state}. "
                               f"Predicted next phase: {predicted}",
                    actionable_prompt=f"{char_name}, feeling {latest.emotional_state}, faces a moment "
                                     f"of personal challenge that forces them to grow. This could involve "
                                     f"[making a difficult choice / confronting their fear / learning a truth]. "
                                     f"Their arc is moving toward {predicted}.",
                    involved_entities=[char_name],
                    urgency_score=min(1.0, chapters_since / 5),
                    impact_score=0.7
                ))
            
            # Check for stagnant emotional state
            if len(timeline) >= 3:
                recent_states = [s.emotional_state for s in timeline[-3:]]
                if len(set(recent_states)) == 1:
                    suggestions.append(PlotSuggestion(
                        type=SuggestionType.CHARACTER_ARC_GAP,
                        priority=SuggestionPriority.MEDIUM,
                        title=f"Shift {char_name}'s emotional state",
                        description=f"{char_name} has been {recent_states[0]} for too long. "
                                   f"Time for emotional development.",
                        actionable_prompt=f"Something happens that challenges {char_name}'s current "
                                         f"{recent_states[0]} mindset. This event forces them to experience "
                                         f"a new emotion and grow as a character.",
                        involved_entities=[char_name],
                        urgency_score=0.5,
                        impact_score=0.6
                    ))
        
        return suggestions
    
    def _detect_chekhov_violations(self, chapter: int) -> List[PlotSuggestion]:
        """Detect introduced elements that haven't been used."""
        suggestions = []
        
        if not self.story_graph:
            return suggestions
        
        # Find artifacts/items that were introduced but not connected to events
        for node_id in self.story_graph.graph.nodes():
            entity = self.story_graph.get_entity(node_id)
            if not entity:
                continue
            
            if entity.type in ['ARTIFACT', 'ITEM', 'OBJECT']:
                # Check if this item has been involved in any events
                connections = list(self.story_graph.graph.edges(node_id))
                event_connections = [c for c in connections 
                                    if self.story_graph.get_entity(c[1]) 
                                    and self.story_graph.get_entity(c[1]).type == 'EVENT']
                
                chapters_since = chapter - entity.first_appearance
                
                if not event_connections and chapters_since >= 2:
                    suggestions.append(PlotSuggestion(
                        type=SuggestionType.CHEKHOV_GUN,
                        priority=SuggestionPriority.HIGH,
                        title=f"Use the {entity.name}",
                        description=f"The {entity.name} was introduced in chapter {entity.first_appearance} "
                                   f"but hasn't been used yet. Chekhov's principle suggests it should matter.",
                        actionable_prompt=f"The {entity.name}, introduced earlier, finally becomes important "
                                         f"when [character] discovers its true purpose. It holds the key to "
                                         f"[solving a problem / revealing a secret / changing the situation].",
                        involved_entities=[entity.name],
                        urgency_score=min(1.0, chapters_since / 4),
                        impact_score=0.8
                    ))
        
        return suggestions
    
    def _detect_underutilized_locations(self, chapter: int) -> List[PlotSuggestion]:
        """Detect locations that haven't been revisited."""
        suggestions = []
        
        if not self.story_graph:
            return suggestions
        
        for node_id in self.story_graph.graph.nodes():
            entity = self.story_graph.get_entity(node_id)
            if not entity or entity.type != 'LOCATION':
                continue
            
            # Check when location was last used
            last_used = entity.first_appearance
            for edge in self.story_graph.graph.edges(node_id, data=True):
                edge_chapter = edge[2].get('temporal_context', 1)
                last_used = max(last_used, edge_chapter)
            
            chapters_since = chapter - last_used
            
            if chapters_since >= 3:
                suggestions.append(PlotSuggestion(
                    type=SuggestionType.UNDERUTILIZED_LOCATION,
                    priority=SuggestionPriority.LOW,
                    title=f"Return to {entity.name}",
                    description=f"{entity.name} was established but hasn't appeared recently. "
                               f"Consider bringing it back for continuity.",
                    actionable_prompt=f"The characters return to {entity.name}, finding it changed "
                                     f"since their last visit. [describe what's different]. This location "
                                     f"now holds [new significance / hidden secrets / unexpected occupants].",
                    involved_entities=[entity.name],
                    urgency_score=0.3,
                    impact_score=0.4
                ))
        
        return suggestions
    
    def _detect_relationship_opportunities(self, chapter: int) -> List[PlotSuggestion]:
        """Detect opportunities for relationship development."""
        suggestions = []
        
        if not self.story_graph:
            return suggestions
        
        # Find character pairs with weak or no connection
        characters = [n for n in self.story_graph.graph.nodes()
                     if self.story_graph.get_entity(n) 
                     and self.story_graph.get_entity(n).type == 'CHARACTER']
        
        for i, char_a in enumerate(characters):
            for char_b in characters[i+1:]:
                # Check if they have a relationship
                has_edge = (self.story_graph.graph.has_edge(char_a, char_b) or 
                           self.story_graph.graph.has_edge(char_b, char_a))
                
                entity_a = self.story_graph.get_entity(char_a)
                entity_b = self.story_graph.get_entity(char_b)
                
                if not entity_a or not entity_b:
                    continue
                
                if not has_edge:
                    # No relationship - opportunity for one
                    if (chapter - entity_a.first_appearance >= 1 and 
                        chapter - entity_b.first_appearance >= 1):
                        suggestions.append(PlotSuggestion(
                            type=SuggestionType.RELATIONSHIP_OPPORTUNITY,
                            priority=SuggestionPriority.MEDIUM,
                            title=f"Connect {entity_a.name} and {entity_b.name}",
                            description=f"{entity_a.name} and {entity_b.name} haven't interacted yet. "
                                       f"Their meeting could create new dynamics.",
                            actionable_prompt=f"{entity_a.name} and {entity_b.name} meet for the first time. "
                                             f"Their interaction reveals [shared goals / conflicting interests / "
                                             f"unexpected connection]. This encounter will [change the story].",
                            involved_entities=[entity_a.name, entity_b.name],
                            urgency_score=0.4,
                            impact_score=0.6
                        ))
        
        return suggestions
    
    def _score_suggestions(
        self,
        suggestions: List[PlotSuggestion],
        chapter: int
    ) -> List[PlotSuggestion]:
        """Score and prioritize suggestions."""
        for s in suggestions:
            # Adjust priority based on scores
            avg_score = (s.urgency_score + s.impact_score) / 2
            
            if avg_score >= 0.8:
                s.priority = SuggestionPriority.URGENT
            elif avg_score >= 0.6:
                s.priority = SuggestionPriority.HIGH
            elif avg_score >= 0.4:
                s.priority = SuggestionPriority.MEDIUM
            else:
                s.priority = SuggestionPriority.LOW
            
            # Complexity based on involved entities
            s.complexity_score = min(1.0, len(s.involved_entities) * 0.3)
        
        return suggestions
