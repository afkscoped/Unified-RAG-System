"""
Dynamic Character Arc Tracker

Tracks character evolution across narrative chapters including:
- Emotional state changes
- Goal progression
- Relationship evolution
- Arc phase detection (setup, rising action, crisis, resolution)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger


@dataclass
class CharacterState:
    """Represents character state at a specific point in the story."""
    chapter: int
    timestamp: datetime = field(default_factory=datetime.now)
    emotional_state: str = "neutral"  # hopeful, desperate, conflicted, etc.
    goals: List[str] = field(default_factory=list)
    relationships: Dict[str, float] = field(default_factory=dict)  # char_id -> strength (-1 to 1)
    location: str = "unknown"
    knowledge: List[str] = field(default_factory=list)  # What they know at this point
    arc_phase: str = "setup"  # setup, rising_action, crisis, climax, resolution
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "chapter": self.chapter,
            "timestamp": self.timestamp.isoformat(),
            "emotional_state": self.emotional_state,
            "goals": self.goals,
            "relationships": self.relationships,
            "location": self.location,
            "knowledge": self.knowledge,
            "arc_phase": self.arc_phase
        }


@dataclass
class ArcTransition:
    """Represents a significant change in character arc."""
    from_chapter: int
    to_chapter: int
    trigger_event: str
    state_before: CharacterState
    state_after: CharacterState
    transformation_type: str  # growth, fall, revelation, reversal
    magnitude: float  # 0-1, how significant the change
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "from_chapter": self.from_chapter,
            "to_chapter": self.to_chapter,
            "trigger_event": self.trigger_event,
            "transformation_type": self.transformation_type,
            "magnitude": self.magnitude,
            "state_before": self.state_before.to_dict(),
            "state_after": self.state_after.to_dict()
        }


class DynamicArcTracker:
    """
    Tracks character evolution across the narrative.
    
    Automatically detects arc transitions and maintains
    timeline of character states for consistency checks.
    """
    
    def __init__(self, story_graph=None):
        """
        Initialize arc tracker.
        
        Args:
            story_graph: Optional StoryKnowledgeGraph for relationship updates
        """
        self.story_graph = story_graph
        self.character_timelines: Dict[str, List[CharacterState]] = {}
        self.arc_transitions: Dict[str, List[ArcTransition]] = {}
        
        # Transition detection threshold
        self.transition_threshold = 0.4
        
        logger.info("DynamicArcTracker initialized")
        
    def record_character_state(
        self, 
        character_id: str, 
        state: CharacterState
    ) -> Optional[ArcTransition]:
        """
        Record character state at current chapter.
        
        Args:
            character_id: ID of the character
            state: Current CharacterState
            
        Returns:
            ArcTransition if significant change detected, else None
        """
        if character_id not in self.character_timelines:
            self.character_timelines[character_id] = []
        
        self.character_timelines[character_id].append(state)
        
        # Detect transitions if we have previous state
        transition = None
        if len(self.character_timelines[character_id]) > 1:
            transition = self._detect_arc_transition(character_id)
        
        logger.debug(f"Recorded state for {character_id} at chapter {state.chapter}")
        return transition
    
    def _detect_arc_transition(self, character_id: str) -> Optional[ArcTransition]:
        """Automatically detect significant character changes."""
        timeline = self.character_timelines[character_id]
        prev_state = timeline[-2]
        curr_state = timeline[-1]
        
        changes = []
        
        # 1. Emotional shift
        if prev_state.emotional_state != curr_state.emotional_state:
            emotional_weight = self._get_emotional_shift_weight(
                prev_state.emotional_state,
                curr_state.emotional_state
            )
            changes.append(("emotional", emotional_weight))
        
        # 2. Goal changes
        goals_lost = set(prev_state.goals) - set(curr_state.goals)
        goals_gained = set(curr_state.goals) - set(prev_state.goals)
        if goals_lost or goals_gained:
            goal_weight = min(0.7 + 0.1 * (len(goals_lost) + len(goals_gained)), 1.0)
            changes.append(("goals", goal_weight))
        
        # 3. Relationship shifts
        rel_delta = 0
        all_chars = set(prev_state.relationships.keys()) | set(curr_state.relationships.keys())
        for char in all_chars:
            prev_strength = prev_state.relationships.get(char, 0)
            curr_strength = curr_state.relationships.get(char, 0)
            rel_delta += abs(curr_strength - prev_strength)
        
        if rel_delta > 0.3:
            changes.append(("relationships", min(rel_delta, 1.0)))
        
        # 4. Knowledge gained (revelations)
        new_knowledge = set(curr_state.knowledge) - set(prev_state.knowledge)
        if new_knowledge:
            changes.append(("revelation", 0.8))
        
        # 5. Arc phase change
        if prev_state.arc_phase != curr_state.arc_phase:
            changes.append(("phase_change", 0.9))
        
        # Calculate overall magnitude
        if not changes:
            return None
            
        magnitude = sum(weight for _, weight in changes) / len(changes)
        
        # Only record significant transitions
        if magnitude < self.transition_threshold:
            return None
        
        transition = ArcTransition(
            from_chapter=prev_state.chapter,
            to_chapter=curr_state.chapter,
            trigger_event=self._infer_trigger_event(prev_state, curr_state),
            state_before=prev_state,
            state_after=curr_state,
            transformation_type=self._classify_transformation(changes),
            magnitude=magnitude
        )
        
        if character_id not in self.arc_transitions:
            self.arc_transitions[character_id] = []
        self.arc_transitions[character_id].append(transition)
        
        logger.info(
            f"Detected {transition.transformation_type} transition for {character_id} "
            f"(magnitude: {magnitude:.2f})"
        )
        
        return transition
    
    def _get_emotional_shift_weight(self, from_state: str, to_state: str) -> float:
        """Calculate weight of emotional state change."""
        # Major emotional shifts
        major_shifts = {
            ("hopeful", "desperate"): 0.9,
            ("desperate", "hopeful"): 0.9,
            ("neutral", "conflicted"): 0.6,
            ("conflicted", "resolved"): 0.8,
            ("angry", "forgiving"): 0.85,
            ("trusting", "betrayed"): 0.9,
        }
        
        return major_shifts.get((from_state, to_state), 0.5)
    
    def _classify_transformation(self, changes: List[tuple]) -> str:
        """Classify type of character transformation."""
        change_types = [c[0] for c in changes]
        
        if "revelation" in change_types:
            return "revelation"
        elif "phase_change" in change_types:
            return "arc_progression"
        elif "goals" in change_types and "emotional" in change_types:
            return "growth"
        elif "relationships" in change_types:
            if any(c[1] > 0.7 for c in changes if c[0] == "relationships"):
                return "reversal"
            return "relationship_shift"
        else:
            return "gradual_change"
    
    def _infer_trigger_event(
        self, 
        prev: CharacterState, 
        curr: CharacterState
    ) -> str:
        """Attempt to identify what caused the transition."""
        if not self.story_graph:
            return "unknown_catalyst"
        
        # Search for events between these chapters
        events = []
        for chapter in range(prev.chapter, curr.chapter + 1):
            if chapter in self.story_graph.chapter_timeline:
                events.extend(self.story_graph.chapter_timeline[chapter])
        
        if events:
            # Get the most recent event as likely trigger
            latest_event = events[-1]
            if latest_event in self.story_graph.graph:
                event_data = self.story_graph.graph.nodes[latest_event]
                return event_data.get('name', latest_event)
        
        return "unknown_catalyst"
    
    def get_character_arc_summary(self, character_id: str) -> Optional[Dict]:
        """Generate full character arc analysis."""
        if character_id not in self.character_timelines:
            return None
        
        timeline = self.character_timelines[character_id]
        transitions = self.arc_transitions.get(character_id, [])
        
        if not timeline:
            return None
        
        return {
            "character_id": character_id,
            "total_chapters": len(timeline),
            "current_state": timeline[-1].to_dict(),
            "initial_state": timeline[0].to_dict(),
            "key_transitions": [t.to_dict() for t in transitions],
            "arc_trajectory": self._calculate_arc_trajectory(timeline),
            "relationship_evolution": self._track_relationship_changes(timeline),
            "thematic_keywords": self._extract_arc_themes(timeline)
        }
    
    def _calculate_arc_trajectory(self, timeline: List[CharacterState]) -> str:
        """Determine overall arc pattern."""
        if len(timeline) < 3:
            return "early_development"
        
        # Compare initial and final states
        initial = timeline[0]
        final = timeline[-1]
        
        # Count positive changes
        positive_changes = 0
        negative_changes = 0
        
        # Goals
        if len(final.goals) > len(initial.goals):
            positive_changes += 1
        elif len(final.goals) < len(initial.goals):
            negative_changes += 1
        
        # Relationships
        initial_rel_sum = sum(initial.relationships.values())
        final_rel_sum = sum(final.relationships.values())
        if final_rel_sum > initial_rel_sum:
            positive_changes += 1
        elif final_rel_sum < initial_rel_sum:
            negative_changes += 1
        
        # Emotional state (simplified)
        positive_emotions = {"hopeful", "confident", "resolved", "happy", "determined"}
        if final.emotional_state in positive_emotions:
            positive_changes += 1
        if initial.emotional_state in positive_emotions and final.emotional_state not in positive_emotions:
            negative_changes += 1
        
        if positive_changes > negative_changes:
            return "ascending_arc"  # Hero's journey, redemption
        elif negative_changes > positive_changes:
            return "descending_arc"  # Tragedy, fall
        else:
            return "flat_arc"  # Character changes world, not themselves
    
    def _track_relationship_changes(
        self, 
        timeline: List[CharacterState]
    ) -> Dict[str, List[Dict]]:
        """Show how relationships evolved over time."""
        relationship_history = {}
        
        for state in timeline:
            for char, strength in state.relationships.items():
                if char not in relationship_history:
                    relationship_history[char] = []
                relationship_history[char].append({
                    "chapter": state.chapter,
                    "strength": strength
                })
        
        return relationship_history
    
    def _extract_arc_themes(self, timeline: List[CharacterState]) -> List[str]:
        """Identify recurring themes in character arc."""
        from collections import Counter
        
        all_goals = []
        all_emotions = []
        
        for state in timeline:
            all_goals.extend(state.goals)
            all_emotions.append(state.emotional_state)
        
        # Find most common elements
        goal_themes = Counter(all_goals).most_common(3)
        emotion_themes = Counter(all_emotions).most_common(3)
        
        themes = [g[0] for g in goal_themes] + [e[0] for e in emotion_themes]
        return list(set(themes))
    
    def get_current_state_for_generation(
        self, 
        character_id: str, 
        chapter: int
    ) -> Optional[CharacterState]:
        """
        Retrieve character state for specific chapter (for generation context).
        
        Args:
            character_id: Character to get state for
            chapter: Chapter number
            
        Returns:
            Most recent state at or before the given chapter
        """
        if character_id not in self.character_timelines:
            return None
        
        # Find most recent state at or before this chapter
        valid_states = [
            s for s in self.character_timelines[character_id] 
            if s.chapter <= chapter
        ]
        
        return valid_states[-1] if valid_states else None
    
    def predict_next_arc_phase(self, character_id: str) -> str:
        """
        Predict what should happen next in character arc.
        
        Uses simple three-act structure as heuristic.
        """
        if character_id not in self.character_timelines:
            return "setup"
        
        current_state = self.character_timelines[character_id][-1]
        
        # Phase progression order
        phase_order = ["setup", "rising_action", "crisis", "climax", "resolution"]
        
        current_phase = current_state.arc_phase
        if current_phase in phase_order:
            current_idx = phase_order.index(current_phase)
            if current_idx < len(phase_order) - 1:
                return phase_order[current_idx + 1]
        
        return "resolution"
    
    def get_all_character_summaries(self) -> List[Dict]:
        """Get arc summaries for all tracked characters."""
        summaries = []
        for char_id in self.character_timelines:
            summary = self.get_character_arc_summary(char_id)
            if summary:
                summaries.append(summary)
        return summaries
