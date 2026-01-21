"""
Consistency Checker

Evaluates story consistency with detailed violation detection.
Shows why Graph RAG performs better than Unified RAG for narratives.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import re


class ViolationSeverity(Enum):
    """Severity levels for consistency violations."""
    CRITICAL = "critical"   # Breaks story logic completely
    MAJOR = "major"         # Significant inconsistency
    MINOR = "minor"         # Small issues, reader might not notice


class ViolationType(Enum):
    """Types of consistency violations."""
    NAME_INCONSISTENCY = "name_inconsistency"
    CONTRADICTION = "contradiction"
    TEMPORAL_ERROR = "temporal_error"
    RELATIONSHIP_VIOLATION = "relationship_violation"
    CHARACTER_STATE_VIOLATION = "character_state_violation"
    LOCATION_IMPOSSIBILITY = "location_impossibility"
    KNOWLEDGE_VIOLATION = "knowledge_violation"


@dataclass
class ConsistencyViolation:
    """Represents a single consistency violation."""
    type: ViolationType
    severity: ViolationSeverity
    description: str
    evidence: str  # Text excerpt showing the violation
    suggestion: str  # How to fix it
    approach: str  # Which approach produced this
    
    def to_dict(self) -> Dict:
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "description": self.description,
            "evidence": self.evidence,
            "suggestion": self.suggestion,
            "approach": self.approach
        }


@dataclass 
class ConsistencyReport:
    """Full consistency analysis report."""
    violations: List[ConsistencyViolation] = field(default_factory=list)
    score: float = 1.0  # 0-1, 1 being perfectly consistent
    
    # Breakdown by type
    by_type: Dict[str, int] = field(default_factory=dict)
    by_severity: Dict[str, int] = field(default_factory=dict)
    
    # Comparison data
    prevented_by_graph: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "score": self.score,
            "total_violations": len(self.violations),
            "violations": [v.to_dict() for v in self.violations],
            "by_type": self.by_type,
            "by_severity": self.by_severity,
            "prevented_by_graph": self.prevented_by_graph
        }


class ConsistencyChecker:
    """
    Checks story consistency with different levels for different approaches.
    
    Unified RAG: Basic checks (names, contradictions, temporal)
    Graph RAG: Advanced checks (relationship, state, location, knowledge)
    """
    
    def __init__(self, story_graph=None, arc_tracker=None):
        """
        Initialize consistency checker.
        
        Args:
            story_graph: StoryKnowledgeGraph for relationship checks
            arc_tracker: DynamicArcTracker for character state checks
        """
        self.story_graph = story_graph
        self.arc_tracker = arc_tracker
        
        # Known relationship types and their implications
        self.relationship_behaviors = {
            "ALLIES_WITH": ["helps", "supports", "trusts", "works with"],
            "CONFLICTS_WITH": ["fights", "opposes", "hates", "attacks"],
            "LOVES": ["cares for", "protects", "cherishes"],
            "FEARS": ["avoids", "runs from", "trembles"],
            "FAMILY": ["related to", "blood of", "kin"]
        }
        
        # Conflicting relationship pairs
        self.conflicting_relations = [
            ("ALLIES_WITH", "CONFLICTS_WITH"),
            ("LOVES", "HATES"),
            ("TRUSTS", "BETRAYS")
        ]
        
        logger.info("ConsistencyChecker initialized")
    
    def check_unified_rag(
        self,
        text: str,
        context: str,
        entities: List[Dict],
        previous_segments: List[str]
    ) -> ConsistencyReport:
        """
        Check consistency for Unified RAG output (basic checks only).
        """
        violations = []
        
        # Check name consistency
        violations.extend(self._check_name_consistency(text, entities, "unified"))
        
        # Check for contradictions
        violations.extend(self._check_contradictions(text, previous_segments, "unified"))
        
        # Check temporal errors
        violations.extend(self._check_temporal_errors(text, previous_segments, "unified"))
        
        return self._build_report(violations)
    
    def check_graph_rag(
        self,
        text: str,
        context: str,
        entities: List[Dict],
        previous_segments: List[str],
        chapter: int = 1
    ) -> ConsistencyReport:
        """
        Check consistency for Graph RAG output (all checks).
        """
        violations = []
        prevented = []
        
        # Basic checks (same as Unified)
        violations.extend(self._check_name_consistency(text, entities, "graph"))
        violations.extend(self._check_contradictions(text, previous_segments, "graph"))
        violations.extend(self._check_temporal_errors(text, previous_segments, "graph"))
        
        # Advanced checks (Graph RAG specific)
        rel_violations, rel_prevented = self._check_relationship_violations(text, chapter)
        violations.extend(rel_violations)
        prevented.extend(rel_prevented)
        
        state_violations, state_prevented = self._check_character_state_violations(text, chapter)
        violations.extend(state_violations)
        prevented.extend(state_prevented)
        
        loc_violations = self._check_location_impossibilities(text, chapter)
        violations.extend(loc_violations)
        
        knowledge_violations = self._check_knowledge_violations(text, entities, chapter)
        violations.extend(knowledge_violations)
        
        report = self._build_report(violations)
        report.prevented_by_graph = prevented
        return report
    
    def compare_approaches(
        self,
        unified_text: str,
        graph_text: str,
        hybrid_text: str,
        context: str,
        entities: List[Dict],
        previous_segments: List[str],
        chapter: int = 1
    ) -> Dict[str, ConsistencyReport]:
        """
        Compare consistency across all three approaches.
        """
        unified_report = self.check_unified_rag(unified_text, context, entities, previous_segments)
        graph_report = self.check_graph_rag(graph_text, context, entities, previous_segments, chapter)
        
        # Hybrid uses graph checks
        hybrid_report = self.check_graph_rag(hybrid_text, context, entities, previous_segments, chapter)
        
        # Calculate what Graph RAG prevented
        unified_violations = {v.description for v in unified_report.violations}
        graph_violations = {v.description for v in graph_report.violations}
        
        prevented = unified_violations - graph_violations
        graph_report.prevented_by_graph = list(prevented)
        
        return {
            "unified": unified_report,
            "graph": graph_report,
            "hybrid": hybrid_report
        }
    
    def _check_name_consistency(
        self,
        text: str,
        entities: List[Dict],
        approach: str
    ) -> List[ConsistencyViolation]:
        """Check for inconsistent character names."""
        violations = []
        
        # Group entities by similar names
        name_groups = {}
        for entity in entities:
            name = entity.get('name', '')
            base_name = name.split()[0].lower() if name else ''
            if base_name:
                if base_name not in name_groups:
                    name_groups[base_name] = []
                name_groups[base_name].append(name)
        
        # Check for references to partial names when full name established
        for base, names in name_groups.items():
            if len(names) > 1:
                # Multiple variants exist
                for name in names:
                    if name.lower() in text.lower():
                        other_variants = [n for n in names if n != name and n.lower() in text.lower()]
                        if other_variants:
                            violations.append(ConsistencyViolation(
                                type=ViolationType.NAME_INCONSISTENCY,
                                severity=ViolationSeverity.MINOR,
                                description=f"Mixed name references: '{name}' and '{other_variants[0]}'",
                                evidence=f"Text uses both '{name}' and '{other_variants[0]}'",
                                suggestion=f"Use consistent name: '{max(names, key=len)}'",
                                approach=approach
                            ))
                            break
        
        return violations
    
    def _check_contradictions(
        self,
        text: str,
        previous_segments: List[str],
        approach: str
    ) -> List[ConsistencyViolation]:
        """Check for logical contradictions with previous content."""
        violations = []
        
        if not previous_segments:
            return violations
        
        prev_text = " ".join(previous_segments[-3:])
        
        # Simple contradiction patterns
        contradiction_pairs = [
            (r'(\w+) was alive', r'\1 (was|had been) dead'),
            (r'(\w+) was dead', r'\1 (was|is) alive'),
            (r'(\w+) trusted', r'\1 (hated|distrusted)'),
            (r'in the morning', r'at night'),
            (r'it was (bright|sunny)', r'it was (dark|night)'),
        ]
        
        for pattern_prev, pattern_curr in contradiction_pairs:
            prev_matches = re.findall(pattern_prev, prev_text, re.IGNORECASE)
            if prev_matches:
                curr_matches = re.findall(pattern_curr, text, re.IGNORECASE)
                if curr_matches:
                    violations.append(ConsistencyViolation(
                        type=ViolationType.CONTRADICTION,
                        severity=ViolationSeverity.MAJOR,
                        description=f"Contradiction detected in narrative state",
                        evidence=f"Previous: '{pattern_prev}', Current: '{pattern_curr}'",
                        suggestion="Maintain consistent world state",
                        approach=approach
                    ))
        
        return violations
    
    def _check_temporal_errors(
        self,
        text: str,
        previous_segments: List[str],
        approach: str
    ) -> List[ConsistencyViolation]:
        """Check for temporal impossibilities."""
        violations = []
        
        # Check for impossible sequences
        temporal_markers = re.findall(
            r'\b(yesterday|today|tomorrow|last week|next week|'
            r'before|after|earlier|later|first|then|finally)\b',
            text.lower()
        )
        
        # Check for 'before' referring to future events
        if 'before' in temporal_markers and any(s in text.lower() for s in ['will', 'going to']):
            if re.search(r'before.*will', text.lower()):
                violations.append(ConsistencyViolation(
                    type=ViolationType.TEMPORAL_ERROR,
                    severity=ViolationSeverity.MINOR,
                    description="Temporal reference confusion",
                    evidence="'before' used with future tense",
                    suggestion="Clarify temporal sequence",
                    approach=approach
                ))
        
        return violations
    
    def _check_relationship_violations(
        self,
        text: str,
        chapter: int
    ) -> Tuple[List[ConsistencyViolation], List[str]]:
        """Check if text violates established relationships."""
        violations = []
        prevented = []
        
        if not self.story_graph:
            return violations, prevented
        
        # Simple lemmatization map for common behaviors
        behavior_lemmas = {
            "helped": "help", "attacked": "attack", "betrayed": "betray", 
            "supported": "support", "fought": "fight", "embraced": "embrace",
            "hated": "hate", "despised": "despise", "feared": "fear",
            "trusted": "trust", "killed": "kill", "saved": "save"
        }
        
        # Get all established relationships
        for edge in self.story_graph.graph.edges(data=True):
            source, target = edge[0], edge[1]
            edge_data = edge[2]
            rel_type = edge_data.get('relation_type', '')
            
            source_entity = self.story_graph.get_entity(source)
            target_entity = self.story_graph.get_entity(target)
            
            if not source_entity or not target_entity:
                continue
            
            source_name = source_entity.name.lower()
            target_name = target_entity.name.lower()
            
            # Check if both mentioned in text
            if source_name in text.lower() and target_name in text.lower():
                conflicting_behaviors = self._get_conflicting_behaviors(rel_type)
                
                for behavior in conflicting_behaviors:
                    # Check for behavior or its lemma
                    lemma = behavior_lemmas.get(behavior, behavior)
                    
                    if lemma in text.lower():
                        violations.append(ConsistencyViolation(
                            type=ViolationType.RELATIONSHIP_VIOLATION,
                            severity=ViolationSeverity.MAJOR,
                            description=f"Behavior '{behavior}' contradicts {rel_type} between {source_entity.name} and {target_entity.name}",
                            evidence=f"Text contains '{lemma}' while relationship is {rel_type}",
                            suggestion=f"Adjust action to fit {rel_type} relationship",
                            approach="graph"
                        ))
                        prevented.append(f"Detected conflict: {source_entity.name} vs {target_entity.name} regarding '{lemma}'")
        
        return violations, prevented

    def _get_conflicting_behaviors(self, rel_type: str) -> List[str]:
        """Get behaviors that conflict with a relationship type."""
        conflicts = {
            "ALLIES_WITH": ["attacked", "betrayed", "abandoned", "fought against"],
            "CONFLICTS_WITH": ["embraced", "helped", "supported", "trusted completely"],
            "LOVES": ["hated", "despised", "couldn't stand"],
            "FEARS": ["challenged boldly", "attacked fearlessly"]
        }
        return conflicts.get(rel_type, [])

    def _check_knowledge_violations(
        self,
        text: str,
        entities: List[Dict],
        chapter: int
    ) -> List[ConsistencyViolation]:
        """Check if characters know things they shouldn't."""
        violations = []
        # Basic placeholder - would need knowledge tracking for full implementation
        return violations

    def _check_character_state_violations(
        self,
        text: str,
        chapter: int
    ) -> Tuple[List[ConsistencyViolation], List[str]]:
        """Check if character behavior contradicts their arc state."""
        violations = []
        prevented = []
        
        if not self.arc_tracker:
            return violations, prevented
        
        # Check each tracked character
        for char_id, timeline in self.arc_tracker.character_timelines.items():
            if not timeline:
                continue
            
            latest_state = timeline[-1]
            
            # Get character name
            char_name = char_id
            if self.story_graph:
                entity = self.story_graph.get_entity(char_id)
                if entity:
                    char_name = entity.name
            
            if char_name.lower() not in text.lower():
                continue
            
            # Check emotional state consistency
            emotional_state = latest_state.emotional_state
            
            contradicting_emotions = {
                "hopeful": ["despaired", "gave up", "lost all hope"],
                "angry": ["calmly", "peacefully", "gently"],
                "fearful": ["bravely charged", "fearlessly", "without hesitation"],
                "sad": ["laughed joyfully", "celebrated", "was ecstatic"]
            }
            
            if emotional_state in contradicting_emotions:
                for contradiction in contradicting_emotions[emotional_state]:
                    if contradiction in text.lower():
                        violations.append(ConsistencyViolation(
                            type=ViolationType.CHARACTER_STATE_VIOLATION,
                            severity=ViolationSeverity.MAJOR,
                            description=f"Character {char_name} acts '{contradiction}' despite being {emotional_state}",
                            evidence=f"Text: '{contradiction}' vs State: {emotional_state}",
                            suggestion=f"Align action with emotional state or show transition",
                            approach="graph"
                        ))
                        prevented.append(
                            f"Prevented: {char_name} '{contradiction}' (contradicts {emotional_state} state)"
                        )
        
        return violations, prevented

    def _check_location_impossibilities(
        self,
        text: str,
        chapter: int
    ) -> List[ConsistencyViolation]:
        """Check for location-based impossibilities."""
        violations = []
        
        if not self.story_graph:
            return violations
        
        # 1. Get current locations of characters from Graph
        character_locations = {}
        for edge in self.story_graph.graph.edges(data=True):
            if edge[2].get('relation_type') == 'LOCATED_IN':
                char_id = edge[0]
                loc_id = edge[1]
                character_locations[char_id] = loc_id
        
        # 2. Check text for placement of characters in DIFFERENT locations
        # Get all location entities
        all_locations = self.story_graph.get_entities_by_type("LOCATION")
        loc_map = {l.name.lower(): l.id for l in all_locations}
        
        text_lower = text.lower()
        
        for char_id, current_loc_id in character_locations.items():
            char_entity = self.story_graph.get_entity(char_id)
            if not char_entity or char_entity.name.lower() not in text_lower:
                continue
                
            current_loc = self.story_graph.get_entity(current_loc_id)
            if not current_loc:
                continue
                
            # Check if text places character in a DIFFERENT location
            for loc_name, loc_id in loc_map.items():
                if loc_id == current_loc_id:
                    continue
                    
                if loc_name in text_lower:
                    # Heuristic check for static placement
                    static_indicators = ["was in", "stood in", "stood at", "sat in", "at the", "inside the"]
                    
                    # Also check simple proximity "Marcus ... Forest"
                    if f"{char_entity.name.lower()} was in {loc_name}" in text_lower or \
                       f"{char_entity.name.lower()} stood in {loc_name}" in text_lower or \
                       (loc_name in text_lower and char_entity.name.lower() in text_lower and any(ind in text_lower for ind in static_indicators)):
                        
                         violations.append(ConsistencyViolation(
                            type=ViolationType.LOCATION_IMPOSSIBILITY,
                            severity=ViolationSeverity.CRITICAL,
                            description=f"{char_entity.name} appears to be in {loc_name.title()} but is established in {current_loc.name}",
                            evidence=f"Text mentions '{loc_name}' and '{char_entity.name}'",
                            suggestion=f"Move character explicitly or correct location",
                            approach="graph"
                        ))
        
        return violations
    
    def _build_report(self, violations: List[ConsistencyViolation]) -> ConsistencyReport:
        """Build consistency report from violations."""
        report = ConsistencyReport(violations=violations)
        
        # Count by type
        for v in violations:
            type_key = v.type.value
            report.by_type[type_key] = report.by_type.get(type_key, 0) + 1
        
        # Count by severity
        for v in violations:
            sev_key = v.severity.value
            report.by_severity[sev_key] = report.by_severity.get(sev_key, 0) + 1
        
        # Calculate score
        severity_weights = {
            ViolationSeverity.CRITICAL: 0.3,
            ViolationSeverity.MAJOR: 0.15,
            ViolationSeverity.MINOR: 0.05
        }
        
        total_penalty = sum(severity_weights.get(v.severity, 0.1) for v in violations)
        report.score = max(0.0, 1.0 - total_penalty)
        
        return report
