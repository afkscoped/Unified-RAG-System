"""
Plot Timeline Visualizer

Creates timeline visualizations showing plot point progression,
event causality, and story structure analysis.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import re

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class EventType(Enum):
    """Types of plot events."""
    INTRODUCTION = "introduction"
    CONFLICT = "conflict"
    REVELATION = "revelation"
    CLIMAX = "climax"
    RESOLUTION = "resolution"
    DEVELOPMENT = "development"


@dataclass
class PlotPoint:
    """Represents a plot point/event."""
    id: str
    title: str
    description: str
    event_type: EventType
    chapter: int
    sequence: int  # Order within chapter
    involved_characters: List[str] = field(default_factory=list)
    causes: List[str] = field(default_factory=list)  # IDs of causing events
    effects: List[str] = field(default_factory=list)  # IDs of caused events
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "event_type": self.event_type.value,
            "chapter": self.chapter,
            "sequence": self.sequence,
            "involved_characters": self.involved_characters,
            "causes": self.causes,
            "effects": self.effects
        }


class PlotPointExtractor:
    """Extracts plot points from generated story text."""
    
    # Patterns for detecting event types
    EVENT_PATTERNS = {
        EventType.INTRODUCTION: [
            r'\bfirst met\b', r'\bintroduced\b', r'\barrived at\b',
            r'\bappeared\b', r'\bentered\b', r'\bbegan\b'
        ],
        EventType.CONFLICT: [
            r'\bfought\b', r'\bargued\b', r'\bconfronted\b',
            r'\bclashed\b', r'\bdisagreed\b', r'\battacked\b',
            r'\bthreatened\b', r'\bbetrayed\b'
        ],
        EventType.REVELATION: [
            r'\bdiscovered\b', r'\brevealed\b', r'\blearned\b',
            r'\brealized\b', r'\buncovered\b', r'\bfound out\b',
            r'\bsecret\b', r'\btruth\b'
        ],
        EventType.CLIMAX: [
            r'\bfinally\b', r'\bat last\b', r'\bdecisive\b',
            r'\bshowdown\b', r'\bconfrontation\b', r'\bultimate\b'
        ],
        EventType.RESOLUTION: [
            r'\bresolved\b', r'\bended\b', r'\bpeace\b',
            r'\breconciled\b', r'\bforgave\b', r'\bconcluded\b'
        ]
    }
    
    def __init__(self):
        """Initialize plot point extractor."""
        self.plot_points: List[PlotPoint] = []
        self.point_counter = 0
        logger.info("PlotPointExtractor initialized")
    
    def extract_from_text(
        self,
        text: str,
        chapter: int,
        entities: Optional[List[Dict]] = None
    ) -> List[PlotPoint]:
        """
        Extract plot points from a story segment.
        
        Args:
            text: Story text to analyze
            chapter: Current chapter number
            entities: Known entities for character matching
            
        Returns:
            List of extracted PlotPoints
        """
        entities = entities or []
        character_names = [e.get('name', '').lower() for e in entities 
                         if e.get('type') == 'CHARACTER']
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
        
        new_points = []
        
        for i, sentence in enumerate(sentences):
            # Detect event type
            event_type = self._classify_sentence(sentence)
            
            if event_type:
                # Find involved characters
                involved = []
                for name in character_names:
                    if name in sentence.lower():
                        involved.append(name.title())
                
                # Create plot point
                self.point_counter += 1
                point = PlotPoint(
                    id=f"event_{chapter}_{self.point_counter}",
                    title=self._generate_title(sentence, event_type),
                    description=sentence[:200],
                    event_type=event_type,
                    chapter=chapter,
                    sequence=i,
                    involved_characters=involved
                )
                
                new_points.append(point)
                self.plot_points.append(point)
        
        # Link causal relationships
        self._link_events(new_points)
        
        return new_points
    
    def _classify_sentence(self, sentence: str) -> Optional[EventType]:
        """Classify a sentence by event type."""
        sentence_lower = sentence.lower()
        
        # Count matches for each type
        scores = {}
        for event_type, patterns in self.EVENT_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, sentence_lower))
            if score > 0:
                scores[event_type] = score
        
        if not scores:
            # Default to development if sentence seems significant
            if len(sentence) > 50 and any(w in sentence_lower for w in ['then', 'suddenly', 'when']):
                return EventType.DEVELOPMENT
            return None
        
        return max(scores, key=scores.get)
    
    def _generate_title(self, sentence: str, event_type: EventType) -> str:
        """Generate a short title for the event."""
        # Extract first few words
        words = sentence.split()[:6]
        title = " ".join(words)
        if len(title) > 50:
            title = title[:47] + "..."
        return title
    
    def _link_events(self, new_points: List[PlotPoint]) -> None:
        """Link events based on causal patterns."""
        causal_words = ['because', 'therefore', 'thus', 'as a result', 
                       'consequently', 'led to', 'caused']
        
        for i, point in enumerate(new_points):
            # Check if this point references causes
            for word in causal_words:
                if word in point.description.lower():
                    # Link to previous point
                    if i > 0:
                        point.causes.append(new_points[i-1].id)
                        new_points[i-1].effects.append(point.id)
    
    def get_all_points(self) -> List[PlotPoint]:
        """Get all extracted plot points."""
        return self.plot_points
    
    def clear(self) -> None:
        """Clear all plot points."""
        self.plot_points = []
        self.point_counter = 0


class PlotTimelineVisualizer:
    """Creates timeline visualizations for plot points."""
    
    # Colors for event types
    EVENT_COLORS = {
        EventType.INTRODUCTION: "#3498db",
        EventType.CONFLICT: "#e74c3c",
        EventType.REVELATION: "#f39c12",
        EventType.CLIMAX: "#9b59b6",
        EventType.RESOLUTION: "#27ae60",
        EventType.DEVELOPMENT: "#95a5a6"
    }
    
    def __init__(self, extractor: Optional[PlotPointExtractor] = None):
        """
        Initialize timeline visualizer.
        
        Args:
            extractor: PlotPointExtractor instance
        """
        self.extractor = extractor or PlotPointExtractor()
        logger.info("PlotTimelineVisualizer initialized")
    
    def create_timeline(
        self,
        height: int = 400,
        width: int = 900
    ) -> Optional[go.Figure]:
        """Create horizontal timeline of plot points."""
        if not PLOTLY_AVAILABLE:
            return None
        
        points = self.extractor.get_all_points()
        
        if not points:
            return self._empty_figure("No plot points extracted yet", height, width)
        
        # Group by chapter
        by_chapter = {}
        for point in points:
            if point.chapter not in by_chapter:
                by_chapter[point.chapter] = []
            by_chapter[point.chapter].append(point)
        
        # Create traces
        traces = []
        
        for chapter in sorted(by_chapter.keys()):
            chapter_points = by_chapter[chapter]
            
            x_vals = []
            y_vals = []
            colors = []
            texts = []
            hovers = []
            
            for i, point in enumerate(chapter_points):
                x_vals.append(chapter + i * 0.15)  # Spread within chapter
                y_vals.append(0.5 - i * 0.1)  # Stagger vertically
                colors.append(self.EVENT_COLORS.get(point.event_type, "#95a5a6"))
                texts.append(point.title[:20])
                
                hover = f"<b>{point.title}</b><br>"
                hover += f"Type: {point.event_type.value}<br>"
                hover += f"Chapter {point.chapter}<br>"
                if point.involved_characters:
                    hover += f"Characters: {', '.join(point.involved_characters[:3])}"
                hovers.append(hover)
            
            traces.append(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers+text',
                marker=dict(size=15, color=colors),
                text=texts,
                textposition='top center',
                hovertext=hovers,
                hoverinfo='text',
                name=f"Chapter {chapter}",
                showlegend=True
            ))
        
        # Add causal connections
        for point in points:
            for effect_id in point.effects:
                effect = next((p for p in points if p.id == effect_id), None)
                if effect:
                    traces.append(go.Scatter(
                        x=[point.chapter, effect.chapter],
                        y=[0.5, 0.5],
                        mode='lines',
                        line=dict(color='rgba(100,100,100,0.3)', width=1, dash='dot'),
                        hoverinfo='none',
                        showlegend=False
                    ))
        
        fig = go.Figure(data=traces)
        
        fig.update_layout(
            title="Plot Timeline",
            xaxis_title="Chapter",
            height=height,
            width=width,
            showlegend=True,
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            yaxis=dict(showticklabels=False, range=[0, 1])
        )
        
        return fig
    
    def analyze_structure(self) -> Dict:
        """Analyze story structure (three-act detection, pacing)."""
        points = self.extractor.get_all_points()
        
        if not points:
            return {"status": "No plot points to analyze"}
        
        # Count events by chapter
        by_chapter = {}
        type_distribution = {}
        
        for point in points:
            by_chapter[point.chapter] = by_chapter.get(point.chapter, 0) + 1
            type_key = point.event_type.value
            type_distribution[type_key] = type_distribution.get(type_key, 0) + 1
        
        # Calculate pacing
        total_chapters = max(by_chapter.keys()) if by_chapter else 1
        events_per_chapter = len(points) / total_chapters
        
        # Detect structure phases
        phases = self._detect_three_act_structure(points, total_chapters)
        
        return {
            "total_events": len(points),
            "chapters_covered": total_chapters,
            "events_per_chapter": round(events_per_chapter, 2),
            "pacing": self._assess_pacing(events_per_chapter),
            "type_distribution": type_distribution,
            "structure_phases": phases
        }
    
    def detect_plot_holes(self) -> List[Dict]:
        """Detect potential plot holes."""
        points = self.extractor.get_all_points()
        holes = []
        
        if not points:
            return holes
        
        # Check for revelations without setup
        for point in points:
            if point.event_type == EventType.REVELATION:
                if not point.causes:
                    holes.append({
                        "type": "revelation_without_setup",
                        "description": f"Revelation '{point.title}' lacks foreshadowing",
                        "chapter": point.chapter,
                        "suggestion": "Add earlier hints or buildup"
                    })
        
        # Check for conflicts without resolution
        conflict_chapters = [p.chapter for p in points if p.event_type == EventType.CONFLICT]
        resolution_chapters = [p.chapter for p in points if p.event_type == EventType.RESOLUTION]
        
        for conflict_ch in conflict_chapters:
            if not any(r > conflict_ch for r in resolution_chapters):
                holes.append({
                    "type": "unresolved_conflict",
                    "description": f"Conflict in chapter {conflict_ch} may be unresolved",
                    "chapter": conflict_ch,
                    "suggestion": "Add resolution or ongoing consequences"
                })
        
        # Check for dangling effects
        all_ids = {p.id for p in points}
        for point in points:
            for effect_id in point.effects:
                if effect_id not in all_ids:
                    holes.append({
                        "type": "dangling_reference",
                        "description": f"Event '{point.title}' references unknown effect",
                        "chapter": point.chapter,
                        "suggestion": "Ensure effect event exists"
                    })
        
        return holes
    
    def _detect_three_act_structure(
        self,
        points: List[PlotPoint],
        total_chapters: int
    ) -> Dict:
        """Detect three-act structure phases."""
        if total_chapters < 3:
            return {"detected": False, "reason": "Not enough chapters"}
        
        # Estimate act boundaries
        act1_end = max(1, total_chapters // 4)
        act2_end = max(act1_end + 1, total_chapters * 3 // 4)
        
        # Count event types by act
        acts = {"act1": [], "act2": [], "act3": []}
        for point in points:
            if point.chapter <= act1_end:
                acts["act1"].append(point.event_type.value)
            elif point.chapter <= act2_end:
                acts["act2"].append(point.event_type.value)
            else:
                acts["act3"].append(point.event_type.value)
        
        return {
            "detected": True,
            "act1": {
                "chapters": f"1-{act1_end}",
                "events": len(acts["act1"]),
                "dominant_type": max(set(acts["act1"]), key=acts["act1"].count) if acts["act1"] else "none"
            },
            "act2": {
                "chapters": f"{act1_end+1}-{act2_end}",
                "events": len(acts["act2"]),
                "dominant_type": max(set(acts["act2"]), key=acts["act2"].count) if acts["act2"] else "none"
            },
            "act3": {
                "chapters": f"{act2_end+1}-{total_chapters}",
                "events": len(acts["act3"]),
                "dominant_type": max(set(acts["act3"]), key=acts["act3"].count) if acts["act3"] else "none"
            }
        }
    
    def _assess_pacing(self, events_per_chapter: float) -> str:
        """Assess story pacing based on event density."""
        if events_per_chapter < 1:
            return "slow"
        elif events_per_chapter < 2:
            return "moderate"
        elif events_per_chapter < 4:
            return "fast"
        else:
            return "very_fast"
    
    def _empty_figure(self, message: str, height: int, width: int) -> go.Figure:
        """Create empty figure with message."""
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=16, color="gray"),
            xref="paper", yref="paper"
        )
        fig.update_layout(height=height, width=width)
        return fig
