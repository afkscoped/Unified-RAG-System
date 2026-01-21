"""
Advanced Persona Engine with Query Processing Integration
Each persona transforms queries, adjusts search weights, and shapes responses
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class PersonaType(Enum):
    """Available persona archetypes"""
    SCIENTIST = "scientist"
    CREATIVE = "creative"
    ANALYST = "analyst"
    EDUCATOR = "educator"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"


@dataclass
class QueryTransformation:
    """How a persona transforms the original query"""
    expanded_query: str
    sub_queries: List[str]
    focus_keywords: List[str]
    search_strategy: str  # 'precise', 'exploratory', 'comprehensive'


@dataclass
class SearchWeightProfile:
    """Search weight preferences for each persona"""
    bm25_weight: float  # Lexical search importance
    semantic_weight: float  # Semantic search importance
    recency_boost: float  # Boost recent documents
    diversity_preference: float  # Prefer diverse sources
    citation_importance: float  # Prefer highly-cited sources


@dataclass
class ResponseTemplate:
    """How a persona structures responses"""
    opening_style: str  # 'direct', 'contextual', 'provocative'
    explanation_depth: str  # 'concise', 'moderate', 'comprehensive'
    use_analogies: bool
    include_counterpoints: bool
    citation_style: str  # 'inline', 'footnotes', 'embedded'
    follow_up_questions: int  # Number of suggested follow-ups


@dataclass
class Persona:
    """Complete persona configuration with functional behaviors"""
    name: str
    type: PersonaType
    system_prompt: str
    temperature: float
    top_p: float
    max_tokens: int
    thinking_style: str
    icon: str
    
    # Functional components
    search_weights: SearchWeightProfile
    response_template: ResponseTemplate
    query_expansion_strategy: str  # 'conservative', 'moderate', 'aggressive'
    evaluation_priorities: Dict[str, float]


class PersonaEngine:
    """Manages persona-driven document analysis"""
    
    def __init__(self):
        self.personas = self._initialize_personas()
        self.active_persona = self.personas[PersonaType.SCIENTIST]
    
    def _initialize_personas(self) -> Dict[PersonaType, Persona]:
        """Define all personas with complete functional specifications"""
        return {
            PersonaType.SCIENTIST: Persona(
                name="Scientific Analyst",
                type=PersonaType.SCIENTIST,
                system_prompt="""You are a rigorous scientific analyst. Your responses must be:
                - Evidence-based with explicit citations to source documents
                - Precise and technical, using domain-specific terminology
                - Honest about uncertainty and data limitations
                - Structured with clear methodology
                - Focused on reproducibility and verification
                
                Format: Start with the most direct answer, then provide supporting evidence.
                Always distinguish between established facts and hypotheses.""",
                temperature=0.3,
                top_p=0.9,
                max_tokens=1500,
                thinking_style="Methodical, evidence-driven, skeptical",
                icon="🔬",
                search_weights=SearchWeightProfile(
                    bm25_weight=0.65,
                    semantic_weight=0.35,
                    recency_boost=0.1,
                    diversity_preference=0.3,
                    citation_importance=0.8
                ),
                response_template=ResponseTemplate(
                    opening_style='direct',
                    explanation_depth='comprehensive',
                    use_analogies=False,
                    include_counterpoints=True,
                    citation_style='inline',
                    follow_up_questions=2
                ),
                query_expansion_strategy='conservative',
                evaluation_priorities={'faithfulness': 0.5, 'relevance': 0.2, 'citation_coverage': 0.3}
            ),
            
            PersonaType.CREATIVE: Persona(
                name="Creative Synthesizer",
                type=PersonaType.CREATIVE,
                system_prompt="""You are a creative thinker who synthesizes information into compelling narratives.
                - Find unexpected connections between disparate ideas
                - Use vivid analogies and metaphors to illuminate concepts
                - Present information as engaging stories with clear arcs
                - Balance creativity with factual accuracy
                - Generate novel insights by combining sources""",
                temperature=0.8,
                top_p=0.95,
                max_tokens=2000,
                thinking_style="Associative, narrative-driven, exploratory",
                icon="🎨",
                search_weights=SearchWeightProfile(
                    bm25_weight=0.25,
                    semantic_weight=0.75,
                    recency_boost=0.05,
                    diversity_preference=0.9,
                    citation_importance=0.4
                ),
                response_template=ResponseTemplate(
                    opening_style='contextual',
                    explanation_depth='moderate',
                    use_analogies=True,
                    include_counterpoints=False,
                    citation_style='embedded',
                    follow_up_questions=4
                ),
                query_expansion_strategy='aggressive',
                evaluation_priorities={'faithfulness': 0.3, 'relevance': 0.5, 'citation_coverage': 0.2}
            ),
            
            PersonaType.ANALYST: Persona(
                name="Business Analyst",
                type=PersonaType.ANALYST,
                system_prompt="""You are a strategic business analyst focused on actionable insights.
                - Identify key metrics, trends, and patterns
                - Provide data-driven recommendations with clear next steps
                - Assess risks, opportunities, and trade-offs
                - Structure information for executive decision-making
                - Quantify impact when possible""",
                temperature=0.4,
                top_p=0.9,
                max_tokens=1500,
                thinking_style="Strategic, metrics-focused, pragmatic",
                icon="📊",
                search_weights=SearchWeightProfile(
                    bm25_weight=0.45,
                    semantic_weight=0.55,
                    recency_boost=0.7,
                    diversity_preference=0.5,
                    citation_importance=0.6
                ),
                response_template=ResponseTemplate(
                    opening_style='direct',
                    explanation_depth='concise',
                    use_analogies=False,
                    include_counterpoints=True,
                    citation_style='footnotes',
                    follow_up_questions=3
                ),
                query_expansion_strategy='moderate',
                evaluation_priorities={'faithfulness': 0.25, 'relevance': 0.5, 'citation_coverage': 0.25}
            ),
            
            PersonaType.EDUCATOR: Persona(
                name="Educator",
                type=PersonaType.EDUCATOR,
                system_prompt="""You are a patient educator committed to deep understanding.
                - Break complex concepts into digestible building blocks
                - Use progressive disclosure (simple → intermediate → advanced)
                - Provide concrete examples before abstract principles
                - Anticipate and address common misconceptions
                - Check for understanding with reflection questions""",
                temperature=0.5,
                top_p=0.9,
                max_tokens=2000,
                thinking_style="Pedagogical, patient, example-driven",
                icon="👨‍🏫",
                search_weights=SearchWeightProfile(
                    bm25_weight=0.4,
                    semantic_weight=0.6,
                    recency_boost=0.2,
                    diversity_preference=0.7,
                    citation_importance=0.5
                ),
                response_template=ResponseTemplate(
                    opening_style='contextual',
                    explanation_depth='comprehensive',
                    use_analogies=True,
                    include_counterpoints=False,
                    citation_style='embedded',
                    follow_up_questions=5
                ),
                query_expansion_strategy='aggressive',
                evaluation_priorities={'faithfulness': 0.35, 'relevance': 0.4, 'citation_coverage': 0.25}
            ),
            
            PersonaType.CRITIC: Persona(
                name="Critical Reviewer",
                type=PersonaType.CRITIC,
                system_prompt="""You are a critical reviewer who evaluates information skeptically.
                - Identify logical flaws, gaps, and unsupported claims
                - Highlight conflicting evidence and contradictions
                - Question assumptions and challenge conventional wisdom
                - Assess source credibility and potential biases
                - Provide balanced critique acknowledging strengths""",
                temperature=0.4,
                top_p=0.85,
                max_tokens=1500,
                thinking_style="Skeptical, analytical, balanced",
                icon="🔍",
                search_weights=SearchWeightProfile(
                    bm25_weight=0.5,
                    semantic_weight=0.5,
                    recency_boost=0.3,
                    diversity_preference=0.95,
                    citation_importance=0.7
                ),
                response_template=ResponseTemplate(
                    opening_style='provocative',
                    explanation_depth='moderate',
                    use_analogies=False,
                    include_counterpoints=True,
                    citation_style='inline',
                    follow_up_questions=3
                ),
                query_expansion_strategy='moderate',
                evaluation_priorities={'faithfulness': 0.4, 'relevance': 0.3, 'citation_coverage': 0.3}
            ),
            
            PersonaType.SYNTHESIZER: Persona(
                name="Holistic Synthesizer",
                type=PersonaType.SYNTHESIZER,
                system_prompt="""You synthesize information from multiple perspectives into coherent frameworks.
                - Integrate diverse viewpoints into unified understanding
                - Map relationships between concepts and sources
                - Build comprehensive mental models
                - Connect micro-details to macro-patterns
                - Present multi-dimensional analysis""",
                temperature=0.6,
                top_p=0.92,
                max_tokens=2000,
                thinking_style="Integrative, holistic, systematic",
                icon="🌐",
                search_weights=SearchWeightProfile(
                    bm25_weight=0.35,
                    semantic_weight=0.65,
                    recency_boost=0.4,
                    diversity_preference=0.85,
                    citation_importance=0.6
                ),
                response_template=ResponseTemplate(
                    opening_style='contextual',
                    explanation_depth='comprehensive',
                    use_analogies=True,
                    include_counterpoints=True,
                    citation_style='embedded',
                    follow_up_questions=4
                ),
                query_expansion_strategy='aggressive',
                evaluation_priorities={'faithfulness': 0.3, 'relevance': 0.4, 'citation_coverage': 0.3}
            )
        }
    
    def set_persona(self, persona_type: PersonaType) -> None:
        """Switch active persona"""
        if persona_type in self.personas:
            self.active_persona = self.personas[persona_type]
    
    def get_active_persona(self) -> Persona:
        """Get currently active persona"""
        return self.active_persona
    
    def get_all_personas(self) -> List[Persona]:
        """Get list of all available personas"""
        return list(self.personas.values())
    
    def transform_query(self, original_query: str) -> QueryTransformation:
        """Transform query based on active persona's cognitive style"""
        persona = self.active_persona
        
        expanded_query = original_query
        sub_queries = []
        focus_keywords = []
        
        if persona.query_expansion_strategy == 'conservative':
            # Scientist: Precise, minimal expansion
            focus_keywords = self._extract_technical_terms(original_query)
            sub_queries = [
                f"Evidence for {original_query}",
                f"Research on {original_query}"
            ]
            search_strategy = 'precise'
        
        elif persona.query_expansion_strategy == 'moderate':
            # Analyst/Critic: Balanced expansion
            focus_keywords = self._extract_key_concepts(original_query)
            
            if "trend" in original_query.lower() or "change" in original_query.lower():
                sub_queries.append(f"Historical {original_query}")
                sub_queries.append(f"Recent developments in {original_query}")
            
            sub_queries.extend([
                f"{original_query} implications",
                f"{original_query} examples"
            ])
            search_strategy = 'exploratory'
        
        elif persona.query_expansion_strategy == 'aggressive':
            # Creative/Educator/Synthesizer: Broad expansion
            focus_keywords = self._extract_all_concepts(original_query)
            
            sub_queries = [
                f"{original_query} fundamentals",
                f"{original_query} applications",
                f"{original_query} related concepts",
                f"Alternatives to {original_query}",
                f"{original_query} future implications",
                f"What is {original_query} similar to"
            ]
            search_strategy = 'comprehensive'
        else:
            search_strategy = 'precise'
        
        return QueryTransformation(
            expanded_query=expanded_query,
            sub_queries=sub_queries[:4],
            focus_keywords=focus_keywords,
            search_strategy=search_strategy
        )
    
    def _extract_technical_terms(self, query: str) -> List[str]:
        """Extract likely technical terms"""
        words = query.split()
        technical = [w for w in words if len(w) > 0 and (w[0].isupper() or len(w) > 8)]
        return technical[:5]
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract main concepts from query"""
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an'}
        words = [w.lower() for w in query.split() if w.lower() not in stop_words]
        return words[:7]
    
    def _extract_all_concepts(self, query: str) -> List[str]:
        """Extract all meaningful words"""
        stop_words = {'is', 'are', 'the', 'a', 'an', 'and', 'or'}
        words = [w.lower() for w in query.split() if w.lower() not in stop_words]
        return words
    
    def get_search_weights(self) -> SearchWeightProfile:
        """Get search weight profile for active persona"""
        return self.active_persona.search_weights
    
    def get_response_template(self) -> ResponseTemplate:
        """Get response template for active persona"""
        return self.active_persona.response_template
    
    def get_generation_params(self) -> Dict:
        """Get LLM parameters for active persona"""
        return {
            'temperature': self.active_persona.temperature,
            'top_p': self.active_persona.top_p,
            'max_tokens': self.active_persona.max_tokens,
            'system_prompt': self.active_persona.system_prompt
        }
    
    def generate_follow_up_questions(self, query: str, response: str) -> List[str]:
        """Generate persona-specific follow-up questions"""
        persona = self.active_persona
        num_questions = persona.response_template.follow_up_questions
        
        # Extract main topic from query
        topic = query.lower().replace("what is", "").replace("how does", "").strip()
        
        if persona.type == PersonaType.SCIENTIST:
            return [
                f"What evidence supports these claims about {topic}?",
                f"What are the limitations of this research?"
            ][:num_questions]
        
        elif persona.type == PersonaType.CREATIVE:
            return [
                f"What unexpected applications could {topic} have?",
                f"How is {topic} similar to other concepts?",
                f"What would happen if {topic} were different?",
                f"What stories illustrate {topic}?"
            ][:num_questions]
        
        elif persona.type == PersonaType.ANALYST:
            return [
                f"What are the business implications of {topic}?",
                f"What metrics should we track for {topic}?",
                f"What are the risks and opportunities?"
            ][:num_questions]
        
        elif persona.type == PersonaType.EDUCATOR:
            return [
                f"Can you explain {topic} more simply?",
                f"What are real-world examples of {topic}?",
                f"What are common misconceptions about {topic}?",
                f"How does {topic} relate to other concepts?",
                f"What should I learn next?"
            ][:num_questions]
        
        elif persona.type == PersonaType.CRITIC:
            return [
                f"What are the weaknesses in the evidence?",
                f"What contradicts this view of {topic}?",
                f"What assumptions underlie {topic}?"
            ][:num_questions]
        
        elif persona.type == PersonaType.SYNTHESIZER:
            return [
                f"How does {topic} fit into a larger framework?",
                f"What perspectives are we missing?",
                f"How do different sources view {topic}?",
                f"What patterns emerge from studying {topic}?"
            ][:num_questions]
        
        return []
