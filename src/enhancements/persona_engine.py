"""
Persona Engine: Defines AI personalities and generates tailored responses.
Uses the system's native LLMRouter (Groq) to avoid extra API dependencies.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

# Add project root to path if needed (safety check)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.llm_router import LLMRouter
from loguru import logger

@dataclass
class Persona:
    """Definition of an AI persona"""
    id: str
    name: str
    icon: str
    description: str
    system_prompt: str

class PersonaEngine:
    """
    Manages AI personas and generates responses using the unified LLM router.
    """
    
    PERSONAS: Dict[str, Persona] = {
        'scientist': Persona(
            id='scientist',
            name='Scientist',
            icon='🔬',
            description='Detailed, citation-heavy analysis',
            system_prompt='You are a Researcher. Answer with scientific rigor, citing specific documents and evaluating evidence strength.'
        ),
        'analyst': Persona(
            id='analyst',
            name='Analyst',
            icon='📊',
            description='Data-driven business insights',
            system_prompt='You are a Business Analyst. Focus on metrics, trends, and actionable insights. Structure answers with bullet points.'
        ),
        'teacher': Persona(
            id='teacher',
            name='Teacher',
            icon='👨🏫',
            description='Simple explanations with examples',
            system_prompt='You are a Teacher. Explain complex concepts simply, using analogies and real-world examples. Be encouraging.'
        ),
        'checker': Persona(
            id='checker',
            name='Fact Checker',
            icon='🕵️',
            description='Verifies claims against sources',
            system_prompt='You are a Fact Checker. Scrutinize every claim. Explicitly state if a fact comes from (Local Doc) or (Web Search).'
        ),
        'speed': Persona(
            id='speed',
            name='Speed Reader',
            icon='⚡',
            description='Concise TL;DR summaries',
            system_prompt='You are a Speed Reader. Provide extremely concise, high-level summaries. Bullet points only. No fluff.'
        )
    }
    
    def __init__(self, llm_router: Optional[LLMRouter] = None):
        """Initialize with existing router or create new one"""
        try:
            self.llm_router = llm_router or LLMRouter()
            logger.info("PersonaEngine initialized with LLMRouter")
        except Exception as e:
            logger.error(f"Failed to init LLMRouter for PersonaEngine: {e}")
            self.llm_router = None
            
    def generate_response(self, query: str, context: str, persona_id: str) -> str:
        """Generate response using the selected persona"""
        if not self.llm_router:
            return "Error: LLM Router not available."
            
        persona = self.PERSONAS.get(persona_id, self.PERSONAS['scientist'])
        
        # Construct tailored prompt
        prompt = f"""
        [SYSTEM ROLE]
        {persona.system_prompt}
        
        [CONTEXT]
        {context}
        
        [USER QUESTION]
        {query}
        
        [INSTRUCTION]
        Answer the question using ONLY the context provided. Follow your System Role.
        """
        
        try:
            # Using the generate interface of the router
            return self.llm_router.generate(prompt, system_prompt=persona.system_prompt)
        except Exception as e:
            logger.error(f"Persona generation failed: {e}")
            return f"I encountered an error generating a response as {persona.name}. Error: {e}"
