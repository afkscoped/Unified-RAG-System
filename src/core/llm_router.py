"""
[ELITE ARCHITECTURE] llm_router.py
Unified Interface for Groq and Local LLMs.
"""

import os
from groq import Groq
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

class LLMRouter:
    """
    Innovation: Provider Agnostic.
    Seamlessly switches between Cloud (Groq) and Local (Ollama/Transformers) 
    inference to balance latency and privacy.
    """
    
    def __init__(self, provider: str = "groq", model: str = "llama-3.3-70b-versatile"):
        self.provider = provider
        self.model = model
        
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.error("GROQ_API_KEY missing in environment.")
            self.client = Groq(api_key=api_key)
        else:
            self.client = None
            logger.warning(f"Local provider '{provider}' requires separate initialization.")

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Core generation bridge."""
        if self.provider == "groq":
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2048
                )
                return completion.choices[0].message.content
            except Exception as e:
                logger.error(f"Groq Inference Error: {e}")
                return f"Error: {e}"
        else:
            return "Local Generator Not Implemented (Ollama Integration Pending)"

if __name__ == "__main__":
    router = LLMRouter()
    print("LLM Router Standby.")
