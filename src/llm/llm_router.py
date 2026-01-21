"""
LLM Router

Routes LLM requests between Groq and Ollama
with automatic fallback and rate limiting.
"""

import os
import time
from typing import Optional, Dict, Any, Union
from enum import Enum
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class LLMProvider(Enum):
    GROQ = "groq"
    OLLAMA = "ollama"


class LLMRouter:
    """
    Routes LLM requests with fallback support.
    
    Primary: Groq (fast cloud inference)
    Fallback: Ollama (local inference)
    """
    
    def __init__(
        self,
        primary_provider: str = "groq",
        groq_model: str = "llama-3.3-70b-versatile",
        ollama_model: str = "llama3",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: int = 30
    ):
        """
        Initialize LLM router.
        
        Args:
            primary_provider: Primary provider ("groq" or "ollama")
            groq_model: Model name for Groq
            ollama_model: Model name for Ollama
            temperature: Generation temperature
            max_tokens: Max tokens to generate
            timeout: Request timeout in seconds
        """
        self.primary_provider = LLMProvider(primary_provider)
        self.groq_model = groq_model
        self.ollama_model = ollama_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests
        
        # Stats
        self._stats = {
            "groq_calls": 0,
            "ollama_calls": 0,
            "failures": 0,
            "fallbacks": 0
        }
        
        # Initialize clients
        self._groq_client = None
        self._ollama_client = None
        
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize LLM clients."""
        # Groq
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            try:
                from langchain_groq import ChatGroq
                self._groq_client = ChatGroq(
                    model_name=self.groq_model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=groq_api_key
                )
                logger.info(f"Groq client initialized ({self.groq_model})")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq: {e}")
        else:
            logger.warning("GROQ_API_KEY not set")
        
        # Ollama
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        try:
            from langchain_community.llms import Ollama
            self._ollama_client = Ollama(
                model=self.ollama_model,
                base_url=ollama_host,
                temperature=self.temperature
            )
            logger.info(f"Ollama client initialized ({self.ollama_model})")
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama: {e}")
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: User prompt
            provider: Override provider ("groq" or "ollama")
            system_prompt: Optional system prompt
            
        Returns:
            Generated text
        """
        self._rate_limit()
        
        target_provider = LLMProvider(provider) if provider else self.primary_provider
        
        # Try primary provider
        result = self._call_provider(target_provider, prompt, system_prompt)
        if result is not None:
            return result
        
        # Fallback to other provider
        fallback_provider = (
            LLMProvider.OLLAMA if target_provider == LLMProvider.GROQ 
            else LLMProvider.GROQ
        )
        
        logger.warning(f"Falling back from {target_provider.value} to {fallback_provider.value}")
        self._stats["fallbacks"] += 1
        
        result = self._call_provider(fallback_provider, prompt, system_prompt)
        if result is not None:
            return result
        
        # Both failed
        self._stats["failures"] += 1
        raise RuntimeError("All LLM providers failed")
    
    def _call_provider(
        self,
        provider: LLMProvider,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Optional[str]:
        """Call a specific provider."""
        try:
            if provider == LLMProvider.GROQ and self._groq_client:
                messages = []
                if system_prompt:
                    messages.append(("system", system_prompt))
                messages.append(("human", prompt))
                
                response = self._groq_client.invoke(messages)
                self._stats["groq_calls"] += 1
                return response.content
                
            elif provider == LLMProvider.OLLAMA and self._ollama_client:
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                    
                response = self._ollama_client.invoke(full_prompt)
                self._stats["ollama_calls"] += 1
                return response
                
        except Exception as e:
            logger.error(f"Provider {provider.value} failed: {e}")
            return None
            
        return None
    
    async def agenerate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Async version of generate."""
        # For now, just call sync version
        # TODO: Implement true async when langchain-groq supports it better
        return self.generate(prompt, provider, system_prompt)
    
    def get_stats(self) -> Dict:
        """Get router statistics."""
        return dict(self._stats)
    
    @property
    def is_groq_available(self) -> bool:
        return self._groq_client is not None
    
    @property
    def is_ollama_available(self) -> bool:
        return self._ollama_client is not None

