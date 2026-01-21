"""
Feature toggles and configuration for Antigravity Enhancement Layer.
"""
import os
from dataclasses import dataclass
from typing import Literal

@dataclass
class EnhancementConfig:
    """Configuration for optional enhancement features"""
    
    # Feature toggles
    enable_web_search: bool = False
    enable_personas: bool = False
    
    # Defaults
    default_persona: str = 'scientist'
    default_search_mode: Literal['docs', 'web', 'hybrid'] = 'docs'
    
    # Safety & Performance
    web_search_timeout: int = 5  # seconds
    max_web_results: int = 3
    fallback_to_local_on_error: bool = True
    
    @classmethod
    def from_env(cls):
        """Load config from environment variables safely"""
        return cls(
            enable_web_search=os.getenv('ENABLE_WEB_SEARCH', 'false').lower() == 'true',
            enable_personas=os.getenv('ENABLE_PERSONAS', 'false').lower() == 'true',
            default_persona=os.getenv('DEFAULT_PERSONA', 'scientist'),
            default_search_mode=os.getenv('DEFAULT_SEARCH_MODE', 'docs')
        )

# Global safe config instance
config = EnhancementConfig.from_env()
