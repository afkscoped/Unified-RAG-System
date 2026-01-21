"""
Groq API Integration for Instant Dataset Generation
No local model loading required - uses Groq's ultra-fast API
"""

import os
from typing import List, Dict, Optional, Callable
import json
import time

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# IMPORTANT: Do not hardcode API keys in source control.
# Provide via environment variable GROQ_API_KEY, Streamlit secrets, or UI input.
DEFAULT_GROQ_API_KEY = None


class GroqDatasetGenerator:
    """
    Generate high-quality training datasets using Groq API
    Much faster than local models - no downloading required!
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Groq client
        
        Args:
            api_key: Groq API key (or uses default/environment variable)
        """
        if not GROQ_AVAILABLE:
            raise ImportError("Install groq: pip install groq")
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or DEFAULT_GROQ_API_KEY
        if not self.api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY or pass api_key parameter")
        
        self.client = Groq(api_key=self.api_key)
        
        # Available models - all are fast!
        self.models = {
            "llama-3.1-70b": "llama-3.1-70b-versatile",
            "llama-3.1-8b": "llama-3.1-8b-instant",
            "llama-3.3-70b": "llama-3.3-70b-versatile",
            "mixtral-8x7b": "mixtral-8x7b-32768",
            "gemma2-9b": "gemma2-9b-it"
        }
        
        self.default_model = "llama-3.1-8b"  # Fastest option
        
        # Template definitions
        self.templates = {
            "General QA": {
                "topics": [
                    "artificial intelligence", "machine learning", "data science",
                    "programming basics", "web development", "databases",
                    "cloud computing", "cybersecurity", "algorithms"
                ],
                "style": "educational"
            },
            "Coding Assistant": {
                "topics": [
                    "Python programming", "JavaScript basics", "code debugging",
                    "data structures", "API development", "testing",
                    "version control", "code optimization", "best practices"
                ],
                "style": "technical"
            },
            "Customer Support": {
                "topics": [
                    "account management", "billing questions", "technical support",
                    "product features", "returns and refunds", "shipping",
                    "troubleshooting", "account security", "feedback"
                ],
                "style": "conversational"
            },
            "Educational Content": {
                "topics": [
                    "science concepts", "mathematics", "history",
                    "language learning", "study techniques", "critical thinking",
                    "problem solving", "research methods", "learning strategies"
                ],
                "style": "educational"
            },
            "Creative Writing": {
                "topics": [
                    "story writing", "poetry", "dialogue",
                    "character development", "world building", "plot structure",
                    "descriptive writing", "editing techniques", "writer's inspiration"
                ],
                "style": "creative"
            },
            "Business & Marketing": {
                "topics": [
                    "business strategy", "marketing tactics", "sales techniques",
                    "entrepreneurship", "leadership", "productivity",
                    "customer acquisition", "brand building", "analytics"
                ],
                "style": "professional"
            },
            "Technical Documentation": {
                "topics": [
                    "API documentation", "user guides", "code comments",
                    "technical writing", "software architecture", "system design",
                    "deployment guides", "troubleshooting docs", "best practices"
                ],
                "style": "technical"
            }
        }
    
    def test_api_key(self) -> bool:
        """Test if API key is valid"""
        try:
            response = self.client.chat.completions.create(
                model=self.models[self.default_model],
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            print(f"API key test failed: {e}")
            return False
    
    def generate_qa_pair(
        self,
        topic: str,
        style: str = "educational",
        model: Optional[str] = None
    ) -> Optional[Dict[str, str]]:
        """
        Generate a single Q&A pair on a topic
        
        Args:
            topic: Topic or subject area
            style: Style (educational, conversational, technical, creative)
            model: Groq model to use
        
        Returns:
            Dictionary with instruction and output
        """
        model_id = self.models.get(model or self.default_model, self.models[self.default_model])
        
        prompt = f"""Generate a training example for fine-tuning a language model.

Topic: {topic}
Style: {style}

Create ONE question-answer pair in JSON format:
{{
  "instruction": "A clear, specific question or task",
  "output": "A detailed, helpful response (100-200 words)"
}}

Make it high-quality and educational. Output ONLY valid JSON, no other text."""

        try:
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=500
            )
            
            # Parse JSON response
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
                if content.startswith("json"):
                    content = content[4:].strip()
            
            # Try to find JSON in content
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                content = content[start_idx:end_idx]
            
            qa_pair = json.loads(content.strip())
            
            # Validate structure
            if "instruction" in qa_pair and "output" in qa_pair:
                return qa_pair
            return None
            
        except Exception as e:
            print(f"Error generating pair: {e}")
            return None
    
    def generate_dataset(
        self,
        topics: List[str],
        samples_per_topic: int = 10,
        style: str = "educational",
        model: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        delay_between_requests: float = 0.5
    ) -> List[Dict[str, str]]:
        """
        Generate a complete training dataset
        
        Args:
            topics: List of topics to cover
            samples_per_topic: Number of samples per topic
            style: Generation style
            model: Groq model to use
            progress_callback: Function to call with (current, total)
            delay_between_requests: Seconds to wait between API calls
        
        Returns:
            List of instruction-output pairs
        """
        dataset = []
        total = len(topics) * samples_per_topic
        current = 0
        
        for topic in topics:
            for i in range(samples_per_topic):
                qa_pair = self.generate_qa_pair(topic, style, model)
                
                if qa_pair:
                    dataset.append(qa_pair)
                
                current += 1
                if progress_callback:
                    progress_callback(current, total)
                
                # Small delay to avoid rate limits
                if delay_between_requests > 0:
                    time.sleep(delay_between_requests)
        
        return dataset
    
    def generate_from_template(
        self,
        template_name: str,
        num_samples: int = 50,
        model: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, str]]:
        """
        Generate dataset from predefined template
        
        Args:
            template_name: Template category name
            num_samples: Total samples to generate
            model: Groq model to use
            progress_callback: Progress function
        
        Returns:
            List of training samples
        """
        template = self.templates.get(template_name)
        if not template:
            template = self.templates["General QA"]
        
        topics = template["topics"]
        style = template["style"]
        samples_per_topic = max(1, num_samples // len(topics))
        
        # Adjust to get close to requested number
        extra_samples = num_samples - (samples_per_topic * len(topics))
        
        dataset = self.generate_dataset(
            topics=topics,
            samples_per_topic=samples_per_topic,
            style=style,
            model=model,
            progress_callback=progress_callback
        )
        
        # Generate extra samples if needed
        if extra_samples > 0 and len(topics) > 0:
            for i in range(extra_samples):
                topic = topics[i % len(topics)]
                qa_pair = self.generate_qa_pair(topic, style, model)
                if qa_pair:
                    dataset.append(qa_pair)
        
        return dataset
    
    def generate_custom_samples(
        self,
        custom_prompt: str,
        num_samples: int = 10,
        model: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, str]]:
        """
        Generate samples from custom prompt
        
        Args:
            custom_prompt: Custom generation instructions
            num_samples: Number of samples
            model: Model to use
            progress_callback: Progress callback
        
        Returns:
            List of samples
        """
        model_id = self.models.get(model or self.default_model, self.models[self.default_model])
        dataset = []
        
        for i in range(num_samples):
            prompt = f"""{custom_prompt}

Generate ONE unique training example as JSON:
{{
  "instruction": "question or task",
  "output": "detailed response"
}}

This is sample {i+1} of {num_samples}. Make each one different.
Output ONLY valid JSON."""

            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    max_tokens=500
                )
                
                content = response.choices[0].message.content.strip()
                
                # Parse JSON
                if content.startswith("```"):
                    lines = content.split("\n")
                    content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
                
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    content = content[start_idx:end_idx]
                
                qa_pair = json.loads(content)
                if "instruction" in qa_pair and "output" in qa_pair:
                    dataset.append(qa_pair)
                
            except Exception as e:
                print(f"Error generating custom sample: {e}")
            
            if progress_callback:
                progress_callback(i + 1, num_samples)
            
            time.sleep(0.5)  # Rate limit protection
        
        return dataset
    
    def get_available_templates(self) -> List[str]:
        """Get list of available template names"""
        return list(self.templates.keys())
    
    def get_available_models(self) -> Dict[str, str]:
        """Get available Groq models"""
        return self.models.copy()


def generate_groq_dataset(
    api_key: str = None,
    template_name: str = "General QA",
    num_samples: int = 50,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Dict[str, str]]:
    """
    Quick function to generate dataset with Groq
    
    Args:
        api_key: Your Groq API key (uses default if not provided)
        template_name: Template category
        num_samples: Number of samples
        progress_callback: Optional progress function
    
    Returns:
        List of training samples
    """
    generator = GroqDatasetGenerator(api_key)
    return generator.generate_from_template(
        template_name,
        num_samples,
        progress_callback=progress_callback
    )
