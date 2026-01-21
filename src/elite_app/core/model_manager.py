"""
Model Manager - Handles base model loading with automatic optimization
Optimized for consumer GPUs (4GB VRAM)
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from typing import Dict, Optional, Tuple
from pathlib import Path
import json


class ModelManager:
    """
    Manages LLM models with automatic 4-bit quantization
    Optimized for consumer GPUs (4GB VRAM)
    """
    
    SUPPORTED_MODELS = {
        "Llama-2-7B": {
            "id": "meta-llama/Llama-2-7b-hf",
            "size": "7B",
            "vram_required": "3.5GB",
            "context_length": 4096,
            "recommended_for": "General instruction following"
        },
        "Mistral-7B": {
            "id": "mistralai/Mistral-7B-v0.1",
            "size": "7B",
            "vram_required": "3.5GB",
            "context_length": 8192,
            "recommended_for": "Code and reasoning tasks"
        },
        "Phi-2": {
            "id": "microsoft/phi-2",
            "size": "2.7B",
            "vram_required": "2.5GB",
            "context_length": 2048,
            "recommended_for": "Fast training, limited VRAM"
        },
        "TinyLlama-1.1B": {
            "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "size": "1.1B",
            "vram_required": "1.5GB",
            "context_length": 2048,
            "recommended_for": "Ultra-fast experimentation"
        },
        "Gemma-2B": {
            "id": "google/gemma-2b",
            "size": "2B",
            "vram_required": "2.0GB",
            "context_length": 8192,
            "recommended_for": "Efficient, high-quality responses"
        }
    }
    
    def __init__(self, cache_dir: str = "./cache/model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(
        self,
        model_name: str,
        use_4bit: bool = True,
        load_in_8bit: bool = False
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model with automatic quantization
        
        Args:
            model_name: Name from SUPPORTED_MODELS or HuggingFace model ID
            use_4bit: Use 4-bit quantization (recommended for 4GB VRAM)
            load_in_8bit: Use 8-bit instead (needs more VRAM)
        
        Returns:
            (model, tokenizer) tuple
        """
        # Get model ID
        if model_name in self.SUPPORTED_MODELS:
            model_id = self.SUPPORTED_MODELS[model_name]["id"]
        else:
            model_id = model_name
        
        print(f"Loading {model_id}...")
        
        # Configure quantization
        if use_4bit and not load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            print("   Using 4-bit quantization (NF4)")
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            print("   Using 8-bit quantization")
        else:
            bnb_config = None
            print("   Using full precision (not recommended for 4GB VRAM)")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=str(self.cache_dir),
            trust_remote_code=True
        )
        
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=str(self.cache_dir),
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Store references
        self.current_model = model
        self.current_tokenizer = tokenizer
        self.current_model_name = model_name
        
        print(f"Model loaded successfully!")
        print(f"   Device: {self.device}")
        print(f"   VRAM allocated: {self._get_vram_usage():.2f} GB")
        
        return model, tokenizer
    
    def _get_vram_usage(self) -> float:
        """Get current VRAM usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a model"""
        if model_name in self.SUPPORTED_MODELS:
            return self.SUPPORTED_MODELS[model_name]
        return {"id": model_name, "size": "Unknown", "vram_required": "Unknown"}
    
    def list_models(self) -> Dict:
        """Get list of supported models"""
        return self.SUPPORTED_MODELS
    
    def unload_model(self):
        """Free memory by unloading current model"""
        if self.current_model is not None:
            del self.current_model
            del self.current_tokenizer
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("Model unloaded, memory freed")
    
    def get_current_model(self) -> Optional[AutoModelForCausalLM]:
        """Get currently loaded model"""
        return self.current_model
    
    def get_current_tokenizer(self) -> Optional[AutoTokenizer]:
        """Get currently loaded tokenizer"""
        return self.current_tokenizer
