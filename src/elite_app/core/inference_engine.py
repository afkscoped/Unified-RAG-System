"""
Inference Engine - Test fine-tuned models interactively
"""

import torch
from peft import PeftModel
from transformers import GenerationConfig
from typing import List, Dict, Optional


class InferenceEngine:
    """Test fine-tuned models with interactive generation"""
    
    def __init__(self, base_model, tokenizer, adapter_path: str = None):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.model = base_model
        self.adapter_loaded = False
        
        if adapter_path:
            self.load_adapter(adapter_path)
    
    def load_adapter(self, adapter_path: str) -> None:
        """Load fine-tuned LoRA adapter"""
        print(f"Loading adapter from {adapter_path}...")
        self.model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.adapter_loaded = True
        print("Adapter loaded")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.1
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            repetition_penalty: Penalty for repetition
        
        Returns:
            Generated text
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generation config
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """
        Chat interface
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Assistant response
        """
        # Build prompt from chat history
        prompt = self._format_chat_prompt(messages)
        
        # Generate
        response = self.generate(
            prompt, 
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        return response
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into prompt"""
        prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt += f"### Instruction:\n{msg['content']}\n\n"
            else:
                prompt += f"### Response:\n{msg['content']}\n\n"
        
        # Add response prompt
        prompt += "### Response:\n"
        
        return prompt
    
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        callback: Optional[callable] = None
    ) -> str:
        """
        Generate text with streaming output
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            callback: Function called with each new token
        
        Returns:
            Complete generated text
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        generated_text = ""
        input_length = inputs["input_ids"].shape[1]
        
        # Generate token by token
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(**inputs)
                
                # Get logits for last token
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Decode token
                token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                generated_text += token_text
                
                # Callback
                if callback:
                    callback(token_text)
                
                # Update inputs
                inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=-1)
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = torch.cat([
                        inputs["attention_mask"],
                        torch.ones((1, 1), device=inputs["attention_mask"].device)
                    ], dim=-1)
        
        return generated_text
    
    def is_adapter_loaded(self) -> bool:
        """Check if adapter is loaded"""
        return self.adapter_loaded
