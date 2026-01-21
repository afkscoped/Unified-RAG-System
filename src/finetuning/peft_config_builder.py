"""
[ELITE ARCHITECTURE] peft_config_builder.py
Configures Parameter-Efficient Fine-Tuning (PEFT) adapters.
Targeting: All linear layers for maximum adaptation.
"""

import torch
from loguru import logger
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

class PEFTConfigBuilder:
    """
    Orchestrates the conversion of base models into PEFT-ready specialized models.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.ft_config = config.get('finetuning', {})
        self.lora_config = self.ft_config.get('lora', {})

    def prepare_model(self, model):
        """
        Wraps the model for K-bit training and attaches LoRA adapters.
        """
        logger.info("Initializing PEFT Model Preparation Sequence...")
        
        # 1. Gradient Checkpointing & K-Bit Preparation
        model = prepare_model_for_kbit_training(model)
        
        # 2. Target Modules Selection
        # [INNOVATION] For Phi-3 and Mistral, targeting 'all linear layers' 
        # (q, k, v, o, gate, up, down) usually yields superior domain adaptation.
        target_modules = self.lora_config.get('target_modules', 
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        
        # 3. LoRA Hyperparameters
        # r (Rank): Number of trainable parameters per layer.
        # alpha: Scaling factor for the weights.
        config = LoraConfig(
            r=self.lora_config.get('r', 16),
            lora_alpha=self.lora_config.get('alpha', 32),
            target_modules=target_modules,
            lora_dropout=self.lora_config.get('dropout', 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # 4. Wrap Model
        peft_model = get_peft_model(model, config)
        
        # Log parameter counts for verification
        peft_model.print_trainable_parameters()
        
        return peft_model
