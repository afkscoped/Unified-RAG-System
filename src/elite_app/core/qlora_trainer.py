"""
QLoRA Trainer - Production-grade fine-tuning engine
Optimized for 4GB VRAM with real-time monitoring
"""

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
from pathlib import Path
import time
from typing import Optional, Dict, Callable, List
from dataclasses import dataclass, field
import json


@dataclass
class TrainingConfig:
    """Training configuration"""
    output_dir: str = "./data/fine_tuned_models"
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 100
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])


class ProgressCallback(TrainerCallback):
    """Callback for tracking training progress"""
    
    def __init__(self, callback_fn: Optional[Callable] = None):
        self.callback = callback_fn
        self.metrics_history = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            metrics_entry = {
                'step': state.global_step,
                'loss': logs.get('loss', 0),
                'learning_rate': logs.get('learning_rate', 0),
                'epoch': logs.get('epoch', 0)
            }
            self.metrics_history.append(metrics_entry)
            
            if self.callback:
                self.callback(state.global_step, logs.get('loss', 0), logs)


class QLoRATrainer:
    """
    Hardware-optimized QLoRA training with live telemetry
    """
    
    def __init__(self, model, tokenizer, config: Optional[TrainingConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TrainingConfig()
        
        self.trainer = None
        self.peft_model = None
        self.training_active = False
        self.training_metrics = []
        self.progress_callback = None
        
    def prepare_model_for_training(self) -> None:
        """Apply LoRA adapters to base model"""
        print("Preparing model for QLoRA training...")
        
        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Wrap model with LoRA
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        
        print(f"LoRA adapters applied")
        print(f"   Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"   Total params: {total_params:,}")
    
    def train(
        self,
        train_dataset: Dataset,
        progress_callback: Optional[Callable] = None
    ):
        """
        Execute training with real-time monitoring
        
        Args:
            train_dataset: Formatted training dataset
            progress_callback: Function called with (step, loss, metrics)
        """
        # Prepare model if not already done
        if self.peft_model is None:
            self.prepare_model_for_training()
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length"
            )
        
        tokenized_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            fp16=torch.cuda.is_available(),
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
            max_grad_norm=0.3,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="cosine",
            disable_tqdm=False,
            report_to="none",
            save_strategy="steps"
        )
        
        # Setup progress callback
        self.progress_callback = ProgressCallback(progress_callback)
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            callbacks=[self.progress_callback]
        )
        
        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Train
        print("Starting training...")
        self.training_active = True
        start_time = time.time()
        
        try:
            train_result = self.trainer.train()
            
            training_time = time.time() - start_time
            print(f"Training complete in {training_time:.2f}s")
            print(f"   Final loss: {train_result.training_loss:.4f}")
            
            # Store metrics
            self.training_metrics = self.progress_callback.metrics_history
            
            return train_result
        
        finally:
            self.training_active = False
    
    def save_model(self, output_path: str) -> None:
        """Save fine-tuned LoRA adapters"""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapters
        self.peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training config
        config_path = output_dir / "training_config.json"
        config_dict = {
            "output_dir": self.config.output_dir,
            "num_epochs": self.config.num_epochs,
            "batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "max_seq_length": self.config.max_seq_length,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "warmup_ratio": self.config.warmup_ratio,
            "logging_steps": self.config.logging_steps,
            "save_steps": self.config.save_steps,
            "target_modules": self.config.target_modules
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save training metrics if available
        if self.training_metrics:
            metrics_path = output_dir / "training_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.training_metrics, f, indent=2)
        
        print(f"Model saved to {output_dir}")
    
    def get_vram_usage(self) -> float:
        """Get current VRAM usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0
    
    def get_training_metrics(self) -> List[Dict]:
        """Get training metrics history"""
        if self.progress_callback:
            return self.progress_callback.metrics_history
        return self.training_metrics
