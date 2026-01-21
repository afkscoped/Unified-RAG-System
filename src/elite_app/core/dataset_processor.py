"""
Dataset Processor - Handles training data preparation
Supports multiple formats and auto-generation
"""

from typing import List, Dict, Optional
from pathlib import Path
import json
from datasets import Dataset
import pandas as pd
from dataclasses import dataclass


@dataclass
class DatasetStatistics:
    """Statistics about a dataset"""
    total_samples: int
    avg_instruction_length: int
    avg_output_length: int
    format_valid: bool
    issues: List[str]


class DatasetProcessor:
    """
    Process and validate training datasets
    Supports multiple formats and auto-generation
    """
    
    def __init__(self):
        self.current_dataset = None
        self.dataset_stats = None
    
    def load_from_jsonl(self, file_path: str) -> Dataset:
        """
        Load dataset from JSONL file
        Expected format: {"instruction": "...", "output": "..."}
        """
        data = []
        issues = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    
                    # Validate required fields
                    if "instruction" not in item or "output" not in item:
                        issues.append(f"Line {line_num}: Missing required fields")
                        continue
                    
                    data.append(item)
                
                except json.JSONDecodeError:
                    issues.append(f"Line {line_num}: Invalid JSON")
        
        if not data:
            raise ValueError("No valid data found in file")
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(data)
        
        # Calculate statistics
        self.dataset_stats = self._calculate_statistics(data, issues)
        self.current_dataset = dataset
        
        return dataset
    
    def load_from_csv(self, file_path: str, instruction_col: str = "instruction", output_col: str = "output") -> Dataset:
        """Load dataset from CSV"""
        df = pd.read_csv(file_path)
        
        # Validate columns
        if instruction_col not in df.columns or output_col not in df.columns:
            raise ValueError(f"CSV must have '{instruction_col}' and '{output_col}' columns")
        
        # Convert to dict list
        data = [
            {"instruction": row[instruction_col], "output": row[output_col]}
            for _, row in df.iterrows()
        ]
        
        dataset = Dataset.from_list(data)
        self.dataset_stats = self._calculate_statistics(data, [])
        self.current_dataset = dataset
        
        return dataset
    
    def load_from_dict_list(self, data: List[Dict]) -> Dataset:
        """Load dataset from list of dictionaries"""
        # Validate data
        issues = []
        valid_data = []
        
        for i, item in enumerate(data):
            if "instruction" not in item or "output" not in item:
                issues.append(f"Item {i}: Missing required fields")
                continue
            valid_data.append(item)
        
        if not valid_data:
            raise ValueError("No valid data found")
        
        dataset = Dataset.from_list(valid_data)
        self.dataset_stats = self._calculate_statistics(valid_data, issues)
        self.current_dataset = dataset
        
        return dataset
    
    def generate_qa_dataset(
        self,
        text_chunks: List[str],
        num_samples: int = 100
    ) -> Dataset:
        """
        Auto-generate Q&A dataset from text chunks
        Uses simple extraction (can be enhanced with LLM)
        """
        data = []
        
        for i, chunk in enumerate(text_chunks[:num_samples]):
            if not chunk.strip():
                continue
                
            # Simple Q&A generation
            instruction = f"Explain the key points from the following text: {chunk[:100]}..."
            output = chunk
            
            data.append({
                "instruction": instruction,
                "output": output
            })
        
        if not data:
            raise ValueError("Could not generate any samples from text")
        
        dataset = Dataset.from_list(data)
        self.dataset_stats = self._calculate_statistics(data, [])
        self.current_dataset = dataset
        
        return dataset
    
    def format_for_training(
        self,
        dataset: Dataset = None,
        prompt_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{output}"
    ) -> Dataset:
        """
        Format dataset with instruction template
        
        Args:
            dataset: Input dataset (uses current if None)
            prompt_template: Template with {instruction} and {output} placeholders
        """
        if dataset is None:
            dataset = self.current_dataset
        
        if dataset is None:
            raise ValueError("No dataset loaded")
        
        def format_sample(example):
            text = prompt_template.format(
                instruction=example["instruction"],
                output=example["output"]
            )
            return {"text": text}
        
        formatted = dataset.map(format_sample, remove_columns=dataset.column_names)
        return formatted
    
    def _calculate_statistics(self, data: List[Dict], issues: List[str]) -> DatasetStatistics:
        """Calculate dataset statistics"""
        total = len(data)
        
        if total == 0:
            return DatasetStatistics(
                total_samples=0,
                avg_instruction_length=0,
                avg_output_length=0,
                format_valid=False,
                issues=issues + ["No valid samples"]
            )
        
        avg_inst_len = sum(len(str(d.get("instruction", "")).split()) for d in data) / total
        avg_out_len = sum(len(str(d.get("output", "")).split()) for d in data) / total
        
        return DatasetStatistics(
            total_samples=total,
            avg_instruction_length=int(avg_inst_len),
            avg_output_length=int(avg_out_len),
            format_valid=len(issues) == 0,
            issues=issues
        )
    
    def validate_dataset(self, dataset: Dataset = None) -> DatasetStatistics:
        """Validate dataset format and quality"""
        if dataset is None and self.dataset_stats is not None:
            return self.dataset_stats
        
        if dataset is not None:
            data = [item for item in dataset]
            return self._calculate_statistics(data, [])
        
        return DatasetStatistics(
            total_samples=0,
            avg_instruction_length=0,
            avg_output_length=0,
            format_valid=False,
            issues=["No dataset loaded"]
        )
    
    def save_dataset(self, dataset: Dataset, output_path: str):
        """Save dataset to JSONL"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                json.dump(item, f)
                f.write('\n')
    
    def get_sample(self, index: int = 0) -> Optional[Dict]:
        """Get a sample from the current dataset"""
        if self.current_dataset is None:
            return None
        if index >= len(self.current_dataset):
            return None
        return self.current_dataset[index]
    
    def get_samples(self, num_samples: int = 5) -> List[Dict]:
        """Get multiple samples from the current dataset"""
        if self.current_dataset is None:
            return []
        return [self.current_dataset[i] for i in range(min(num_samples, len(self.current_dataset)))]
