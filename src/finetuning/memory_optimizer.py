"""
[ELITE ARCHITECTURE] memory_optimizer.py
Low-level VRAM management and hardware-specific tuning for the GTX 1650.
"""

import torch
import gc
from loguru import logger

class MemoryOptimizer:
    """
    Ensures maximum memory efficiency during large-scale operations.
    Implements aggressive cache clearing and hardware profiling.
    """
    
    @staticmethod
    def flush():
        """Aggressively clears PyTorch CUDA cache and Python GC."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
    @staticmethod
    def get_vram_stats():
        """Returns details on current VRAM utilization."""
        if not torch.cuda.is_available():
            return {"device": "CPU", "allocated": 0, "reserved": 0}
            
        return {
            "device": torch.cuda.get_device_name(0),
            "allocated_gb": torch.cuda.memory_allocated(0) / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved(0) / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated(0) / 1024**3
        }

    @staticmethod
    def configure_memory_hacks():
        """Applies hardware-level optimizations for 4GB VRAM."""
        # trades precision for VRAM in matmul (TF32 not on 1650, but we use fp16)
        torch.backends.cuda.matmul.allow_tf32 = False 
        
        # Ensures fragmented memory is returned to the OS
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        logger.info("Memory Optimization Protocol: COMPLETED")

import os # for os.environ
