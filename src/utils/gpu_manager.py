"""
GPU Manager

Handles GPU device allocation and VRAM management.
"""

from typing import Optional, Dict
from loguru import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GPUManager:
    """Manages GPU resources and device allocation."""
    
    def __init__(self, device_id: int = 0, max_memory_fraction: float = 0.8):
        """
        Initialize GPU manager.
        
        Args:
            device_id: CUDA device index
            max_memory_fraction: Max fraction of VRAM to use (0-1)
        """
        self.device_id = device_id
        self.max_memory_fraction = max_memory_fraction
        self._initialized = False
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self._initialize_gpu()
        else:
            logger.warning("CUDA not available, running on CPU")
            
    def _initialize_gpu(self):
        """Initialize GPU settings."""
        try:
            torch.cuda.set_device(self.device_id)
            torch.cuda.set_per_process_memory_fraction(
                self.max_memory_fraction, 
                self.device_id
            )
            
            # Enable TF32 for better performance on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            self._initialized = True
            
            props = torch.cuda.get_device_properties(self.device_id)
            logger.info(
                f"GPU initialized: {props.name} "
                f"({props.total_memory / 1024**3:.1f} GB, "
                f"max {self.max_memory_fraction:.0%})"
            )
        except Exception as e:
            logger.error(f"GPU initialization failed: {e}")
            
    @property
    def device(self) -> str:
        """Get device string for torch."""
        if self._initialized:
            return f"cuda:{self.device_id}"
        return "cpu"
    
    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return self._initialized
    
    def get_memory_info(self) -> Optional[Dict]:
        """Get GPU memory information."""
        if not self._initialized:
            return None
            
        return {
            "allocated_gb": torch.cuda.memory_allocated(self.device_id) / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved(self.device_id) / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated(self.device_id) / 1024**3
        }
    
    def empty_cache(self):
        """Clear GPU cache."""
        if self._initialized:
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
            
    def reset_peak_stats(self):
        """Reset peak memory tracking."""
        if self._initialized:
            torch.cuda.reset_peak_memory_stats(self.device_id)

