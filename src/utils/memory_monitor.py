"""
Memory Monitor

Tracks RAM and GPU memory usage with auto-cleanup.
"""

import gc
import psutil
from typing import Dict, Optional
from loguru import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False


class MemoryMonitor:
    """Monitors and manages system memory."""
    
    def __init__(
        self,
        max_ram_gb: float = 12.0,
        max_vram_gb: float = 4.8,
        ram_threshold: float = 0.80,
        vram_threshold: float = 0.85
    ):
        """
        Initialize memory monitor.
        
        Args:
            max_ram_gb: Maximum RAM to use
            max_vram_gb: Maximum VRAM to use
            ram_threshold: Trigger cleanup when RAM exceeds this %
            vram_threshold: Trigger cleanup when VRAM exceeds this %
        """
        self.max_ram_gb = max_ram_gb
        self.max_vram_gb = max_vram_gb
        self.ram_threshold = ram_threshold
        self.vram_threshold = vram_threshold
        
        self._cleanup_count = 0
        
    def get_ram_usage(self) -> Dict[str, float]:
        """Get current RAM usage."""
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "used_gb": mem.used / (1024**3),
            "available_gb": mem.available / (1024**3),
            "percent": mem.percent / 100.0
        }
    
    def get_vram_usage(self) -> Optional[Dict[str, float]]:
        """Get current VRAM usage (if GPU available)."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None
            
        try:
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            
            if PYNVML_AVAILABLE:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total = info.total / (1024**3)
                used = info.used / (1024**3)
            else:
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                used = reserved
                
            return {
                "total_gb": total,
                "used_gb": used,
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "percent": used / total if total > 0 else 0
            }
        except Exception as e:
            logger.warning(f"Failed to get VRAM usage: {e}")
            return None
    
    def check_and_cleanup(self) -> bool:
        """
        Check memory and cleanup if needed.
        
        Returns:
            True if cleanup was performed
        """
        cleaned = False
        
        # Check RAM
        ram = self.get_ram_usage()
        if ram["percent"] > self.ram_threshold:
            gc.collect()
            cleaned = True
            logger.warning(f"RAM cleanup triggered: {ram['percent']:.1%} used")
        
        # Check VRAM
        vram = self.get_vram_usage()
        if vram and vram["percent"] > self.vram_threshold:
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()
            cleaned = True
            logger.warning(f"VRAM cleanup triggered: {vram['percent']:.1%} used")
            
        if cleaned:
            self._cleanup_count += 1
            
        return cleaned
    
    def get_status(self) -> Dict:
        """Get full memory status."""
        status = {
            "ram": self.get_ram_usage(),
            "cleanup_count": self._cleanup_count
        }
        
        vram = self.get_vram_usage()
        if vram:
            status["vram"] = vram
            
        return status
    
    def log_status(self):
        """Log current memory status."""
        ram = self.get_ram_usage()
        logger.info(f"RAM: {ram['used_gb']:.1f}/{ram['total_gb']:.1f} GB ({ram['percent']:.1%})")
        
        vram = self.get_vram_usage()
        if vram:
            logger.info(f"VRAM: {vram['used_gb']:.1f}/{vram['total_gb']:.1f} GB ({vram['percent']:.1%})")

