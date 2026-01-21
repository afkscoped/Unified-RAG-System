import torch
import pynvml
import time
from loguru import logger

class MemoryProfiler:
    """
    Utility for real-time monitoring of CUDA VRAM usage.
    Ensures safe operations on low-VRAM hardware (e.g., GTX 1650 4GB).
    """
    
    def __init__(self, device_id: int = 0, safety_limit_gb: float = 3.8):
        self.device_id = device_id
        self.safety_limit_bytes = safety_limit_gb * 1024**3
        
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self.has_nvml = True
        except Exception as e:
            logger.warning(f"NVML initialization failed: {e}. Falling back to torch.cuda.")
            self.has_nvml = False

    def get_vram_usage(self) -> dict:
        """Returns current VRAM usage in bytes."""
        if self.has_nvml:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            used = info.used
            total = info.total
        else:
            used = torch.cuda.memory_allocated(self.device_id)
            total = torch.cuda.get_device_properties(self.device_id).total_memory
            
        return {
            "used": used,
            "total": total,
            "used_gb": used / 1024**3,
            "total_gb": total / 1024**3,
            "percent": (used / total) * 100
        }

    def check_safety(self) -> bool:
        """Checks if current usage is below the safety limit."""
        usage = self.get_vram_usage()
        if usage["used"] > self.safety_limit_bytes:
            logger.error(f"VRAM CRITICAL: {usage['used_gb']:.2f}GB / {usage['total_gb']:.2f}GB used. Safety limit is {self.safety_limit_bytes / 1024**3:.2f}GB.")
            return False
        return True

    def log_status(self, prefix: str = "Memory Status"):
        """Logs the current memory usage."""
        usage = self.get_vram_usage()
        logger.info(f"{prefix} | VRAM: {usage['used_gb']:.2f}/{usage['total_gb']:.2f} GB ({usage['percent']:.1f}%)")

if __name__ == "__main__":
    # Test utility
    profiler = MemoryProfiler()
    while True:
        profiler.log_status()
        time.sleep(5)
