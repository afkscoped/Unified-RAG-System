"""
VRAM Monitor - Real-time GPU memory tracking
"""

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import threading
import time


@dataclass
class VRAMSnapshot:
    """Single VRAM measurement"""
    timestamp: datetime
    allocated_gb: float
    reserved_gb: float
    total_gb: float
    percent_used: float


class VRAMMonitor:
    """
    Real-time VRAM monitoring with statistics
    """
    
    def __init__(self):
        self.history: List[VRAMSnapshot] = []
        self.is_monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Check GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            self.gpu_name = "No GPU"
            self.total_vram = 0.0
    
    def get_current_usage(self) -> VRAMSnapshot:
        """Get current VRAM usage"""
        if not self.gpu_available:
            return VRAMSnapshot(
                timestamp=datetime.now(),
                allocated_gb=0.0,
                reserved_gb=0.0,
                total_gb=0.0,
                percent_used=0.0
            )
        
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = self.total_vram
        percent = (allocated / total * 100) if total > 0 else 0
        
        return VRAMSnapshot(
            timestamp=datetime.now(),
            allocated_gb=allocated,
            reserved_gb=reserved,
            total_gb=total,
            percent_used=percent
        )
    
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start continuous VRAM monitoring in background thread"""
        if self.is_monitoring:
            return
        
        self._stop_event.clear()
        self.is_monitoring = True
        
        def monitor_loop():
            while not self._stop_event.is_set():
                snapshot = self.get_current_usage()
                self.history.append(snapshot)
                
                # Keep only last 1000 measurements
                if len(self.history) > 1000:
                    self.history = self.history[-1000:]
                
                time.sleep(interval_seconds)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self._stop_event.set()
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
    
    def get_statistics(self) -> Dict:
        """Get statistics from monitoring history"""
        if not self.history:
            return {
                "samples": 0,
                "min_gb": 0.0,
                "max_gb": 0.0,
                "avg_gb": 0.0,
                "current_gb": 0.0
            }
        
        allocations = [s.allocated_gb for s in self.history]
        current = self.get_current_usage()
        
        return {
            "samples": len(self.history),
            "min_gb": min(allocations),
            "max_gb": max(allocations),
            "avg_gb": sum(allocations) / len(allocations),
            "current_gb": current.allocated_gb,
            "total_gb": current.total_gb,
            "percent_used": current.percent_used
        }
    
    def clear_history(self):
        """Clear monitoring history"""
        self.history = []
    
    def get_gpu_info(self) -> Dict:
        """Get GPU information"""
        if not self.gpu_available:
            return {
                "available": False,
                "name": "No GPU detected",
                "total_vram_gb": 0.0,
                "cuda_version": "N/A"
            }
        
        return {
            "available": True,
            "name": self.gpu_name,
            "total_vram_gb": self.total_vram,
            "cuda_version": torch.version.cuda or "N/A",
            "device_count": torch.cuda.device_count()
        }
    
    def clear_cache(self):
        """Clear CUDA cache to free memory"""
        if self.gpu_available:
            torch.cuda.empty_cache()
            return True
        return False
    
    def get_history_for_plot(self) -> Dict[str, List]:
        """Get history data formatted for plotting"""
        if not self.history:
            return {"timestamps": [], "allocated": [], "reserved": []}
        
        return {
            "timestamps": [s.timestamp.isoformat() for s in self.history],
            "allocated": [s.allocated_gb for s in self.history],
            "reserved": [s.reserved_gb for s in self.history]
        }
