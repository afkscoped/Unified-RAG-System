"""
Metrics Logger - Training metrics logging and persistence
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import json
import csv


@dataclass
class TrainingMetric:
    """Single training metric entry"""
    step: int
    loss: float
    learning_rate: float
    epoch: float
    timestamp: str
    vram_gb: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class MetricsLogger:
    """
    Log and persist training metrics
    Supports JSON and CSV export
    """
    
    def __init__(self, log_dir: str = "./data/training_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics: List[TrainingMetric] = []
        self.run_id: Optional[str] = None
        self.run_start: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}
    
    def start_run(self, run_name: str = None, metadata: Dict = None):
        """Start a new logging run"""
        self.run_start = datetime.now()
        self.run_id = run_name or f"run_{self.run_start.strftime('%Y%m%d_%H%M%S')}"
        self.metrics = []
        self.metadata = metadata or {}
        self.metadata['run_id'] = self.run_id
        self.metadata['start_time'] = self.run_start.isoformat()
    
    def log_metric(
        self,
        step: int,
        loss: float,
        learning_rate: float = 0.0,
        epoch: float = 0.0,
        vram_gb: float = None,
        **extra
    ):
        """Log a single metric entry"""
        metric = TrainingMetric(
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            epoch=epoch,
            timestamp=datetime.now().isoformat(),
            vram_gb=vram_gb
        )
        self.metrics.append(metric)
        
        # Store extra fields in metadata
        if extra:
            if 'extra_metrics' not in self.metadata:
                self.metadata['extra_metrics'] = []
            self.metadata['extra_metrics'].append({
                'step': step,
                **extra
            })
    
    def log_batch(self, metrics: List[Dict]):
        """Log multiple metrics at once"""
        for m in metrics:
            self.log_metric(**m)
    
    def get_metrics(self) -> List[Dict]:
        """Get all logged metrics"""
        return [m.to_dict() for m in self.metrics]
    
    def get_latest(self) -> Optional[Dict]:
        """Get most recent metric"""
        if self.metrics:
            return self.metrics[-1].to_dict()
        return None
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.metrics:
            return {
                "total_steps": 0,
                "min_loss": None,
                "max_loss": None,
                "avg_loss": None,
                "final_loss": None
            }
        
        losses = [m.loss for m in self.metrics]
        
        return {
            "run_id": self.run_id,
            "total_steps": len(self.metrics),
            "min_loss": min(losses),
            "max_loss": max(losses),
            "avg_loss": sum(losses) / len(losses),
            "final_loss": losses[-1],
            "start_time": self.metadata.get('start_time'),
            "duration_seconds": (datetime.now() - self.run_start).total_seconds() if self.run_start else 0
        }
    
    def save_json(self, filepath: str = None) -> str:
        """Save metrics to JSON file"""
        if filepath is None:
            filepath = self.log_dir / f"{self.run_id}_metrics.json"
        else:
            filepath = Path(filepath)
        
        data = {
            "metadata": self.metadata,
            "summary": self.get_summary(),
            "metrics": self.get_metrics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)
    
    def save_csv(self, filepath: str = None) -> str:
        """Save metrics to CSV file"""
        if filepath is None:
            filepath = self.log_dir / f"{self.run_id}_metrics.csv"
        else:
            filepath = Path(filepath)
        
        if not self.metrics:
            return str(filepath)
        
        fieldnames = ['step', 'loss', 'learning_rate', 'epoch', 'timestamp', 'vram_gb']
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in self.metrics:
                writer.writerow(m.to_dict())
        
        return str(filepath)
    
    def load_json(self, filepath: str) -> bool:
        """Load metrics from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.metadata = data.get('metadata', {})
            self.run_id = self.metadata.get('run_id')
            
            metrics_data = data.get('metrics', [])
            self.metrics = [
                TrainingMetric(**m) for m in metrics_data
            ]
            
            return True
        except Exception:
            return False
    
    def clear(self):
        """Clear all logged metrics"""
        self.metrics = []
        self.metadata = {}
        self.run_id = None
        self.run_start = None
    
    def end_run(self) -> Dict:
        """End the current run and return summary"""
        summary = self.get_summary()
        
        # Auto-save
        if self.run_id:
            self.save_json()
            self.save_csv()
        
        return summary
