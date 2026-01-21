"""
[ELITE ARCHITECTURE] model_registry.py
Tracks system versions and LoRA adapter states.
"""

import json
import os
from datetime import datetime
from loguru import logger

class ModelRegistry:
    """
    Innovation: Experiment Provenance.
    Tracks all fine-tuned adapters, their base models, and performance 
    metrics to ensure reproducible research.
    """
    
    def __init__(self, registry_path: str = "data/finetuning/registry.json"):
        self.registry_path = registry_path
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        self.registry = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {"adapters": [], "active_adapter": None}

    def register_adapter(self, adapter_id: str, base_model: str, path: str, metrics: dict = None):
        """Adds a new LoRA adapter to the registry."""
        entry = {
            "id": adapter_id,
            "base_model": base_model,
            "path": path,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics or {}
        }
        self.registry["adapters"].append(entry)
        self._save()
        logger.info(f"Registry: Registered new adapter {adapter_id}")

    def set_active(self, adapter_id: str):
        if any(a["id"] == adapter_id for a in self.registry["adapters"]):
            self.registry["active_adapter"] = adapter_id
            self._save()
            logger.success(f"Registry: Switched active adapter to {adapter_id}")
        else:
            logger.error(f"Registry: Adapter {adapter_id} not found.")

    def _save(self):
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=4)

if __name__ == "__main__":
    reg = ModelRegistry()
    print("Model Registry Ready.")
