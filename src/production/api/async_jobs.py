"""
[ELITE ARCHITECTURE] async_jobs.py
Thread-safe Background Job Queue for Heavy Operations.
"""

import queue
import threading
import time
from typing import Callable, Any
from loguru import logger
import gc
import torch

class AsyncJobManager:
    """
    Innovation: Background Orchestration.
    Offloads ingestion and fine-tuning to background threads to keep the 
    Streamlit UI responsive. Includes VRAM cleanup hooks.
    """
    
    def __init__(self):
        self.job_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self.results = {}
        self.active_job_id = None

    def submit_job(self, job_id: str, func: Callable, *args, **kwargs):
        """Adds a job to the background queue."""
        logger.info(f"Job Submitted: {job_id}")
        self.job_queue.put((job_id, func, args, kwargs))
        self.results[job_id] = {"status": "pending"}

    def _worker(self):
        """Background worker loop."""
        while True:
            job_id, func, args, kwargs = self.job_queue.get()
            self.active_job_id = job_id
            self.results[job_id]["status"] = "processing"
            
            try:
                logger.debug(f"Executing Background Job: {job_id}")
                result = func(*args, **kwargs)
                self.results[job_id] = {"status": "completed", "result": result}
            except Exception as e:
                logger.error(f"Background Job Failed [{job_id}]: {e}")
                self.results[job_id] = {"status": "failed", "error": str(e)}
            finally:
                # VRAM Cleanup Hook
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.active_job_id = None
                self.job_queue.task_done()

    def get_status(self, job_id: str) -> dict:
        return self.results.get(job_id, {"status": "not_found"})

if __name__ == "__main__":
    ajm = AsyncJobManager()
    print("Async Job Manager Active.")
