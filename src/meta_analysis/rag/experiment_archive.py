"""
Experiment Archive

Searchable archive of historical experiments for benchmarking.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path
from loguru import logger

from src.meta_analysis.mcp_servers.base_experiment_mcp import StandardizedStudy


class ExperimentArchive:
    """
    Searchable archive of historical experiments.
    
    Stores analyzed experiments for future benchmarking and comparison.
    """
    
    def __init__(self, archive_path: Optional[str] = None):
        """
        Initialize archive.
        
        Args:
            archive_path: Path to archive JSON file
        """
        self.archive_path = Path(archive_path) if archive_path else None
        self._experiments: Dict[str, Dict] = {}
        self._meta_analyses: Dict[str, Dict] = {}
        
        if self.archive_path and self.archive_path.exists():
            self._load_archive()
    
    def _load_archive(self):
        """Load archive from disk."""
        try:
            with open(self.archive_path, 'r') as f:
                data = json.load(f)
                self._experiments = data.get("experiments", {})
                self._meta_analyses = data.get("meta_analyses", {})
            logger.info(f"Loaded {len(self._experiments)} experiments from archive")
        except Exception as e:
            logger.warning(f"Could not load archive: {e}")
    
    def _save_archive(self):
        """Save archive to disk."""
        if not self.archive_path:
            return
        
        try:
            self.archive_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.archive_path, 'w') as f:
                json.dump({
                    "experiments": self._experiments,
                    "meta_analyses": self._meta_analyses,
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save archive: {e}")
    
    def add_experiment(self, study: StandardizedStudy) -> str:
        """
        Add an experiment to the archive.
        
        Args:
            study: StandardizedStudy to archive
            
        Returns:
            Archive ID
        """
        archive_id = f"exp_{study.study_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self._experiments[archive_id] = {
            "study": study.to_dict(),
            "archived_at": datetime.now().isoformat()
        }
        
        self._save_archive()
        return archive_id
    
    def add_experiments(self, studies: List[StandardizedStudy]) -> List[str]:
        """Add multiple experiments."""
        return [self.add_experiment(s) for s in studies]
    
    def add_meta_analysis(
        self,
        name: str,
        result: Dict[str, Any],
        studies: List[StandardizedStudy]
    ) -> str:
        """
        Archive a meta-analysis result.
        
        Args:
            name: Name/description of the analysis
            result: MetaAnalysisResult as dict
            studies: Studies included
            
        Returns:
            Archive ID
        """
        archive_id = f"meta_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self._meta_analyses[archive_id] = {
            "name": name,
            "result": result,
            "study_ids": [s.study_id for s in studies],
            "n_studies": len(studies),
            "archived_at": datetime.now().isoformat()
        }
        
        self._save_archive()
        return archive_id
    
    def search_experiments(
        self,
        query: Optional[str] = None,
        metric_type: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        min_sample_size: Optional[int] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search archived experiments.
        
        Args:
            query: Text search query
            metric_type: Filter by metric type
            date_from: Filter by date range start
            date_to: Filter by date range end
            min_sample_size: Minimum sample size filter
            limit: Maximum results
            
        Returns:
            List of matching experiments
        """
        results = []
        
        for archive_id, data in self._experiments.items():
            study = data["study"]
            
            # Apply filters
            if query:
                query_lower = query.lower()
                if (query_lower not in study.get("study_name", "").lower() and
                    query_lower not in study.get("metric_name", "").lower()):
                    continue
            
            if metric_type:
                if study.get("metric_type", "") != metric_type:
                    continue
            
            if min_sample_size:
                total_n = study.get("sample_size_control", 0) + study.get("sample_size_treatment", 0)
                if total_n < min_sample_size:
                    continue
            
            results.append({
                "archive_id": archive_id,
                **study,
                "archived_at": data["archived_at"]
            })
        
        return results[:limit]
    
    def get_benchmark_data(self, metric_name: str) -> Dict[str, Any]:
        """
        Get benchmark statistics for a specific metric.
        
        Args:
            metric_name: Metric name to benchmark
            
        Returns:
            Benchmark statistics
        """
        matching_effects = []
        matching_sample_sizes = []
        
        for data in self._experiments.values():
            study = data["study"]
            if study.get("metric_name", "").lower() == metric_name.lower():
                matching_effects.append(study.get("effect_size", 0))
                matching_sample_sizes.append(
                    study.get("sample_size_control", 0) + study.get("sample_size_treatment", 0)
                )
        
        if not matching_effects:
            return {"found": False, "metric": metric_name}
        
        import numpy as np
        return {
            "found": True,
            "metric": metric_name,
            "n_experiments": len(matching_effects),
            "historical_effect": float(np.mean(matching_effects)),
            "effect_std": float(np.std(matching_effects)),
            "effect_min": float(np.min(matching_effects)),
            "effect_max": float(np.max(matching_effects)),
            "median_sample_size": float(np.median(matching_sample_sizes))
        }
    
    def get_meta_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent meta-analysis history.
        
        Args:
            limit: Maximum results
            
        Returns:
            List of archived meta-analyses
        """
        results = [
            {"archive_id": aid, **data}
            for aid, data in self._meta_analyses.items()
        ]
        
        # Sort by date descending
        results.sort(key=lambda x: x.get("archived_at", ""), reverse=True)
        return results[:limit]
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get archive statistics."""
        return {
            "n_experiments": len(self._experiments),
            "n_meta_analyses": len(self._meta_analyses)
        }
