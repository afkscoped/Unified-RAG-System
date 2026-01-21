"""
CSV Experiment MCP Server

MCP Server for CSV-based A/B test data ingestion.
Supports: Kaggle datasets, custom CSV exports, Excel files.
"""

import os
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from loguru import logger

from src.meta_analysis.mcp_servers.base_experiment_mcp import (
    ExperimentMCPBase,
    StandardizedStudy,
    ValidationReport,
    ValidationIssue,
    ExperimentFilter,
    MetricType,
    EffectSizeType
)
from src.meta_analysis.utils.effect_size_converters import (
    calculate_log_odds_ratio,
    calculate_standard_error_log_odds
)


class CSVExperimentMCP(ExperimentMCPBase):
    """
    MCP Server for CSV-based A/B test data ingestion.
    
    Handles various CSV formats commonly used for A/B test results,
    including Kaggle datasets and custom exports.
    
    Attributes:
        data_dir: Directory containing experiment CSV files
        experiments_cache: Cached parsed experiments
    """
    
    # Common column name mappings for different CSV formats
    COLUMN_MAPPINGS = {
        # Control group columns
        "control_conversions": ["control_conversions", "control_success", "conversions_control", 
                                "control_converted", "baseline_conversions", "control"],
        "control_total": ["control_total", "control_n", "control_size", "n_control",
                          "control_visitors", "baseline_total", "control_users"],
        # Treatment group columns
        "treatment_conversions": ["treatment_conversions", "treatment_success", "conversions_treatment",
                                  "treatment_converted", "variant_conversions", "treatment", "experiment"],
        "treatment_total": ["treatment_total", "treatment_n", "treatment_size", "n_treatment",
                            "treatment_visitors", "variant_total", "treatment_users", "experiment_users", 
                            "variant_visitors", "variant_users", "variant_n", "variant_size"],
        # Metadata columns
        "experiment_id": ["experiment_id", "test_id", "id", "study_id", "exp_id"],
        "experiment_name": ["experiment_name", "test_name", "name", "study_name", "exp_name", "campaign"],
        "metric": ["metric", "metric_name", "kpi", "measure"],
        "date": ["date", "start_date", "timestamp", "created_at", "run_date"],
        # Pre-calculated stats (optional)
        "effect_size": ["effect_size", "lift", "relative_lift", "percent_change"],
        "p_value": ["p_value", "pvalue", "significance", "p"]
    }
    
    def __init__(
        self, 
        data_dir: str = "./data/experiments",
        auto_detect_format: bool = True
    ):
        """
        Initialize CSV MCP Server.
        
        Args:
            data_dir: Directory containing experiment files
            auto_detect_format: Whether to auto-detect column mappings
        """
        self.data_dir = Path(data_dir)
        self.auto_detect_format = auto_detect_format
        self.experiments_cache: Dict[str, List[StandardizedStudy]] = {}
        self._file_hashes: Dict[str, str] = {}
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def list_experiment_files(self) -> List[Dict[str, Any]]:
        """
        List all experiment files in the data directory.
        
        Returns:
            List of file metadata dictionaries
        """
        files = []
        for ext in ["*.csv", "*.xlsx", "*.xls"]:
            for file_path in self.data_dir.glob(ext):
                stat = file_path.stat()
                files.append({
                    "path": str(file_path),
                    "name": file_path.name,
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "type": file_path.suffix.lower()
                })
        return files
    
    def get_file_status(self) -> List[Dict[str, Any]]:
        """
        Get status of all files in data dir, including parsing errors.
        
        Returns:
            List of status dictionaries
        """
        status = []
        for file_info in self.list_experiment_files():
            result = {
                "file": file_info["name"],
                "status": "Unknown",
                "experiments_count": 0,
                "error": None
            }
            try:
                experiments = self.parse_ab_test_csv(file_info["path"])
                result["experiments_count"] = len(experiments)
                if len(experiments) > 0:
                    result["status"] = "Success"
                else:
                    result["status"] = "Warning" 
                    result["error"] = "File parsed but contained 0 valid experiments (check column names?)"
            except Exception as e:
                result["status"] = "Error"
                result["error"] = str(e)
            
            status.append(result)
        return status
    
    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Auto-detect column mappings based on column names.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping standard names to actual column names
        """
        detected = {}
        # Ensure cols are strings to avoid 'int object has no attribute lower'
        df_columns_lower = {str(col).lower().strip(): col for col in df.columns}
        
        for standard_name, possible_names in self.COLUMN_MAPPINGS.items():
            for possible in possible_names:
                if possible.lower() in df_columns_lower:
                    detected[standard_name] = df_columns_lower[possible.lower()]
                    break
        
        return detected
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for cache invalidation."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def parse_ab_test_csv(
        self, 
        file_path: Union[str, Path],
        column_config: Optional[Dict[str, str]] = None,
        chunk_size: int = 10000
    ) -> List[StandardizedStudy]:
        """
        Parse CSV file into standardized experiments.
        
        Args:
            file_path: Path to CSV/Excel file
            column_config: Optional manual column mapping
            chunk_size: Rows to process at a time (for large files)
            
        Returns:
            List of standardized studies
        """
        file_path = Path(file_path)
        
        # Check cache
        file_hash = self._calculate_file_hash(file_path)
        cache_key = f"{file_path}:{file_hash}"
        if cache_key in self.experiments_cache:
            logger.debug(f"Returning cached experiments for {file_path.name}")
            return self.experiments_cache[cache_key]
        
        # Read file
        try:
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                # Use chunked reading for large CSVs
                df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise ValueError(f"Could not read file: {e}")
        
        # Detect or use provided column mapping
        if column_config:
            columns = column_config
        elif self.auto_detect_format:
            columns = self._detect_columns(df)
        else:
            raise ValueError("Column configuration required when auto_detect_format is False")
        
        # Validate required columns exist
        required = ["control_conversions", "control_total", "treatment_conversions", "treatment_total"]
        missing = [col for col in required if col not in columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Detected columns: {list(columns.keys())}")
        
        # Parse experiments
        experiments = []
        for idx, row in df.iterrows():
            try:
                study = self._row_to_study(row, columns, idx, file_path.stem)
                if study:
                    experiments.append(study)
            except Exception as e:
                logger.warning(f"Failed to parse row {idx}: {e}")
                continue
        
        # Cache results
        self.experiments_cache[cache_key] = experiments
        self._file_hashes[str(file_path)] = file_hash
        
        logger.info(f"Parsed {len(experiments)} experiments from {file_path.name}")
        return experiments
    
    def _row_to_study(
        self, 
        row: pd.Series, 
        columns: Dict[str, str],
        idx: int,
        file_stem: str
    ) -> Optional[StandardizedStudy]:
        """Convert a DataFrame row to a StandardizedStudy."""
        
        # Extract counts
        control_conv = int(row[columns["control_conversions"]])
        control_total = int(row[columns["control_total"]])
        treatment_conv = int(row[columns["treatment_conversions"]])
        treatment_total = int(row[columns["treatment_total"]])
        
        # Validate basic requirements
        if control_total <= 0 or treatment_total <= 0:
            return None
        if control_conv < 0 or treatment_conv < 0:
            return None
        if control_conv > control_total or treatment_conv > treatment_total:
            return None
        
        # Calculate effect size (log odds ratio)
        effect_size = calculate_log_odds_ratio(
            control_conv, control_total - control_conv,
            treatment_conv, treatment_total - treatment_conv
        )
        
        # Calculate standard error
        standard_error = calculate_standard_error_log_odds(
            control_conv, control_total - control_conv,
            treatment_conv, treatment_total - treatment_conv
        )
        
        # Handle infinite/NaN values
        if not np.isfinite(effect_size) or not np.isfinite(standard_error):
            logger.warning(f"Non-finite effect size or SE at row {idx}")
            return None
        
        # Extract optional fields
        study_id = str(row.get(columns.get("experiment_id", ""), f"{file_stem}_{idx}"))
        study_name = str(row.get(columns.get("experiment_name", ""), f"Experiment {idx}"))
        metric_name = str(row.get(columns.get("metric", ""), "conversion"))
        
        # Parse date if available
        timestamp = None
        if "date" in columns and pd.notna(row.get(columns["date"])):
            try:
                timestamp = pd.to_datetime(row[columns["date"]])
            except:
                pass
        
        # Extract p-value if available
        p_value = None
        if "p_value" in columns and pd.notna(row.get(columns.get("p_value"))):
            try:
                p_value = float(row[columns["p_value"]])
            except:
                pass
        
        return StandardizedStudy(
            study_id=study_id,
            study_name=study_name,
            effect_size=effect_size,
            standard_error=standard_error,
            sample_size_control=control_total,
            sample_size_treatment=treatment_total,
            metric_name=metric_name,
            metric_type=MetricType.CONVERSION,
            effect_size_type=EffectSizeType.LOG_ODDS_RATIO,
            platform="csv",
            timestamp=timestamp,
            p_value=p_value,
            metadata={
                "source_file": file_stem,
                "row_index": idx,
                "control_conversions": control_conv,
                "treatment_conversions": treatment_conv
            }
        )
    
    def list_experiments(
        self, 
        filters: Optional[ExperimentFilter] = None
    ) -> List[Dict[str, Any]]:
        """List all available experiments from cached files."""
        all_experiments = []
        
        # Parse all files if not cached
        for file_info in self.list_experiment_files():
            try:
                experiments = self.parse_ab_test_csv(file_info["path"])
                for exp in experiments:
                    all_experiments.append({
                        "study_id": exp.study_id,
                        "study_name": exp.study_name,
                        "metric_name": exp.metric_name,
                        "sample_size": exp.total_sample_size,
                        "effect_size": exp.effect_size,
                        "timestamp": exp.timestamp.isoformat() if exp.timestamp else None,
                        "source_file": file_info["name"]
                    })
            except Exception as e:
                logger.warning(f"Could not parse {file_info['name']}: {e}")
        
        # Apply filters
        if filters:
            all_experiments = self._apply_filters(all_experiments, filters)
        
        return all_experiments
    
    def _apply_filters(
        self, 
        experiments: List[Dict], 
        filters: ExperimentFilter
    ) -> List[Dict]:
        """Apply filters to experiment list."""
        result = experiments
        
        if filters.date_from:
            result = [e for e in result if e.get("timestamp") and 
                      datetime.fromisoformat(e["timestamp"]) >= filters.date_from]
        
        if filters.date_to:
            result = [e for e in result if e.get("timestamp") and 
                      datetime.fromisoformat(e["timestamp"]) <= filters.date_to]
        
        if filters.min_sample_size:
            result = [e for e in result if e.get("sample_size", 0) >= filters.min_sample_size]
        
        if filters.search_query:
            query = filters.search_query.lower()
            result = [e for e in result if query in e.get("study_name", "").lower() or 
                      query in e.get("metric_name", "").lower()]
        
        return result
    
    def get_experiment_details(
        self, 
        experiment_id: str
    ) -> Optional[StandardizedStudy]:
        """Get a specific experiment by ID."""
        for file_info in self.list_experiment_files():
            try:
                experiments = self.parse_ab_test_csv(file_info["path"])
                for exp in experiments:
                    if exp.study_id == experiment_id:
                        return exp
            except:
                continue
        return None
    
    def get_experiments(
        self, 
        experiment_ids: Optional[List[str]] = None,
        filters: Optional[ExperimentFilter] = None
    ) -> List[StandardizedStudy]:
        """Get multiple experiments."""
        all_experiments = []
        
        for file_info in self.list_experiment_files():
            try:
                experiments = self.parse_ab_test_csv(file_info["path"])
                all_experiments.extend(experiments)
            except Exception as e:
                logger.warning(f"Could not parse {file_info['name']}: {e}")
        
        # Filter by IDs if provided
        if experiment_ids:
            all_experiments = [e for e in all_experiments if e.study_id in experiment_ids]
        
        # Apply additional filters
        if filters:
            if filters.min_sample_size:
                all_experiments = [e for e in all_experiments 
                                   if e.total_sample_size >= filters.min_sample_size]
            if filters.date_from:
                all_experiments = [e for e in all_experiments 
                                   if e.timestamp and e.timestamp >= filters.date_from]
            if filters.date_to:
                all_experiments = [e for e in all_experiments 
                                   if e.timestamp and e.timestamp <= filters.date_to]
        
        return all_experiments
    
    def validate_data(
        self, 
        experiments: List[StandardizedStudy]
    ) -> ValidationReport:
        """Validate experiment data quality."""
        report = ValidationReport(
            is_valid=True,
            total_records=len(experiments),
            valid_records=0
        )
        
        valid_count = 0
        effect_sizes = []
        sample_sizes = []
        
        for exp in experiments:
            is_valid = True
            
            # Check effect size bounds
            if abs(exp.effect_size) > 5:
                report.add_issue(
                    "effect_size", "warning",
                    f"Unusually large effect size: {exp.effect_size:.3f}",
                    exp.effect_size
                )
            
            # Check standard error
            if exp.standard_error <= 0:
                report.add_issue(
                    "standard_error", "error",
                    f"Invalid standard error: {exp.standard_error}",
                    exp.standard_error
                )
                is_valid = False
            
            # Check sample sizes
            if exp.sample_size_control < 10 or exp.sample_size_treatment < 10:
                report.add_issue(
                    "sample_size", "warning",
                    f"Small sample size: control={exp.sample_size_control}, treatment={exp.sample_size_treatment}",
                    exp.total_sample_size
                )
            
            if is_valid:
                valid_count += 1
                effect_sizes.append(exp.effect_size)
                sample_sizes.append(exp.total_sample_size)
        
        report.valid_records = valid_count
        
        # Summary statistics
        if effect_sizes:
            report.summary = {
                "mean_effect_size": float(np.mean(effect_sizes)),
                "std_effect_size": float(np.std(effect_sizes)),
                "min_effect_size": float(np.min(effect_sizes)),
                "max_effect_size": float(np.max(effect_sizes)),
                "median_sample_size": float(np.median(sample_sizes)),
                "total_sample_size": int(np.sum(sample_sizes))
            }
        
        return report
    
    def upload_file(self, file_content: Union[bytes, str], filename: Optional[str] = None) -> str:
        """
        Save an uploaded file to the data directory.
        
        Args:
            file_content: Raw file bytes OR file path string (for legacy support)
            filename: Original filename (optional if file_content is path)
            
        Returns:
            Path to saved file
        """
        if filename is None and isinstance(file_content, str):
            # Handle legacy/buggy calls where only path is passed
            source_path = Path(file_content)
            filename = source_path.name
            with open(source_path, 'rb') as f:
                content = f.read()
        else:
            if filename is None:
                raise ValueError("Filename required when passing bytes")
            content = file_content

        file_path = self.data_dir / filename
        with open(file_path, 'wb') as f:
            f.write(content)
        
        logger.info(f"Saved uploaded file: {file_path}")
        return str(file_path)

    def clear_all_files(self) -> int:
        """
        Delete all experiment files in the data directory.
        
        Returns:
            Number of files deleted
        """
        count = 0
        for ext in ["*.csv", "*.xlsx", "*.xls"]:
            for file_path in self.data_dir.glob(ext):
                try:
                    file_path.unlink()
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
        
        # Clear cache
        self.experiments_cache.clear()
        self._file_hashes.clear()
        
        return count
