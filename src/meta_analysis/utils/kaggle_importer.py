"""
Kaggle A/B Test Dataset Importer

Downloads and processes A/B test datasets from Kaggle.
Handles column detection, cleaning, and validation.
"""

import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from loguru import logger


# Common A/B test column name patterns
AB_COLUMN_PATTERNS = {
    "experiment_name": [
        "experiment_name", "test_name", "campaign", "experiment", "test", 
        "name", "exp_name", "campaign_name", "variant_name"
    ],
    "control_visitors": [
        "control_visitors", "control_users", "control_size", "control_n",
        "control_impressions", "baseline_visitors", "control_total"
    ],
    "control_conversions": [
        "control_conversions", "control_success", "control_converted",
        "baseline_conversions", "control_clicks", "control_purchases"
    ],
    "variant_visitors": [
        "variant_visitors", "treatment_visitors", "treatment_users", 
        "treatment_size", "treatment_n", "experiment_visitors", "variant_total"
    ],
    "variant_conversions": [
        "variant_conversions", "treatment_conversions", "treatment_success",
        "treatment_converted", "experiment_conversions", "variant_clicks"
    ]
}


class KaggleABTestImporter:
    """
    Imports A/B test datasets from Kaggle.
    
    Handles:
    - Dataset download via Kaggle API
    - Column detection for A/B test data
    - Data cleaning and validation
    """
    
    def __init__(self, download_dir: Optional[str] = None):
        """
        Initialize the importer.
        
        Args:
            download_dir: Directory for downloaded files (default: temp dir or cache)
        """
        # --- CACHING LOGIC ---
        self.cache_dir = os.path.expanduser("~/.unified_rag/kaggle_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.download_dir = download_dir or self.cache_dir
        self._kaggle_available = self._check_kaggle()

    def fetch_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Fetch dataset metadata efficiently."""
        if not self._kaggle_available:
            return {"error": "Kaggle API unavailable"}
            
        try:
            import kaggle
            dataset_id = self._clean_id(dataset_id)
            # Use lower-level API to list files without downloading
            # This is a bit of a hack as Kaggle API doesn't have a clean 'get_metadata' 
            # for unauth users or easy file listing, but list_files helps.
            files = kaggle.api.dataset_list_files(dataset_id).files
            
            total_size = sum(f.size for f in files) if files else 0
            file_count = len(files) if files else 0
            
            # Find largest/smallest CSVs
            csv_files = [f for f in files if f.name.endswith('.csv')]
            
            metadata = {
                "total_size_mb": round(total_size / (1024*1024), 2),
                "file_count": file_count,
                "csv_count": len(csv_files),
                "files": [{"name": f.name, "size": f.size} for f in files]
            }
            return metadata
        except Exception as e:
            logger.error(f"Failed to fetch metadata: {e}")
            return {"error": str(e)}

    def _clean_id(self, dataset_id: str) -> str:
        dataset_id = dataset_id.strip()
        if dataset_id.startswith("https://"):
            parts = dataset_id.split("/")
            if len(parts) >= 2:
                dataset_id = f"{parts[-2]}/{parts[-1]}"
        return dataset_id
        
    def _check_kaggle(self) -> bool:
        """Check if Kaggle API is configured."""
        try:
            import kaggle
            kaggle.api.authenticate()
            return True
        except Exception as e:
            logger.warning(f"Kaggle API not configured: {e}")
            return False
    
    def download_dataset(
        self, 
        dataset_id: str,
        force: bool = False,
        fast_mode: bool = True
    ) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """
        Download a dataset from Kaggle.
        
        Args:
            dataset_id: Kaggle dataset ID
            force: Force re-download even if exists
            fast_mode: If True, try to download only the relevant CSV file (default: True)
            
        Returns:
            Tuple of (success, message, DataFrame or None)
        """
        if not self._kaggle_available:
            return False, "Kaggle API not configured. Please set up ~/.kaggle/kaggle.json", None
        
        try:
            import kaggle
            dataset_id = self._clean_id(dataset_id)
            safe_id = dataset_id.replace("/", "_")
            
            logger.info(f"Downloading Kaggle dataset: {dataset_id} (Fast Mode: {fast_mode})")
            
            download_path = Path(self.download_dir) / safe_id
            download_path.mkdir(parents=True, exist_ok=True)
            
            main_csv = None
            
            # --- CACHE CHECK ---
            # Check if we already have CSVs in this folder from previous run
            existing_csvs = list(download_path.glob("**/*.csv"))
            if not force and existing_csvs:
                logger.info(f"Using cached files in {download_path}")
                main_csv = existing_csvs[0]
                if len(existing_csvs) > 1:
                     main_csv = self._pick_best_csv(existing_csvs)
                
                # Verify validity (basic check)
                if main_csv.stat().st_size > 0:
                     return self._load_csv(main_csv)
            
            # --- FAST MODE DOWNLOAD ---
            if fast_mode:
                try:
                    import time
                    t0 = time.time()
                    logger.info("Fast Mode: Listing files via API...")
                    files = kaggle.api.dataset_list_files(dataset_id).files
                    logger.info(f"Fast Mode: File list took {time.time()-t0:.2f}s")
                    
                    csv_files = [f for f in files if f.name.endswith('.csv')]
                    
                    if not csv_files:
                        logger.warning("No CSVs found in file list, falling back to full download.")
                    else:
                        # Pick smallest valid CSV or one with 'ab' in name
                        target_file = None
                        
                        # Preference 1: Contains 'ab', 'test' etc AND is < 50MB
                        candidates = [f for f in csv_files if any(kw in f.name.lower() for kw in ['ab', 'test', 'exp', 'result'])]
                        candidates = [f for f in candidates if f.size < 50 * 1024 * 1024]
                        
                        if candidates:
                            target_file = sorted(candidates, key=lambda x: x.size)[0] # Smallest relevant
                        else:
                            # Preference 2: Any small CSV < 10MB
                            small_candidates = [f for f in csv_files if f.size < 10 * 1024 * 1024]
                            if small_candidates:
                                target_file = small_candidates[0]
                        
                        if target_file:
                            logger.info(f"Fast Mode: Downloading single file {target_file.name} ({target_file.size} bytes)")
                            t1 = time.time()
                            kaggle.api.dataset_download_file(
                                dataset_id,
                                target_file.name,
                                path=str(download_path),
                                force=force
                            )
                            logger.info(f"Fast Mode: Download took {time.time()-t1:.2f}s")
                            
                            # Handle if it comes as zip (Kaggle sometimes zips single files)
                            downloaded_file = download_path / target_file.name
                            if not downloaded_file.exists():
                                # Check if it was zipped
                                zipped = downloaded_file.with_suffix(downloaded_file.suffix + ".zip")
                                if zipped.exists():
                                    with zipfile.ZipFile(zipped, 'r') as zip_ref:
                                        zip_ref.extractall(download_path)
                            
                            main_csv = download_path / target_file.name
                            if main_csv.exists():
                                return self._load_csv(main_csv, mode_info="Fast Mode")
                        else:
                             logger.warning("No suitable small CSV found for Fast Mode.")

                except Exception as e:
                    logger.warning(f"Fast Mode failed ({e}), falling back to full download.")

            # --- FULL DOWNLOAD FALLBACK ---
            logger.info("Starting Full Download (Fast Mode skipped or failed)")
            t2 = time.time()
            kaggle.api.dataset_download_files(
                dataset_id, 
                path=str(download_path),
                unzip=False, # We will unzip selectively
                force=force
            )
            logger.info(f"Full Download took {time.time()-t2:.2f}s")
            
            # Find zip files and extract ONLY CSVs
            zip_files = list(download_path.glob("**/*.zip"))
            for zf in zip_files:
                with zipfile.ZipFile(zf, 'r') as zip_ref:
                    # Only extract CSVs, ignore images/other junk
                    csv_members = [m for m in zip_ref.namelist() if m.lower().endswith('.csv')]
                    if csv_members:
                         logger.info(f"Extracting {len(csv_members)} CSVs from zip...")
                         zip_ref.extractall(download_path, members=csv_members)
            
            # Find CSV files
            csv_files = list(download_path.glob("**/*.csv"))
            
            if not csv_files:
                return False, "No CSV files found in dataset", None
            
            main_csv = csv_files[0]
            if len(csv_files) > 1:
                main_csv = self._pick_best_csv(csv_files)
            
            return self._load_csv(main_csv, mode_info="Full Download")
            
        except Exception as e:
            logger.error(f"Kaggle download failed: {e}")
            return False, f"Download failed: {str(e)}", None

    def _pick_best_csv(self, csv_files: List[Path]) -> Path:
        """Helper to pick best CSV from a list."""
        for cf in csv_files:
             if any(kw in cf.name.lower() for kw in ['ab', 'test', 'experiment']):
                 return cf
        return csv_files[0]

    def _load_csv(self, path: Path, mode_info: str = "") -> Tuple[bool, str, Optional[pd.DataFrame]]:
        try:
            df = pd.read_csv(path)
            logger.info(f"Loaded {len(df)} rows from {path.name}")
            return True, f"Successfully imported {path.name} ({mode_info})", df
        except Exception as e:
             return False, f"Failed to load CSV: {e}", None
    
    def detect_ab_columns(
        self, 
        df: pd.DataFrame
    ) -> Dict[str, Optional[str]]:
        """
        Auto-detect A/B test columns in a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping standard column names to detected columns
        """
        detected = {}
        df_columns_lower = {col.lower().strip(): col for col in df.columns}
        
        for standard_name, patterns in AB_COLUMN_PATTERNS.items():
            detected[standard_name] = None
            for pattern in patterns:
                if pattern.lower() in df_columns_lower:
                    detected[standard_name] = df_columns_lower[pattern.lower()]
                    break
        
        return detected
    
    def clean_and_validate(
        self, 
        df: pd.DataFrame,
        column_mapping: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """
        Clean and validate A/B test data.
        
        Args:
            df: Input DataFrame
            column_mapping: Optional manual column mapping
            
        Returns:
            Tuple of (success, message, cleaned DataFrame or None)
        """
        if column_mapping is None:
            column_mapping = self.detect_ab_columns(df)
        
        # Check required columns
        required = ["control_visitors", "control_conversions", 
                    "variant_visitors", "variant_conversions"]
        missing = [col for col in required if column_mapping.get(col) is None]
        
        if missing:
            return False, f"Missing required columns: {missing}", None
        
        try:
            # Create standardized DataFrame
            cleaned = pd.DataFrame()
            
            # Map columns
            if column_mapping.get("experiment_name"):
                cleaned["experiment_name"] = df[column_mapping["experiment_name"]].astype(str)
            else:
                cleaned["experiment_name"] = [f"experiment_{i+1}" for i in range(len(df))]
            
            # Numeric columns
            for std_col in required:
                src_col = column_mapping[std_col]
                cleaned[std_col] = pd.to_numeric(df[src_col], errors='coerce')
            
            # Drop rows with NaN
            initial_len = len(cleaned)
            cleaned = cleaned.dropna()
            dropped = initial_len - len(cleaned)
            
            if dropped > 0:
                logger.warning(f"Dropped {dropped} rows with invalid data")
            
            # Ensure integers
            for col in required:
                cleaned[col] = cleaned[col].astype(int)
            
            # Validate: conversions <= visitors
            invalid_control = cleaned["control_conversions"] > cleaned["control_visitors"]
            invalid_variant = cleaned["variant_conversions"] > cleaned["variant_visitors"]
            
            if invalid_control.any() or invalid_variant.any():
                n_invalid = invalid_control.sum() + invalid_variant.sum()
                cleaned = cleaned[~(invalid_control | invalid_variant)]
                logger.warning(f"Removed {n_invalid} rows with conversions > visitors")
            
            # Validate: positive values
            cleaned = cleaned[
                (cleaned["control_visitors"] > 0) & 
                (cleaned["variant_visitors"] > 0)
            ]
            
            if len(cleaned) == 0:
                return False, "No valid rows after cleaning", None
            
            # Rename to standard format for MCP
            cleaned = cleaned.rename(columns={
                "control_visitors": "control_total",
                "variant_visitors": "treatment_total",
                "variant_conversions": "treatment_conversions"
            })
            
            return True, f"Cleaned dataset: {len(cleaned)} valid experiments", cleaned
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            return False, f"Cleaning failed: {str(e)}", None
    
    def import_dataset(
        self, 
        dataset_id: str,
        fast_mode: bool = True
    ) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """
        Full import workflow: download, detect, clean.
        
        Args:
            dataset_id: Kaggle dataset ID or URL
            fast_mode: Enable fast single-file download
            
        Returns:
            Tuple of (success, message, cleaned DataFrame or None)
        """
        # Download
        success, msg, df = self.download_dataset(dataset_id, fast_mode=fast_mode)
        if not success:
            return success, msg, None
        
        # Detect columns
        column_mapping = self.detect_ab_columns(df)
        
        # Clean and validate
        return self.clean_and_validate(df, column_mapping)
    
    @staticmethod
    def get_sample_datasets() -> List[Dict[str, str]]:
        """
        Get list of known A/B test datasets on Kaggle.
        
        Returns:
            List of dataset info dictionaries
        """
        return [
            {
                "id": "zhangluyuan/ab-testing",
                "name": "E-commerce A/B Testing",
                "description": "Conversion rates (Simple)"
            },
            {
                "id": "andrewmvd/marketing-ab-testing",
                "name": "Marketing A/B Testing",
                "description": "Ad campaigns (Classic)"
            },
            {
                "id": "yufengsui/mobile-games-ab-testing",
                "name": "Mobile Games A/B",
                "description": "Retention (Cookie Cats)"
            },
            {
                "id": "osuolaleemmanuel/ad-ab-testing",
                "name": "Ad A/B Testing",
                "description": "SmartAd campaign"
            },
            {
                "id": "faviovaz/marketing-ab-testing",
                "name": "Marketing Campaign",
                "description": "Conversion rates"
            },
            {
                "id": "chebotinaa/ab-testing-marketing",
                "name": "A/B Marketing Promos",
                "description": "Promo effectiveness"
            },
            {
                "id": "auton1/ab-testing-results",
                "name": "A/B Test Results",
                "description": "General purpose"
            },
            {
                "id": "dikshabhati2002/ab-testing-dataset",
                "name": "Website A/B Test",
                "description": "Theme change impact"
            },
            {
                "id": "raahimghaffar/ab-testing-dataset",
                "name": "Campaign AB Test",
                "description": "Revenue impact"
            },
            {
                "id": "putdejudomthai/ecommerce-ab-testing-2022-dataset",
                "name": "E-comm 2022 AB",
                "description": "Landing page tests"
            }
        ]
