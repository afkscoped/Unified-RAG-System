"""
Schema Mapper

Maps domain-specific schemas to StandardizedStudy format for meta-analysis.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from loguru import logger


class SchemaMapper:
    """
    Map domain-specific A/B test data to StandardizedStudy format.
    
    StandardizedStudy format required by meta-analysis pipeline:
    - study_id: Unique identifier
    - study_name: Display name
    - effect_size: Computed effect size (log odds ratio, etc.)
    - standard_error: Standard error of effect size
    - control_n: Control sample size
    - treatment_n: Treatment sample size
    - control_events: Control conversions/successes
    - treatment_events: Treatment conversions/successes
    """
    
    # Column mapping for different schemas
    COLUMN_MAPPINGS = {
        "experiment_id": ["experiment_id", "exp_id", "id", "study_id"],
        "experiment_name": ["experiment_name", "exp_name", "name", "study_name", "test_name"],
        "control_visitors": ["control_visitors", "control_n", "control_sample_size", "control_size"],
        "control_conversions": ["control_conversions", "control_events", "control_successes"],
        "variant_visitors": ["variant_visitors", "treatment_n", "treatment_sample_size", "variant_size"],
        "variant_conversions": ["variant_conversions", "treatment_events", "treatment_successes"],
    }
    
    @classmethod
    def to_standardized_study(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DataFrame to StandardizedStudy format.
        
        Args:
            df: Input DataFrame with A/B test data
            
        Returns:
            DataFrame in StandardizedStudy format
        """
        result = pd.DataFrame()
        
        # Map columns
        for target_col, source_options in cls.COLUMN_MAPPINGS.items():
            for source_col in source_options:
                if source_col in df.columns:
                    result[target_col] = df[source_col]
                    break
        
        # Compute effect sizes if we have the required columns
        if cls._has_required_columns(result):
            result = cls._compute_effect_sizes(result)
        
        # Add MCP-compatible column aliases (MCP expects these names)
        if "control_visitors" in result.columns:
            result["control_total"] = result["control_visitors"]
        if "variant_visitors" in result.columns:
            result["treatment_total"] = result["variant_visitors"]
        if "variant_conversions" in result.columns:
            result["treatment_conversions"] = result["variant_conversions"]
        
        # Add metadata columns
        if "platform" in df.columns:
            result["platform"] = df["platform"]
        if "domain" in df.columns:
            result["domain"] = df["domain"]
        if "test_duration_days" in df.columns:
            result["test_duration_days"] = df["test_duration_days"]
        
        # Copy segment columns
        segment_cols = ["device_type", "user_type", "region", "plan_type", 
                        "user_cohort", "age_group", "severity_level", 
                        "ad_format", "platform", "learning_style", "subject_area"]
        for col in segment_cols:
            if col in df.columns and col not in result.columns:
                result[col] = df[col]
        
        logger.info(f"Mapped {len(df)} rows to StandardizedStudy format")
        return result
    
    @classmethod
    def _has_required_columns(cls, df: pd.DataFrame) -> bool:
        """Check if DataFrame has required columns for effect size computation."""
        required = ["control_visitors", "control_conversions", 
                    "variant_visitors", "variant_conversions"]
        return all(col in df.columns for col in required)
    
    @classmethod
    def _compute_effect_sizes(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Compute effect sizes (log odds ratio) and standard errors."""
        df = df.copy()
        
        # Extract values
        c_n = df["control_visitors"].values.astype(float)
        c_e = df["control_conversions"].values.astype(float)
        t_n = df["variant_visitors"].values.astype(float)
        t_e = df["variant_conversions"].values.astype(float)
        
        # Add small correction to avoid division by zero
        correction = 0.5
        
        c_e_adj = c_e + correction
        c_non_e = c_n - c_e + correction
        t_e_adj = t_e + correction
        t_non_e = t_n - t_e + correction
        
        # Log odds ratio
        log_or = np.log((t_e_adj * c_non_e) / (c_e_adj * t_non_e))
        
        # Standard error of log odds ratio
        se = np.sqrt(1/c_e_adj + 1/c_non_e + 1/t_e_adj + 1/t_non_e)
        
        df["effect_size"] = log_or
        df["standard_error"] = se
        
        # Also compute simple relative lift
        c_rate = c_e / c_n
        t_rate = t_e / t_n
        df["control_rate"] = c_rate
        df["treatment_rate"] = t_rate
        df["relative_lift"] = (t_rate - c_rate) / np.maximum(c_rate, 1e-6)
        
        # Total sample size
        df["total_sample_size"] = c_n + t_n
        
        return df
    
    @classmethod
    def from_domain_format(cls, df: pd.DataFrame, domain: str) -> pd.DataFrame:
        """
        Convert domain-specific format to standard format.
        
        Args:
            df: Domain-specific DataFrame
            domain: Domain name
            
        Returns:
            Standardized DataFrame
        """
        # Domain-specific column mappings
        domain_mappings = {
            "ecommerce": {
                "conversions": "variant_conversions",
                "visitors": "variant_visitors",
                "success_rate": None,  # Will compute
            },
            "healthcare": {
                "successes": "variant_conversions",
                "participants": "variant_visitors",
                "success_rate": "treatment_rate"
            },
            "saas": {
                "activations": "variant_conversions", 
                "users": "variant_visitors"
            },
            "marketing": {
                "clicks": "variant_conversions",
                "impressions": "variant_visitors"
            },
            "edtech": {
                "completions": "variant_conversions",
                "enrollments": "variant_visitors"
            }
        }
        
        mapping = domain_mappings.get(domain, {})
        df_copy = df.copy()
        
        for source, target in mapping.items():
            if source in df_copy.columns and target is not None:
                df_copy[target] = df_copy[source]
        
        return cls.to_standardized_study(df_copy)
    
    @classmethod
    def validate_schema(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that DataFrame has required schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict with validation results
        """
        required = ["experiment_id", "control_visitors", "control_conversions",
                    "variant_visitors", "variant_conversions"]
        
        missing = [col for col in required if col not in df.columns]
        
        # Check for alternative column names
        found_alternatives = {}
        for req_col in missing:
            if req_col in cls.COLUMN_MAPPINGS:
                for alt in cls.COLUMN_MAPPINGS[req_col]:
                    if alt in df.columns:
                        found_alternatives[req_col] = alt
                        break
        
        return {
            "valid": len(missing) == 0 or len(found_alternatives) == len(missing),
            "missing_columns": missing,
            "found_alternatives": found_alternatives,
            "available_columns": list(df.columns)
        }
