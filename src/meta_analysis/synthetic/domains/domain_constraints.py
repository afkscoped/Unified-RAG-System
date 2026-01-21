"""
Domain Constraints Module

Defines constraints and parameter ranges for 5 A/B testing domains:
- E-Commerce / Retail
- SaaS / Product
- Healthcare / Clinical Trials
- Digital Marketing / Ad Tech
- Education / EdTech
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class DomainConfig:
    """Configuration for a specific domain."""
    name: str
    display_name: str
    primary_metric: str
    secondary_metrics: List[str]
    sample_size_range: Tuple[int, int]
    test_duration_range: Tuple[int, int]  # days
    effect_size_range: Tuple[float, float]  # relative lift
    metric_ranges: Dict[str, Tuple[float, float]]
    required_segments: List[str]
    experiment_prefixes: List[str]
    correlations: Dict[Tuple[str, str], float]  # Expected correlations between metrics


# Domain configurations for 5 domains
DOMAIN_CONFIGS: Dict[str, DomainConfig] = {
    "ecommerce": DomainConfig(
        name="ecommerce",
        display_name="E-Commerce / Retail",
        primary_metric="conversion_rate",
        secondary_metrics=["avg_order_value", "cart_abandonment_rate", "ctr"],
        sample_size_range=(1000, 500000),
        test_duration_range=(7, 30),
        effect_size_range=(0.02, 0.15),
        metric_ranges={
            "conversion_rate": (0.01, 0.30),
            "avg_order_value": (10.0, 500.0),
            "cart_abandonment_rate": (0.40, 0.80),
            "ctr": (0.01, 0.15),
            "revenue_per_visitor": (0.50, 50.0)
        },
        required_segments=["device_type", "user_type", "region"],
        experiment_prefixes=["checkout", "product", "cart", "homepage", "search"],
        correlations={
            ("avg_order_value", "conversion_rate"): -0.3,
            ("ctr", "conversion_rate"): 0.5
        }
    ),
    
    "saas": DomainConfig(
        name="saas",
        display_name="SaaS / Product",
        primary_metric="activation_rate",
        secondary_metrics=["engagement_minutes", "retention_rate", "time_to_value"],
        sample_size_range=(500, 50000),
        test_duration_range=(14, 60),
        effect_size_range=(0.05, 0.25),
        metric_ranges={
            "activation_rate": (0.05, 0.50),
            "engagement_minutes": (5.0, 240.0),
            "retention_rate": (0.20, 0.80),
            "time_to_value": (1.0, 30.0),  # days
            "feature_adoption_rate": (0.10, 0.60)
        },
        required_segments=["plan_type", "user_cohort", "feature_tier"],
        experiment_prefixes=["onboarding", "feature", "pricing", "dashboard", "notification"],
        correlations={
            ("engagement_minutes", "retention_rate"): 0.6,
            ("activation_rate", "retention_rate"): 0.4
        }
    ),
    
    "healthcare": DomainConfig(
        name="healthcare",
        display_name="Healthcare / Clinical Trials",
        primary_metric="success_rate",
        secondary_metrics=["adverse_event_rate", "symptom_improvement", "compliance_rate"],
        sample_size_range=(50, 5000),
        test_duration_range=(30, 365),
        effect_size_range=(0.10, 0.40),
        metric_ranges={
            "success_rate": (0.30, 0.90),
            "adverse_event_rate": (0.01, 0.20),
            "symptom_improvement": (1.0, 10.0),  # scale 1-10
            "compliance_rate": (0.50, 0.95),
            "readmission_rate": (0.05, 0.30)
        },
        required_segments=["age_group", "severity_level", "comorbidity_status"],
        experiment_prefixes=["treatment", "dosage", "protocol", "intervention", "therapy"],
        correlations={
            ("compliance_rate", "success_rate"): 0.5,
            ("adverse_event_rate", "success_rate"): -0.2
        }
    ),
    
    "marketing": DomainConfig(
        name="marketing",
        display_name="Digital Marketing / Ad Tech",
        primary_metric="ctr",
        secondary_metrics=["cpa", "viewability_rate", "engagement_rate"],
        sample_size_range=(10000, 1000000),
        test_duration_range=(7, 21),
        effect_size_range=(0.03, 0.20),
        metric_ranges={
            "ctr": (0.005, 0.10),
            "cpa": (5.0, 200.0),
            "viewability_rate": (0.30, 0.90),
            "engagement_rate": (0.01, 0.15),
            "conversion_rate": (0.01, 0.10)
        },
        required_segments=["ad_format", "platform", "audience_segment", "time_of_day"],
        experiment_prefixes=["campaign", "creative", "targeting", "bidding", "placement"],
        correlations={
            ("ctr", "engagement_rate"): 0.7,
            ("cpa", "conversion_rate"): -0.4
        }
    ),
    
    "edtech": DomainConfig(
        name="edtech",
        display_name="Education / EdTech",
        primary_metric="completion_rate",
        secondary_metrics=["assessment_score", "time_on_platform", "skill_mastery"],
        sample_size_range=(100, 10000),
        test_duration_range=(30, 180),
        effect_size_range=(0.08, 0.30),
        metric_ranges={
            "completion_rate": (0.20, 0.85),
            "assessment_score": (40.0, 95.0),
            "time_on_platform": (10.0, 480.0),  # minutes per session
            "skill_mastery": (0.30, 0.90),
            "engagement_score": (1.0, 10.0)
        },
        required_segments=["learning_style", "subject_area", "difficulty_level", "age_group"],
        experiment_prefixes=["lesson", "quiz", "content", "gamification", "feedback"],
        correlations={
            ("time_on_platform", "completion_rate"): 0.5,
            ("assessment_score", "skill_mastery"): 0.8
        }
    )
}


class DomainConstraints:
    """
    Domain-specific rules and constraints for synthetic data generation.
    
    Provides methods to:
    - Get domain configuration
    - Validate generated data
    - Enforce constraints
    - Check for violations
    """
    
    @classmethod
    def get_config(cls, domain: str) -> DomainConfig:
        """
        Get configuration for a specific domain.
        
        Args:
            domain: Domain name (ecommerce, saas, healthcare, marketing, edtech)
            
        Returns:
            DomainConfig for the domain
            
        Raises:
            ValueError: If domain is unknown
        """
        if domain not in DOMAIN_CONFIGS:
            raise ValueError(f"Unknown domain: {domain}. Valid domains: {list(DOMAIN_CONFIGS.keys())}")
        return DOMAIN_CONFIGS[domain]
    
    @classmethod
    def get_all_domains(cls) -> List[str]:
        """Get list of all available domains."""
        return list(DOMAIN_CONFIGS.keys())
    
    @classmethod
    def enforce_constraints(cls, df: pd.DataFrame, domain: str) -> pd.DataFrame:
        """
        Enforce domain-specific constraints on generated data.
        
        Args:
            df: Generated DataFrame
            domain: Domain name
            
        Returns:
            DataFrame with enforced constraints
        """
        df = df.copy()
        config = cls.get_config(domain)
        
        # Enforce metric ranges
        for col, (min_val, max_val) in config.metric_ranges.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=min_val, upper=max_val)
        
        # Ensure integer columns for sample sizes
        sample_cols = ["sample_size", "control_sample_size", "treatment_sample_size", 
                       "control_visitors", "variant_visitors", "control_conversions", 
                       "variant_conversions", "successes", "events"]
        for col in sample_cols:
            if col in df.columns:
                df[col] = df[col].abs().astype(int)
        
        # Ensure conversions <= sample size
        if "control_conversions" in df.columns and "control_visitors" in df.columns:
            df["control_conversions"] = df.apply(
                lambda r: min(int(r["control_conversions"]), int(r["control_visitors"])), axis=1
            )
        if "variant_conversions" in df.columns and "variant_visitors" in df.columns:
            df["variant_conversions"] = df.apply(
                lambda r: min(int(r["variant_conversions"]), int(r["variant_visitors"])), axis=1
            )
        
        # Ensure sample size within domain range
        min_size, max_size = config.sample_size_range
        if "sample_size" in df.columns:
            df["sample_size"] = df["sample_size"].clip(lower=min_size, upper=max_size)
        if "control_visitors" in df.columns:
            df["control_visitors"] = df["control_visitors"].clip(lower=min_size // 2, upper=max_size)
        if "variant_visitors" in df.columns:
            df["variant_visitors"] = df["variant_visitors"].clip(lower=min_size // 2, upper=max_size)
        
        # Ensure test duration within range
        min_dur, max_dur = config.test_duration_range
        if "test_duration_days" in df.columns:
            df["test_duration_days"] = df["test_duration_days"].clip(lower=min_dur, upper=max_dur)
        
        # Ensure rates are between 0 and 1
        rate_cols = [c for c in df.columns if "rate" in c.lower()]
        for col in rate_cols:
            df[col] = df[col].clip(lower=0.001, upper=0.999)
        
        logger.debug(f"Applied constraints for domain: {domain}")
        return df
    
    @classmethod
    def check_violations(cls, df: pd.DataFrame, domain: str) -> List[str]:
        """
        Check for constraint violations in data.
        
        Args:
            df: DataFrame to check
            domain: Domain name
            
        Returns:
            List of violation descriptions
        """
        violations = []
        config = cls.get_config(domain)
        
        # Check metric ranges
        for col, (min_val, max_val) in config.metric_ranges.items():
            if col in df.columns:
                if df[col].min() < min_val:
                    violations.append(f"{col} below minimum: {df[col].min():.4f} < {min_val}")
                if df[col].max() > max_val:
                    violations.append(f"{col} above maximum: {df[col].max():.4f} > {max_val}")
        
        # Check sample size range
        min_size, max_size = config.sample_size_range
        for col in ["sample_size", "control_visitors", "variant_visitors"]:
            if col in df.columns:
                if df[col].min() < min_size // 2:
                    violations.append(f"{col} too small: {df[col].min()} < {min_size // 2}")
        
        # Check for negative values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (df[col] < 0).any():
                violations.append(f"{col} contains negative values")
        
        # Check conversions <= visitors
        if "control_conversions" in df.columns and "control_visitors" in df.columns:
            invalid = (df["control_conversions"] > df["control_visitors"]).sum()
            if invalid > 0:
                violations.append(f"{invalid} rows have control_conversions > control_visitors")
        
        if "variant_conversions" in df.columns and "variant_visitors" in df.columns:
            invalid = (df["variant_conversions"] > df["variant_visitors"]).sum()
            if invalid > 0:
                violations.append(f"{invalid} rows have variant_conversions > variant_visitors")
        
        return violations
    
    @classmethod
    def get_segment_values(cls, domain: str) -> Dict[str, List[str]]:
        """
        Get valid segment values for a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Dict mapping segment name to list of valid values
        """
        segment_values = {
            "ecommerce": {
                "device_type": ["mobile", "desktop", "tablet"],
                "user_type": ["new", "returning"],
                "region": ["US", "EU", "APAC", "LATAM"]
            },
            "saas": {
                "plan_type": ["free", "pro", "enterprise"],
                "user_cohort": ["week1", "week2", "month1", "month3"],
                "feature_tier": ["basic", "advanced", "premium"]
            },
            "healthcare": {
                "age_group": ["18-34", "35-49", "50-64", "65+"],
                "severity_level": ["mild", "moderate", "severe"],
                "comorbidity_status": ["none", "one", "multiple"]
            },
            "marketing": {
                "ad_format": ["display", "video", "native", "search"],
                "platform": ["facebook", "google", "instagram", "tiktok"],
                "audience_segment": ["broad", "interest", "lookalike", "retargeting"],
                "time_of_day": ["morning", "afternoon", "evening", "night"]
            },
            "edtech": {
                "learning_style": ["visual", "auditory", "kinesthetic"],
                "subject_area": ["math", "science", "language", "arts"],
                "difficulty_level": ["beginner", "intermediate", "advanced"],
                "age_group": ["K-5", "6-8", "9-12", "adult"]
            }
        }
        return segment_values.get(domain, {})
