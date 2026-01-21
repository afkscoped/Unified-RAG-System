"""
Synthetic A/B Test Data Generator

Uses SDV (Synthetic Data Vault) to generate realistic A/B test datasets.
Supports CTGAN and CopulaGAN models with domain-specific parameters.
"""

import pandas as pd
import numpy as np
from typing import Optional, Literal
from loguru import logger

# Domain-specific parameter ranges
DOMAIN_CONFIGS = {
    "marketing": {
        "visitors_range": (1000, 50000),
        "base_conversion_rate": (0.02, 0.08),
        "lift_range": (-0.25, 0.35), # Widened
        "experiment_prefixes": ["campaign", "ad", "promo", "email_blast", "social"]
    },
    "product": {
        "visitors_range": (500, 20000),
        "base_conversion_rate": (0.05, 0.15),
        "lift_range": (-0.20, 0.30), # Widened
        "experiment_prefixes": ["feature", "ui", "checkout", "onboarding", "pricing"]
    },
    "email": {
        "visitors_range": (2000, 100000),
        "base_conversion_rate": (0.01, 0.05),
        "lift_range": (-0.30, 0.40), # Widened
        "experiment_prefixes": ["subject", "cta", "template", "timing", "segment"]
    },
    "ux": {
        "visitors_range": (200, 10000),
        "base_conversion_rate": (0.10, 0.30),
        "lift_range": (-0.15, 0.25), # Widened
        "experiment_prefixes": ["button", "layout", "color", "flow", "copy"]
    }
}

DomainType = Literal["marketing", "product", "email", "ux"]


class SyntheticABTestGenerator:
    """
    Generates synthetic A/B test datasets using SDV models.
    
    Provides two generation methods:
    - CTGAN: Uses Conditional Tabular GAN for realistic distributions
    - CopulaGAN: Uses Gaussian Copula for correlated data
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the generator.
        
        Args:
            random_seed: Seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self._sdv_available = self._check_sdv()
        
    def _check_sdv(self) -> bool:
        """Check if SDV is available."""
        try:
            from sdv.single_table import CTGANSynthesizer, CopulaGANSynthesizer
            return True
        except ImportError:
            logger.warning("SDV not installed. Using fallback random generation.")
            return False
    
    def _generate_base_data(
        self, 
        n_experiments: int, 
        domain: DomainType
    ) -> pd.DataFrame:
        """
        Generate base realistic A/B test data using domain parameters.
        
        Args:
            n_experiments: Number of experiments to generate
            domain: Domain type for parameter ranges
            
        Returns:
            DataFrame with A/B test data
        """
        config = DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS["marketing"])
        
        data = []
        for i in range(n_experiments):
            # Generate experiment name
            prefix = np.random.choice(config["experiment_prefixes"])
            exp_name = f"{prefix}_test_{i+1:03d}"
            
            # Generate visitor counts
            min_v, max_v = config["visitors_range"]
            control_visitors = np.random.randint(min_v, max_v)
            # Treatment usually has similar size
            treatment_visitors = int(control_visitors * np.random.uniform(0.9, 1.1))
            
            # Generate conversion rates
            min_cr, max_cr = config["base_conversion_rate"]
            control_rate = np.random.uniform(min_cr, max_cr)
            
            # Apply lift for treatment
            min_lift, max_lift = config["lift_range"]
            lift = np.random.uniform(min_lift, max_lift)
            treatment_rate = control_rate * (1 + lift)
            treatment_rate = max(0.001, min(0.999, treatment_rate))  # Clamp
            
            # Generate conversions from rates
            control_conversions = np.random.binomial(control_visitors, control_rate)
            treatment_conversions = np.random.binomial(treatment_visitors, treatment_rate)
            
            data.append({
                "experiment_name": exp_name,
                "control_visitors": control_visitors,
                "control_conversions": control_conversions,
                "variant_visitors": treatment_visitors,
                "variant_conversions": treatment_conversions
            })
        
        return pd.DataFrame(data)
    
    def generate_ctgan(
        self, 
        n_experiments: int, 
        domain: DomainType = "marketing"
    ) -> pd.DataFrame:
        """
        Generate synthetic A/B test data using CTGAN.
        
        Args:
            n_experiments: Number of experiments to generate
            domain: Domain type (marketing, product, email, ux)
            
        Returns:
            DataFrame with synthetic A/B test data
        """
        # Generate base data first
        base_data = self._generate_base_data(max(100, n_experiments * 2), domain)
        
        if not self._sdv_available:
            logger.info("Using fallback generation (SDV not available)")
            return base_data.head(n_experiments)
        
        try:
            from sdv.single_table import CTGANSynthesizer
            from sdv.metadata import SingleTableMetadata
            
            # Create metadata
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(base_data)
            
            # Train CTGAN
            synthesizer = CTGANSynthesizer(
                metadata,
                epochs=100,
                verbose=False
            )
            synthesizer.fit(base_data)
            
            # Generate synthetic data
            synthetic_data = synthesizer.sample(n_experiments)
            
            # Post-process: ensure valid values
            synthetic_data = self._post_process(synthetic_data, domain)
            
            logger.info(f"Generated {n_experiments} experiments using CTGAN")
            return synthetic_data
            
        except Exception as e:
            logger.error(f"CTGAN generation failed: {e}. Using fallback.")
            return base_data.head(n_experiments)
    
    def generate_copula(
        self, 
        n_experiments: int, 
        domain: DomainType = "marketing"
    ) -> pd.DataFrame:
        """
        Generate synthetic A/B test data using CopulaGAN.
        
        Args:
            n_experiments: Number of experiments to generate
            domain: Domain type (marketing, product, email, ux)
            
        Returns:
            DataFrame with synthetic A/B test data
        """
        # Generate base data first
        base_data = self._generate_base_data(max(100, n_experiments * 2), domain)
        
        if not self._sdv_available:
            logger.info("Using fallback generation (SDV not available)")
            return base_data.head(n_experiments)
        
        try:
            from sdv.single_table import CopulaGANSynthesizer
            from sdv.metadata import SingleTableMetadata
            
            # Create metadata
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(base_data)
            
            # Train CopulaGAN
            synthesizer = CopulaGANSynthesizer(
                metadata,
                epochs=100,
                verbose=False
            )
            synthesizer.fit(base_data)
            
            # Generate synthetic data
            synthetic_data = synthesizer.sample(n_experiments)
            
            # Post-process
            synthetic_data = self._post_process(synthetic_data, domain)
            
            logger.info(f"Generated {n_experiments} experiments using CopulaGAN")
            return synthetic_data
            
        except Exception as e:
            logger.error(f"CopulaGAN generation failed: {e}. Using fallback.")
            return base_data.head(n_experiments)
    
    def _post_process(self, df: pd.DataFrame, domain: DomainType) -> pd.DataFrame:
        """
        Post-process generated data to ensure validity.
        
        Args:
            df: Generated DataFrame
            domain: Domain type
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        config = DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS["marketing"])
        
        # Ensure integer columns
        int_cols = ["control_visitors", "control_conversions", 
                    "variant_visitors", "variant_conversions"]
        for col in int_cols:
            if col in df.columns:
                df[col] = df[col].astype(int).abs()
        
        # Ensure conversions <= visitors
        df["control_conversions"] = df.apply(
            lambda r: min(r["control_conversions"], r["control_visitors"]), axis=1
        )
        df["variant_conversions"] = df.apply(
            lambda r: min(r["variant_conversions"], r["variant_visitors"]), axis=1
        )
        
        # Ensure minimum visitors
        min_visitors = config["visitors_range"][0]
        df.loc[df["control_visitors"] < min_visitors, "control_visitors"] = min_visitors
        df.loc[df["variant_visitors"] < min_visitors, "variant_visitors"] = min_visitors
        
        # Regenerate experiment names if needed
        prefixes = config["experiment_prefixes"]
        df["experiment_name"] = [
            f"{np.random.choice(prefixes)}_synth_{i+1:03d}" 
            for i in range(len(df))
        ]
        
        return df
    
    def generate(
        self, 
        n_experiments: int, 
        domain: DomainType = "marketing",
        method: Literal["ctgan", "copula", "random"] = "random"
    ) -> pd.DataFrame:
        """
        Generate synthetic data using specified method.
        
        Args:
            n_experiments: Number of experiments
            domain: Domain type
            method: Generation method
            
        Returns:
            DataFrame with synthetic A/B test data
        """
        if method == "ctgan":
            return self.generate_ctgan(n_experiments, domain)
        elif method == "copula":
            return self.generate_copula(n_experiments, domain)
        else:
            return self._generate_base_data(n_experiments, domain)
