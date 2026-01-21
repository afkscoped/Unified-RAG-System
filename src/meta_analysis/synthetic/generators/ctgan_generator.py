"""
CTGAN Generator

Synthetic A/B test generator using Conditional Tabular GAN (CTGAN).
Handles mixed data types, imbalanced data, and multimodal distributions.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import pickle
import pandas as pd
import numpy as np
from loguru import logger

from src.meta_analysis.synthetic.generators.base_generator import (
    BaseGenerator, SyntheticConfig, ValidationReport, DomainType
)
from src.meta_analysis.synthetic.domains.domain_constraints import DomainConstraints, DOMAIN_CONFIGS
from src.meta_analysis.synthetic.validation.quality_metrics import QualityValidator


class CTGANGenerator(BaseGenerator):
    """
    CTGAN-based synthetic A/B test generator.
    
    CTGAN (Conditional Tabular GAN) handles:
    - Mixed data types (categorical + continuous)
    - Imbalanced data
    - Multimodal distributions
    
    Key hyperparameters:
    - epochs: 300-500 for small datasets, 100-200 for large
    - batch_size: 500 (default)
    - generator_dim: (256, 256) for complex data
    - discriminator_dim: (256, 256)
    """
    
    def __init__(self, domain: DomainType, model_dir: Optional[Path] = None):
        super().__init__(domain, model_dir)
        self._synthesizer = None
        self._sdv_available = None  # Lazy check
    
    def _check_sdv(self) -> bool:
        """Check if SDV is available (lazy evaluation)."""
        if self._sdv_available is not None:
            return self._sdv_available
        
        try:
            # Import in a controlled way to catch all errors
            import importlib
            sdv = importlib.import_module("sdv.single_table")
            self._sdv_available = hasattr(sdv, "CTGANSynthesizer")
            return self._sdv_available
        except Exception as e:
            # Catch ANY exception including OSError, DLL errors, etc.
            logger.warning(f"SDV not available ({type(e).__name__}: {str(e)[:50]}). Using fallback.")
            self._sdv_available = False
            return False
    
    def fit(self, real_data: pd.DataFrame, epochs: int = 300, verbose: bool = False) -> None:
        """
        Train CTGAN on real A/B test data.
        
        Args:
            real_data: DataFrame with A/B test data
            epochs: Number of training epochs (100-500)
            verbose: Show training progress
        """
        self._seed_data = real_data.copy()
        
        # Lazy check for SDV availability
        if not self._check_sdv():
            logger.info("SDV not available, using statistical sampling fallback")
            self._is_trained = True
            return
        
        try:
            from sdv.single_table import CTGANSynthesizer
            from sdv.metadata import SingleTableMetadata
            
            logger.info(f"Training CTGAN on {len(real_data)} rows for {epochs} epochs...")
            
            # Create metadata
            self.metadata = SingleTableMetadata()
            self.metadata.detect_from_dataframe(real_data)
            
            # Initialize synthesizer with tuned parameters
            self._synthesizer = CTGANSynthesizer(
                metadata=self.metadata,
                epochs=epochs,
                batch_size=min(500, len(real_data)),
                generator_dim=(256, 256),
                discriminator_dim=(256, 256),
                verbose=verbose
            )
            
            # Train model
            self._synthesizer.fit(real_data)
            self._is_trained = True
            
            logger.info(f"CTGAN training complete for domain: {self.domain}")
            
        except Exception as e:
            logger.error(f"CTGAN training failed: {e}")
            # Fall back to statistical sampling
            self._is_trained = True
    
    def generate(self, config: SyntheticConfig) -> pd.DataFrame:
        """
        Generate synthetic A/B tests.
        
        Args:
            config: SyntheticConfig with generation parameters
            
        Returns:
            DataFrame with synthetic experiments
        """
        np.random.seed(config.seed)
        
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        n = config.num_experiments
        
        # Use SDV synthesizer if available
        if self._sdv_available and self._synthesizer is not None:
            try:
                logger.info(f"Generating {n} synthetic experiments with CTGAN...")
                synthetic_df = self._synthesizer.sample(num_rows=n)
                synthetic_df = self._apply_constraints(synthetic_df)
                return self._post_process(synthetic_df, config)
            except Exception as e:
                logger.warning(f"CTGAN generation failed, using fallback: {e}")
        
        # Fallback: statistical sampling from seed data
        return self._generate_fallback(config)
    
    def _generate_fallback(self, config: SyntheticConfig) -> pd.DataFrame:
        """Generate synthetic data using statistical sampling."""
        logger.info(f"Using statistical fallback for {config.num_experiments} experiments")
        
        domain_config = DOMAIN_CONFIGS[self.domain]
        n = config.num_experiments
        
        # Generate base data
        data = []
        for i in range(n):
            exp_prefix = np.random.choice(domain_config.experiment_prefixes)
            
            # Sample size based on config and domain
            min_size = max(config.sample_size_range[0], domain_config.sample_size_range[0])
            max_size = min(config.sample_size_range[1], domain_config.sample_size_range[1])
            control_visitors = np.random.randint(min_size // 2, max_size // 2)
            variant_visitors = int(control_visitors * np.random.uniform(0.9, 1.1))
            
            # Base conversion rate from domain config
            primary_metric = domain_config.primary_metric
            metric_range = domain_config.metric_ranges.get(primary_metric, (0.05, 0.20))
            base_rate = np.random.uniform(metric_range[0], metric_range[1])
            
            # Apply effect size with heterogeneity
            effect_min, effect_max = config.effect_size_range
            heterogeneity_factors = {"low": 0.5, "medium": 1.0, "high": 1.5}
            het_factor = heterogeneity_factors.get(config.heterogeneity_level, 1.0)
            
            effect = np.random.uniform(effect_min, effect_max) * het_factor
            if np.random.random() < 0.3:  # 30% chance of negative effect
                effect = -effect * 0.5
            
            treatment_rate = base_rate * (1 + effect)
            treatment_rate = np.clip(treatment_rate, 0.001, 0.999)
            
            # Generate conversions
            control_conversions = np.random.binomial(control_visitors, base_rate)
            variant_conversions = np.random.binomial(variant_visitors, treatment_rate)
            
            # Test duration
            dur_min, dur_max = domain_config.test_duration_range
            test_duration = np.random.randint(dur_min, dur_max + 1)
            
            row = {
                "experiment_id": f"{exp_prefix}_{self.domain}_{i+1:04d}",
                "experiment_name": f"{exp_prefix.title()} Test {i+1}",
                "variant": "A/B",
                "control_visitors": control_visitors,
                "control_conversions": control_conversions,
                "variant_visitors": variant_visitors,
                "variant_conversions": variant_conversions,
                "test_duration_days": test_duration,
                "platform": self.domain,
                "domain": self.domain
            }
            
            # Add segments if requested
            if config.include_segments:
                segment_values = DomainConstraints.get_segment_values(self.domain)
                for seg_name, seg_options in segment_values.items():
                    row[seg_name] = np.random.choice(seg_options)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        return self._apply_constraints(df)
    
    def _post_process(self, df: pd.DataFrame, config: SyntheticConfig) -> pd.DataFrame:
        """Post-process generated data."""
        df = df.copy()
        
        # Regenerate experiment IDs
        exp_ids = self._generate_experiment_ids(len(df), 
            prefix=np.random.choice(DOMAIN_CONFIGS[self.domain].experiment_prefixes))
        df["experiment_id"] = exp_ids
        
        # Add segments if missing and requested
        if config.include_segments:
            segment_values = DomainConstraints.get_segment_values(self.domain)
            for seg_name, seg_options in segment_values.items():
                if seg_name not in df.columns:
                    df[seg_name] = [np.random.choice(seg_options) for _ in range(len(df))]
        
        return df
    
    def validate(self, synthetic: pd.DataFrame, real: pd.DataFrame) -> ValidationReport:
        """
        Validate quality of synthetic data.
        
        Args:
            synthetic: Generated synthetic data
            real: Original real data
            
        Returns:
            ValidationReport with quality metrics
        """
        return QualityValidator.full_validation(synthetic, real, self.domain)
    
    def save_model(self, path: Optional[Path] = None) -> Path:
        """Save trained model to disk."""
        save_path = path or self.get_model_path()
        
        model_data = {
            "domain": self.domain,
            "is_trained": self._is_trained,
            "seed_data": self._seed_data,
            "synthesizer": self._synthesizer,
            "metadata": self.metadata
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved CTGAN model to {save_path}")
        return save_path
    
    def load_model(self, path: Path) -> bool:
        """Load trained model from disk."""
        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)
            
            self.domain = model_data["domain"]
            self._is_trained = model_data["is_trained"]
            self._seed_data = model_data["seed_data"]
            self._synthesizer = model_data["synthesizer"]
            self.metadata = model_data["metadata"]
            
            logger.info(f"Loaded CTGAN model from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load CTGAN model: {e}")
            return False
