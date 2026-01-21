"""
CopulaGAN Generator

Synthetic A/B test generator using CopulaGAN.
Better for preserving complex correlations and faster training on smaller datasets.
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


class CopulaGANGenerator(BaseGenerator):
    """
    CopulaGAN-based synthetic A/B test generator.
    
    CopulaGAN is better for:
    - Preserving complex correlations between variables
    - Smaller datasets (faster training)
    - When interpretability matters
    
    Uses Gaussian Copula to model dependencies separately from marginals.
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
            self._sdv_available = hasattr(sdv, "CopulaGANSynthesizer")
            return self._sdv_available
        except Exception as e:
            # Catch ANY exception including OSError, DLL errors, etc.
            logger.warning(f"SDV not available ({type(e).__name__}: {str(e)[:50]}). Using fallback.")
            self._sdv_available = False
            return False
    
    def fit(self, real_data: pd.DataFrame, epochs: int = 300, verbose: bool = False) -> None:
        """
        Train CopulaGAN on real A/B test data.
        
        Args:
            real_data: DataFrame with A/B test data
            epochs: Number of training epochs
            verbose: Show training progress
        """
        self._seed_data = real_data.copy()
        
        # Lazy check for SDV availability
        if not self._check_sdv():
            logger.info("SDV not available, using correlation-aware fallback")
            self._is_trained = True
            return
        
        try:
            from sdv.single_table import CopulaGANSynthesizer
            from sdv.metadata import SingleTableMetadata
            
            logger.info(f"Training CopulaGAN on {len(real_data)} rows for {epochs} epochs...")
            
            # Create metadata
            self.metadata = SingleTableMetadata()
            self.metadata.detect_from_dataframe(real_data)
            
            # Initialize CopulaGAN synthesizer
            self._synthesizer = CopulaGANSynthesizer(
                metadata=self.metadata,
                epochs=epochs,
                batch_size=min(500, len(real_data)),
                verbose=verbose
            )
            
            # Train model
            self._synthesizer.fit(real_data)
            self._is_trained = True
            
            logger.info(f"CopulaGAN training complete for domain: {self.domain}")
            
        except Exception as e:
            logger.error(f"CopulaGAN training failed: {e}")
            self._is_trained = True
    
    def generate(self, config: SyntheticConfig) -> pd.DataFrame:
        """
        Generate synthetic A/B tests with preserved correlations.
        
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
                logger.info(f"Generating {n} synthetic experiments with CopulaGAN...")
                synthetic_df = self._synthesizer.sample(num_rows=n)
                synthetic_df = self._apply_constraints(synthetic_df)
                return self._post_process(synthetic_df, config)
            except Exception as e:
                logger.warning(f"CopulaGAN generation failed, using fallback: {e}")
        
        # Fallback: correlation-aware statistical sampling
        return self._generate_fallback(config)
    
    def _generate_fallback(self, config: SyntheticConfig) -> pd.DataFrame:
        """Generate synthetic data using correlation-aware sampling."""
        logger.info(f"Using correlation-aware fallback for {config.num_experiments} experiments")
        
        domain_config = DOMAIN_CONFIGS[self.domain]
        n = config.num_experiments
        
        # If we have seed data, use its statistics
        if self._seed_data is not None and len(self._seed_data) > 0:
            return self._generate_from_seed_stats(config)
        
        # Otherwise generate from scratch with expected correlations
        data = []
        for i in range(n):
            row = self._generate_correlated_row(i, config, domain_config)
            data.append(row)
        
        df = pd.DataFrame(data)
        return self._apply_constraints(df)
    
    def _generate_from_seed_stats(self, config: SyntheticConfig) -> pd.DataFrame:
        """Generate using seed data statistics and correlations."""
        seed = self._seed_data
        n = config.num_experiments
        
        # Get numeric columns
        numeric_cols = seed.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Compute correlation matrix and use multivariate normal
            means = seed[numeric_cols].mean().values
            cov = seed[numeric_cols].cov().values
            
            # Make covariance matrix positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-6)
            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            try:
                samples = np.random.multivariate_normal(means, cov, size=n)
                df = pd.DataFrame(samples, columns=numeric_cols)
            except:
                # Fallback to independent sampling
                df = pd.DataFrame({
                    col: np.random.normal(seed[col].mean(), seed[col].std(), n)
                    for col in numeric_cols
                })
        else:
            df = pd.DataFrame({
                col: np.random.normal(seed[col].mean(), seed[col].std(), n)
                for col in numeric_cols
            })
        
        # Add categorical columns
        for col in seed.columns:
            if col not in numeric_cols:
                if seed[col].dtype == 'object' or seed[col].dtype.name == 'category':
                    value_counts = seed[col].value_counts(normalize=True)
                    df[col] = np.random.choice(
                        value_counts.index, 
                        size=n, 
                        p=value_counts.values
                    )
        
        # Ensure required columns exist
        domain_config = DOMAIN_CONFIGS[self.domain]
        
        if "experiment_id" not in df.columns:
            df["experiment_id"] = self._generate_experiment_ids(n, 
                prefix=np.random.choice(domain_config.experiment_prefixes))
        
        if "experiment_name" not in df.columns:
            df["experiment_name"] = [f"Experiment {i+1}" for i in range(n)]
        
        # Add segments if requested
        if config.include_segments:
            segment_values = DomainConstraints.get_segment_values(self.domain)
            for seg_name, seg_options in segment_values.items():
                if seg_name not in df.columns:
                    df[seg_name] = [np.random.choice(seg_options) for _ in range(n)]
        
        return self._apply_constraints(df)
    
    def _generate_correlated_row(self, idx: int, config: SyntheticConfig, 
                                  domain_config) -> Dict[str, Any]:
        """Generate a single row with expected correlations."""
        exp_prefix = np.random.choice(domain_config.experiment_prefixes)
        
        # Sample size
        min_size = max(config.sample_size_range[0], domain_config.sample_size_range[0])
        max_size = min(config.sample_size_range[1], domain_config.sample_size_range[1])
        control_visitors = np.random.randint(min_size // 2, max_size // 2)
        variant_visitors = int(control_visitors * np.random.uniform(0.9, 1.1))
        
        # Base rate with correlation awareness
        primary_metric = domain_config.primary_metric
        metric_range = domain_config.metric_ranges.get(primary_metric, (0.05, 0.20))
        base_rate = np.random.uniform(metric_range[0], metric_range[1])
        
        # Effect size with heterogeneity
        effect_min, effect_max = config.effect_size_range
        het_factors = {"low": 0.5, "medium": 1.0, "high": 1.5}
        het_factor = het_factors.get(config.heterogeneity_level, 1.0)
        
        effect = np.random.uniform(effect_min, effect_max) * het_factor
        if np.random.random() < 0.25:
            effect = -effect * 0.5
        
        treatment_rate = np.clip(base_rate * (1 + effect), 0.001, 0.999)
        
        # Generate conversions
        control_conversions = np.random.binomial(control_visitors, base_rate)
        variant_conversions = np.random.binomial(variant_visitors, treatment_rate)
        
        # Duration
        dur_min, dur_max = domain_config.test_duration_range
        test_duration = np.random.randint(dur_min, dur_max + 1)
        
        row = {
            "experiment_id": f"{exp_prefix}_{self.domain}_{idx+1:04d}",
            "experiment_name": f"{exp_prefix.title()} Test {idx+1}",
            "variant": "A/B",
            "control_visitors": control_visitors,
            "control_conversions": control_conversions,
            "variant_visitors": variant_visitors,
            "variant_conversions": variant_conversions,
            "test_duration_days": test_duration,
            "platform": self.domain,
            "domain": self.domain
        }
        
        # Add segments
        if config.include_segments:
            segment_values = DomainConstraints.get_segment_values(self.domain)
            for seg_name, seg_options in segment_values.items():
                row[seg_name] = np.random.choice(seg_options)
        
        return row
    
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
        
        logger.info(f"Saved CopulaGAN model to {save_path}")
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
            
            logger.info(f"Loaded CopulaGAN model from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load CopulaGAN model: {e}")
            return False
