"""
Tests for Enhanced Synthetic Data Generator
Covers 5 domains, CTGAN & CopulaGAN generators, validation metrics, and constraints.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from src.meta_analysis.synthetic.generators.ctgan_generator import CTGANGenerator
from src.meta_analysis.synthetic.generators.copulagan_generator import CopulaGANGenerator
from src.meta_analysis.synthetic.generators.base_generator import SyntheticConfig
from src.meta_analysis.synthetic.domains.domain_constraints import DomainConstraints, DOMAIN_CONFIGS
from src.meta_analysis.synthetic.validation.quality_metrics import QualityValidator
from src.meta_analysis.synthetic.utils.schema_mapper import SchemaMapper


class TestEnhancedSyntheticGenerator(unittest.TestCase):
    
    def setUp(self):
        # Create a small seed dataset for testing
        self.seed_data = pd.DataFrame({
            "experiment_id": [f"exp_{i}" for i in range(20)],
            "variant": ["A/B"] * 20,
            "control_visitors": np.random.randint(1000, 5000, 20),
            "control_conversions": np.random.randint(50, 200, 20),
            "variant_visitors": np.random.randint(1000, 5000, 20),
            "variant_conversions": np.random.randint(50, 250, 20),
            "test_duration_days": np.random.randint(7, 30, 20),
            "platform": ["web"] * 20,
            "device_type": np.random.choice(["mobile", "desktop"], 20),
            "region": np.random.choice(["US", "EU"], 20)
        })
        
        self.domains = ["ecommerce", "saas", "healthcare", "marketing", "edtech"]
        
    def test_domain_constraints_loading(self):
        """Test that all 5 domains have valid configs."""
        for domain in self.domains:
            config = DomainConstraints.get_config(domain)
            self.assertIsNotNone(config)
            self.assertGreater(len(config.metric_ranges), 0)
            self.assertGreater(len(config.required_segments), 0)
    
    def test_constraint_enforcement(self):
        """Test constraints are applied correctly."""
        # Create data violating constraints (negative values, conversions > visitors)
        bad_data = pd.DataFrame({
            "control_visitors": [100, -50],
            "control_conversions": [150, 10],  # 150 > 100
            "conversion_rate": [0.5, 1.5],     # > 1.0
            "test_duration_days": [2, 1000]    # Out of range
        })
        
        constrained = DomainConstraints.enforce_constraints(bad_data, "ecommerce")
        
        # Check corrections
        self.assertTrue((constrained["control_conversions"] <= constrained["control_visitors"]).all())
        self.assertTrue((constrained["control_visitors"] >= 0).all())
        self.assertLessEqual(constrained["conversion_rate"].max(), 0.999)
        self.assertTrue(constrained["test_duration_days"].between(7, 30).all())

    def test_ctgan_generator_initialization(self):
        """Test CTGAN generator initialization."""
        generator = CTGANGenerator("ecommerce")
        self.assertEqual(generator.domain, "ecommerce")
        self.assertFalse(generator.is_trained)
        
    def test_copulagan_generator_initialization(self):
        """Test CopulaGAN generator initialization."""
        generator = CopulaGANGenerator("healthcare")
        self.assertEqual(generator.domain, "healthcare")
        self.assertFalse(generator.is_trained)
        
    def test_training_and_generation_flow(self):
        """Test full training and generation flow (using fallback/fast mode)."""
        generator = CTGANGenerator("marketing")
        
        # Force fallback to avoid long training during test
        generator._sdv_available = False 
        
        # Train
        generator.fit(self.seed_data, epochs=1)
        self.assertTrue(generator.is_trained)
        
        # Generate
        config = SyntheticConfig(
            domain="marketing",
            num_experiments=10,
            model_type="ctgan",
            include_segments=True
        )
        synthetic = generator.generate(config)
        
        self.assertEqual(len(synthetic), 10)
        self.assertIn("experiment_id", synthetic.columns)
        self.assertIn("ad_format", synthetic.columns) # From segments
        
        # Validate
        report = generator.validate(synthetic, self.seed_data)
        self.assertIsNotNone(report)
        self.assertIsInstance(report.ks_statistic, float)
    
    def test_quality_metrics(self):
        """Test validation metrics calculation."""
        real = np.random.normal(0, 1, 100)
        synth_good = np.random.normal(0, 1, 100)
        synth_bad = np.random.normal(5, 1, 100)
        
        # KS Test
        ks_good, _ = QualityValidator.kolmogorov_smirnov(real, synth_good)
        ks_bad, _ = QualityValidator.kolmogorov_smirnov(real, synth_bad)
        self.assertLess(ks_good, ks_bad)
        
        # JS Divergence
        js_good = QualityValidator.jensen_shannon_divergence(real, synth_good)
        js_bad = QualityValidator.jensen_shannon_divergence(real, synth_bad)
        self.assertLess(js_good, js_bad)

    def test_model_persistence(self):
        """Test saving and loading models."""
        generator = CTGANGenerator("saas")
        generator.fit(self.seed_data, epochs=1)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_model.pkl"
            generator.save_model(save_path)
            
            self.assertTrue(save_path.exists())
            
            # Load into new instance
            new_gen = CTGANGenerator("saas")
            success = new_gen.load_model(save_path)
            
            self.assertTrue(success)
            self.assertTrue(new_gen.is_trained)
            self.assertIsNotNone(new_gen._seed_data)

    def test_schema_mapper(self):
        """Test schema mapping to standardized format."""
        df = pd.DataFrame({
            "experiment_id": ["1"],
            "users": [1000],          # saas specific
            "activations": [100],     # saas specific
            "platform": ["web"]
        })
        
        mapped = SchemaMapper.from_domain_format(df, "saas")
        
        self.assertIn("variant_visitors", mapped.columns)     # users -> variant_visitors
        self.assertIn("variant_conversions", mapped.columns)  # activations -> variant_conversions
        self.assertEqual(mapped["variant_visitors"][0], 1000)

if __name__ == '__main__':
    unittest.main()
