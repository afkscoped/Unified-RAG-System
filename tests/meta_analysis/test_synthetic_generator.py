"""
Tests for Synthetic Data Generator
"""

import unittest
import pandas as pd
import numpy as np

from src.meta_analysis.utils.synthetic_generator import (
    SyntheticABTestGenerator,
    DOMAIN_CONFIGS,
    DomainType
)


class TestSyntheticGenerator(unittest.TestCase):
    
    def setUp(self):
        self.generator = SyntheticABTestGenerator(random_seed=42)
    
    def test_generate_base_data_shape(self):
        """Test that generated data has correct shape."""
        n_experiments = 10
        df = self.generator._generate_base_data(n_experiments, "marketing")
        
        self.assertEqual(len(df), n_experiments)
        self.assertIn("experiment_name", df.columns)
        self.assertIn("control_visitors", df.columns)
        self.assertIn("control_conversions", df.columns)
        self.assertIn("variant_visitors", df.columns)
        self.assertIn("variant_conversions", df.columns)
    
    def test_generate_columns_are_correct_type(self):
        """Test that numeric columns are integers."""
        df = self.generator._generate_base_data(20, "product")
        
        self.assertTrue(pd.api.types.is_integer_dtype(df["control_visitors"]))
        self.assertTrue(pd.api.types.is_integer_dtype(df["control_conversions"]))
        self.assertTrue(pd.api.types.is_integer_dtype(df["variant_visitors"]))
        self.assertTrue(pd.api.types.is_integer_dtype(df["variant_conversions"]))
    
    def test_conversions_not_exceed_visitors(self):
        """Test that conversions never exceed visitors."""
        df = self.generator._generate_base_data(50, "email")
        
        self.assertTrue((df["control_conversions"] <= df["control_visitors"]).all())
        self.assertTrue((df["variant_conversions"] <= df["variant_visitors"]).all())
    
    def test_different_domains(self):
        """Test generation for all domains."""
        for domain in ["marketing", "product", "email", "ux"]:
            df = self.generator._generate_base_data(5, domain)
            self.assertEqual(len(df), 5)
            self.assertTrue(df["control_visitors"].min() >= DOMAIN_CONFIGS[domain]["visitors_range"][0])
    
    def test_generate_method_random(self):
        """Test the main generate method with random fallback."""
        df = self.generator.generate(15, "marketing", "random")
        
        self.assertEqual(len(df), 15)
        self.assertEqual(len(df.columns), 5)
    
    def test_positive_values(self):
        """Test that all numeric values are positive."""
        df = self.generator._generate_base_data(30, "ux")
        
        self.assertTrue((df["control_visitors"] > 0).all())
        self.assertTrue((df["variant_visitors"] > 0).all())
        self.assertTrue((df["control_conversions"] >= 0).all())
        self.assertTrue((df["variant_conversions"] >= 0).all())
    
    def test_experiment_names_unique(self):
        """Test that experiment names are unique."""
        df = self.generator._generate_base_data(100, "marketing")
        self.assertEqual(len(df["experiment_name"].unique()), len(df))


class TestDomainConfigs(unittest.TestCase):
    
    def test_all_domains_have_required_keys(self):
        """Test that all domain configs have required keys."""
        required_keys = ["visitors_range", "base_conversion_rate", "lift_range", "experiment_prefixes"]
        
        for domain, config in DOMAIN_CONFIGS.items():
            for key in required_keys:
                self.assertIn(key, config, f"Domain {domain} missing key {key}")
    
    def test_ranges_are_valid(self):
        """Test that ranges are proper tuples with min < max."""
        for domain, config in DOMAIN_CONFIGS.items():
            self.assertLess(config["visitors_range"][0], config["visitors_range"][1])
            self.assertLess(config["base_conversion_rate"][0], config["base_conversion_rate"][1])
            self.assertLess(config["lift_range"][0], config["lift_range"][1])


if __name__ == '__main__':
    unittest.main()
