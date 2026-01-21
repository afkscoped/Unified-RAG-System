"""
Tests for Kaggle Importer
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from src.meta_analysis.utils.kaggle_importer import (
    KaggleABTestImporter,
    AB_COLUMN_PATTERNS
)


class TestKaggleImporter(unittest.TestCase):
    
    def setUp(self):
        # Disable actual Kaggle API
        with patch.object(KaggleABTestImporter, '_check_kaggle', return_value=False):
            self.importer = KaggleABTestImporter()
    
    def test_detect_ab_columns_standard(self):
        """Test column detection with standard column names."""
        df = pd.DataFrame({
            "experiment_name": ["exp1", "exp2"],
            "control_visitors": [100, 200],
            "control_conversions": [10, 20],
            "variant_visitors": [100, 200],
            "variant_conversions": [15, 25]
        })
        
        detected = self.importer.detect_ab_columns(df)
        
        self.assertEqual(detected["experiment_name"], "experiment_name")
        self.assertEqual(detected["control_visitors"], "control_visitors")
        self.assertEqual(detected["control_conversions"], "control_conversions")
        self.assertEqual(detected["variant_visitors"], "variant_visitors")
        self.assertEqual(detected["variant_conversions"], "variant_conversions")
    
    def test_detect_ab_columns_alternate_names(self):
        """Test column detection with alternate column names."""
        df = pd.DataFrame({
            "campaign": ["camp1", "camp2"],
            "control_users": [100, 200],
            "control_success": [10, 20],
            "treatment_users": [100, 200],
            "treatment_success": [15, 25]
        })
        
        detected = self.importer.detect_ab_columns(df)
        
        self.assertEqual(detected["experiment_name"], "campaign")
        self.assertEqual(detected["control_visitors"], "control_users")
        self.assertEqual(detected["control_conversions"], "control_success")
        self.assertEqual(detected["variant_visitors"], "treatment_users")
        self.assertEqual(detected["variant_conversions"], "treatment_success")
    
    def test_clean_and_validate_success(self):
        """Test successful data cleaning."""
        df = pd.DataFrame({
            "experiment_name": ["exp1", "exp2", "exp3"],
            "control_visitors": [100, 200, 150],
            "control_conversions": [10, 20, 15],
            "variant_visitors": [100, 200, 150],
            "variant_conversions": [15, 25, 20]
        })
        
        success, msg, cleaned = self.importer.clean_and_validate(df)
        
        self.assertTrue(success)
        self.assertEqual(len(cleaned), 3)
        self.assertIn("control_total", cleaned.columns)
        self.assertIn("treatment_total", cleaned.columns)
    
    def test_clean_and_validate_removes_invalid_rows(self):
        """Test that invalid rows are removed during cleaning."""
        df = pd.DataFrame({
            "experiment_name": ["exp1", "exp2", "exp3"],
            "control_visitors": [100, 200, 0],  # 0 is invalid
            "control_conversions": [10, 20, 15],
            "variant_visitors": [100, 200, 150],
            "variant_conversions": [15, 250, 20]  # 250 > 200 is invalid
        })
        
        success, msg, cleaned = self.importer.clean_and_validate(df)
        
        self.assertTrue(success)
        self.assertEqual(len(cleaned), 1)  # Only exp1 is valid
    
    def test_clean_and_validate_missing_columns(self):
        """Test that missing required columns causes failure."""
        df = pd.DataFrame({
            "experiment_name": ["exp1"],
            "control_visitors": [100],
            # Missing other required columns
        })
        
        success, msg, cleaned = self.importer.clean_and_validate(df)
        
        self.assertFalse(success)
        self.assertIn("Missing required columns", msg)
        self.assertIsNone(cleaned)
    
    def test_get_sample_datasets(self):
        """Test that sample datasets are returned."""
        samples = KaggleABTestImporter.get_sample_datasets()
        
        self.assertIsInstance(samples, list)
        self.assertTrue(len(samples) > 0)
        self.assertIn("id", samples[0])
        self.assertIn("name", samples[0])


class TestABColumnPatterns(unittest.TestCase):
    
    def test_all_required_patterns_exist(self):
        """Test that patterns exist for all required column types."""
        required = [
            "experiment_name", 
            "control_visitors", 
            "control_conversions",
            "variant_visitors", 
            "variant_conversions"
        ]
        
        for col in required:
            self.assertIn(col, AB_COLUMN_PATTERNS)
            self.assertTrue(len(AB_COLUMN_PATTERNS[col]) > 0)


if __name__ == '__main__':
    unittest.main()
