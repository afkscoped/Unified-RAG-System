"""
Tests for Data Processing and MCP
"""

import unittest
from pathlib import Path
import tempfile
import pandas as pd
import os

from src.meta_analysis.mcp_servers.csv_experiment_server import CSVExperimentMCP
from src.meta_analysis.statistical.data_harmonizer import DataHarmonizer
from src.meta_analysis.mcp_servers.base_experiment_mcp import EffectSizeType


class TestCSVServer(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.server = CSVExperimentMCP(data_dir=self.temp_dir)
        
        # Create dummy CSV
        self.df = pd.DataFrame({
            "experiment_id": ["e1", "e2"],
            "experiment_name": ["Exp 1", "Exp 2"],
            "control_visitors": [1000, 2000],
            "control_conversions": [100, 200],
            "treatment_visitors": [1000, 2000],
            "treatment_conversions": [120, 210]
        })
        self.csv_path = os.path.join(self.temp_dir, "test_experiments.csv")
        self.df.to_csv(self.csv_path, index=False)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_load_experiments(self):
        studies = self.server.get_experiments()
        self.assertEqual(len(studies), 2)
        
        s1 = studies[0]
        self.assertEqual(s1.study_name, "Exp 1")
        self.assertEqual(s1.sample_size_control, 1000)
        
        # Check if effect size calculated
        self.assertIsNotNone(s1.effect_size)
        self.assertIsNotNone(s1.standard_error)


class TestHarmonizer(unittest.TestCase):
    
    def setUp(self):
        self.harmonizer = DataHarmonizer()
        
    def test_effect_conversion(self):
        # Convert Cohen's d to Log Odds
        d = 0.5
        log_odds, _ = self.harmonizer._convert_effect_size(
            d, 0.1, EffectSizeType.COHENS_D, 100
        )
        # d * pi / sqrt(3) approx 1.81 * d
        expected = d * 1.8138
        self.assertAlmostEqual(log_odds, expected, places=1)

if __name__ == '__main__':
    unittest.main()
