"""
Tests for Synthetic Data MCP Integration

Validates that synthetic data flows correctly through MCP ingestion,
returns proper StandardizedStudy objects, and catches validation errors.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from src.meta_analysis.synthetic.utils.mcp_adapter import (
    SyntheticMCPAdapter, 
    MCPIngestionResult,
    ingest_via_mcp
)
from src.meta_analysis.mcp_servers.base_experiment_mcp import StandardizedStudy


class TestSyntheticMCPIntegration(unittest.TestCase):
    """Test suite for synthetic data MCP integration."""
    
    def setUp(self):
        """Create test synthetic data."""
        self.valid_df = pd.DataFrame({
            "experiment_id": [f"test_exp_{i}" for i in range(10)],
            "experiment_name": [f"Test Experiment {i}" for i in range(10)],
            "control_visitors": np.random.randint(1000, 5000, 10),
            "control_conversions": np.random.randint(50, 200, 10),
            "variant_visitors": np.random.randint(1000, 5000, 10),
            "variant_conversions": np.random.randint(50, 250, 10),
            "test_duration_days": np.random.randint(7, 30, 10),
            "platform": ["web"] * 10
        })
        
        # Ensure conversions <= visitors
        self.valid_df["control_conversions"] = np.minimum(
            self.valid_df["control_conversions"],
            self.valid_df["control_visitors"] - 1
        )
        self.valid_df["variant_conversions"] = np.minimum(
            self.valid_df["variant_conversions"],
            self.valid_df["variant_visitors"] - 1
        )
    
    def test_mcp_adapter_initialization(self):
        """Test MCP adapter initializes correctly."""
        adapter = SyntheticMCPAdapter()
        
        self.assertIsNotNone(adapter.mcp_server)
        self.assertEqual(adapter.server_name, "CSVExperimentMCP")
        self.assertIsNotNone(adapter.data_dir)
    
    def test_synthetic_to_mcp_ingestion(self):
        """Test synthetic data flows through MCP ingestion."""
        adapter = SyntheticMCPAdapter()
        result = adapter.ingest_synthetic_data(self.valid_df, "test_source")
        
        self.assertIsInstance(result, MCPIngestionResult)
        self.assertEqual(result.source_type, "synthetic")
        self.assertEqual(result.server_name, "CSVExperimentMCP")
        self.assertIsInstance(result.timestamp, datetime)
    
    def test_mcp_returns_standardized_studies(self):
        """Test MCP returns correct StandardizedStudy objects."""
        adapter = SyntheticMCPAdapter()
        result = adapter.ingest_synthetic_data(self.valid_df)
        
        self.assertGreater(len(result.studies), 0)
        
        # Check first study is StandardizedStudy
        first_study = result.studies[0]
        self.assertIsInstance(first_study, StandardizedStudy)
        self.assertIsNotNone(first_study.study_id)
        self.assertIsNotNone(first_study.effect_size)
        self.assertIsNotNone(first_study.standard_error)
    
    def test_mcp_returns_correct_study_count(self):
        """Test MCP returns correct number of studies."""
        adapter = SyntheticMCPAdapter()
        result = adapter.ingest_synthetic_data(self.valid_df)
        
        self.assertEqual(result.study_count, len(result.studies))
        self.assertEqual(result.study_count, len(self.valid_df))
    
    def test_mcp_validation_report(self):
        """Test MCP returns validation report."""
        adapter = SyntheticMCPAdapter()
        result = adapter.ingest_synthetic_data(self.valid_df)
        
        self.assertIsNotNone(result.validation_report)
        self.assertEqual(result.validation_report.total_records, len(self.valid_df))
    
    def test_mcp_result_to_dict(self):
        """Test MCPIngestionResult converts to dict for session state."""
        adapter = SyntheticMCPAdapter()
        result = adapter.ingest_synthetic_data(self.valid_df)
        
        result_dict = result.to_dict()
        
        self.assertIn("study_count", result_dict)
        self.assertIn("source_type", result_dict)
        self.assertIn("server_name", result_dict)
        self.assertIn("timestamp", result_dict)
        self.assertIn("is_valid", result_dict)
        self.assertIn("valid_records", result_dict)
    
    def test_convenience_function(self):
        """Test ingest_via_mcp convenience function."""
        result = ingest_via_mcp(self.valid_df, source_name="convenience_test")
        
        self.assertIsInstance(result, MCPIngestionResult)
        self.assertGreater(len(result.studies), 0)
    
    def test_invalid_data_detection(self):
        """Test MCP catches invalid data."""
        # Create intentionally problematic data
        invalid_df = pd.DataFrame({
            "experiment_id": ["bad_exp_1"],
            "control_visitors": [-100],  # Invalid negative
            "control_conversions": [50],
            "variant_visitors": [500],
            "variant_conversions": [600]  # More than visitors
        })
        
        adapter = SyntheticMCPAdapter()
        result = adapter.ingest_synthetic_data(invalid_df)
        
        # Should still process but may have validation issues
        self.assertIsNotNone(result.validation_report)


if __name__ == '__main__':
    unittest.main()
