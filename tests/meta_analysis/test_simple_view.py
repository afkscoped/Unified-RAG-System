"""
Tests for Simple View Renderer
"""

import unittest
from unittest.mock import MagicMock

from src.meta_analysis.ui.components.simple_view_renderer import (
    get_result_color,
    get_confidence_level,
    get_heterogeneity_description
)


class TestColorCoding(unittest.TestCase):
    
    def test_green_significant_positive(self):
        """Test green color for significant positive effect."""
        color = get_result_color(
            pooled_effect=0.5,
            p_value=0.01,
            ci_lower=0.2,
            ci_upper=0.8
        )
        self.assertEqual(color, "green")
    
    def test_red_significant_negative(self):
        """Test red color for significant negative effect."""
        color = get_result_color(
            pooled_effect=-0.5,
            p_value=0.01,
            ci_lower=-0.8,
            ci_upper=-0.2
        )
        self.assertEqual(color, "red")
    
    def test_yellow_not_significant(self):
        """Test yellow color for non-significant result."""
        color = get_result_color(
            pooled_effect=0.5,
            p_value=0.10,  # Not significant
            ci_lower=0.2,
            ci_upper=0.8
        )
        self.assertEqual(color, "yellow")
    
    def test_yellow_ci_crosses_zero(self):
        """Test yellow color when CI crosses zero."""
        color = get_result_color(
            pooled_effect=0.1,
            p_value=0.04,  # Significant, but CI crosses zero
            ci_lower=-0.1,
            ci_upper=0.3
        )
        self.assertEqual(color, "yellow")


class TestConfidenceLevel(unittest.TestCase):
    
    def test_very_high_confidence(self):
        """Test very high confidence for p < 0.001."""
        level = get_confidence_level(0.0005)
        self.assertEqual(level, "Very High")
    
    def test_high_confidence(self):
        """Test high confidence for p < 0.01."""
        level = get_confidence_level(0.005)
        self.assertEqual(level, "High")
    
    def test_moderate_confidence(self):
        """Test moderate confidence for p < 0.05."""
        level = get_confidence_level(0.03)
        self.assertEqual(level, "Moderate")
    
    def test_low_confidence(self):
        """Test low confidence for p < 0.10."""
        level = get_confidence_level(0.08)
        self.assertEqual(level, "Low")
    
    def test_very_low_confidence(self):
        """Test very low confidence for p >= 0.10."""
        level = get_confidence_level(0.15)
        self.assertEqual(level, "Very Low")


class TestHeterogeneityDescription(unittest.TestCase):
    
    def test_low_heterogeneity(self):
        """Test description for low I2."""
        desc = get_heterogeneity_description(20)
        self.assertIn("Low", desc)
        self.assertIn("consistent", desc.lower())
    
    def test_moderate_heterogeneity(self):
        """Test description for moderate I2."""
        desc = get_heterogeneity_description(40)
        self.assertIn("Moderate", desc)
    
    def test_substantial_heterogeneity(self):
        """Test description for substantial I2."""
        desc = get_heterogeneity_description(60)
        self.assertIn("Substantial", desc)
    
    def test_high_heterogeneity(self):
        """Test description for high I2."""
        desc = get_heterogeneity_description(80)
        self.assertIn("High", desc)
        self.assertIn("different", desc.lower())


if __name__ == '__main__':
    unittest.main()
