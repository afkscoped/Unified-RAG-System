"""
Tests for Statistical Engine
"""

import unittest
import numpy as np
from src.meta_analysis.mcp_servers.base_experiment_mcp import StandardizedStudy, EffectSizeType
from src.meta_analysis.statistical.meta_analyzer import MetaAnalyzer
from src.meta_analysis.statistical.publication_bias import PublicationBiasDetector
from src.meta_analysis.statistical.sensitivity_analysis import SensitivityAnalyzer


class TestMetaAnalyzer(unittest.TestCase):
    
    def setUp(self):
        self.analyzer = MetaAnalyzer()
        
        # Creates a synthetic dataset
        # True effect = 0.5
        self.studies = [
            StandardizedStudy(
                study_id="s1", study_name="Study 1", 
                effect_size=0.5, standard_error=0.1, 
                sample_size_control=100, sample_size_treatment=100,
                metric_name="conv", metric_type="rate", effect_size_type=EffectSizeType.LOG_ODDS_RATIO,
                platform="csv", timestamp=None
            ),
            StandardizedStudy(
                study_id="s2", study_name="Study 2", 
                effect_size=0.6, standard_error=0.1, 
                sample_size_control=100, sample_size_treatment=100,
                metric_name="conv", metric_type="rate", effect_size_type=EffectSizeType.LOG_ODDS_RATIO,
                platform="csv", timestamp=None
            ),
            StandardizedStudy(
                study_id="s3", study_name="Study 3", 
                effect_size=0.4, standard_error=0.2, 
                sample_size_control=50, sample_size_treatment=50,
                metric_name="conv", metric_type="rate", effect_size_type=EffectSizeType.LOG_ODDS_RATIO,
                platform="csv", timestamp=None
            )
        ]
        
    def test_fixed_effects(self):
        result = self.analyzer.fixed_effects(self.studies)
        
        # Check weighted average logic manually
        # weights = 1/0.1^2, 1/0.1^2, 1/0.2^2 = 100, 100, 25
        # sum weights = 225
        # weighted sum = 100*0.5 + 100*0.6 + 25*0.4 = 50 + 60 + 10 = 120
        # pooled = 120 / 225 = 0.5333
        
        self.assertAlmostEqual(result.pooled_effect, 0.5333, places=3)
        self.assertEqual(result.model_type, "fixed")
        self.assertLess(result.p_value, 0.05)  # Should be significant
        
    def test_heterogeneity(self):
        # Create heterogeneous studies
        het_studies = [
            StandardizedStudy(
                study_id="h1", study_name="H1", effect_size=0.1, standard_error=0.1,
                sample_size_control=100, sample_size_treatment=100,
                metric_name="m", metric_type="rate", effect_size_type=EffectSizeType.LOG_ODDS_RATIO,
                platform="csv", timestamp=None
            ),
            StandardizedStudy(
                study_id="h2", study_name="H2", effect_size=0.9, standard_error=0.1,
                sample_size_control=100, sample_size_treatment=100,
                metric_name="m", metric_type="rate", effect_size_type=EffectSizeType.LOG_ODDS_RATIO,
                platform="csv", timestamp=None
            )
        ]
        
        result = self.analyzer.calculate_heterogeneity(het_studies)
        
        self.assertGreater(result["I2"], 50)  # Should be high heterogeneity
        self.assertGreater(result["Q"], 0)
        
    def test_random_effects_selection(self):
        # Should auto-select random effects for heterogeneous data
        het_studies = [
            StandardizedStudy(
                study_id="h1", study_name="H1", effect_size=0.1, standard_error=0.05,
                sample_size_control=100, sample_size_treatment=100,
                metric_name="m", metric_type="rate", effect_size_type=EffectSizeType.LOG_ODDS_RATIO,
                platform="csv", timestamp=None
            ),
            StandardizedStudy(
                study_id="h2", study_name="H2", effect_size=0.9, standard_error=0.05,
                sample_size_control=100, sample_size_treatment=100,
                metric_name="m", metric_type="rate", effect_size_type=EffectSizeType.LOG_ODDS_RATIO,
                platform="csv", timestamp=None
            )
        ]
        
        result = self.analyzer.analyze(het_studies, model="auto")
        self.assertEqual(result.model_type, "random")


class TestPublicationBias(unittest.TestCase):
    
    def setUp(self):
        self.detector = PublicationBiasDetector()
        
    def test_eggers_test_no_bias(self):
        # Symmetric funnel
        studies = [
            StandardizedStudy("s1", "S1", 0.5, 0.1, 100, 100, "m", "t", EffectSizeType.LOG_ODDS_RATIO, "p", None),
            StandardizedStudy("s2", "S2", 0.5, 0.2, 50, 50, "m", "t", EffectSizeType.LOG_ODDS_RATIO, "p", None),
            StandardizedStudy("s3", "S3", 0.5, 0.05, 400, 400, "m", "t", EffectSizeType.LOG_ODDS_RATIO, "p", None),
            StandardizedStudy("s4", "S4", 0.45, 0.15, 80, 80, "m", "t", EffectSizeType.LOG_ODDS_RATIO, "p", None),
            StandardizedStudy("s5", "S5", 0.55, 0.15, 80, 80, "m", "t", EffectSizeType.LOG_ODDS_RATIO, "p", None),
        ]
        
        result = self.detector.eggers_test(studies)
        self.assertFalse(result["significant"])
        
    def test_failsafe_n(self):
        # Significant study setup
        studies = [
            StandardizedStudy("s1", "S1", 0.8, 0.1, 100, 100, "m", "t", EffectSizeType.LOG_ODDS_RATIO, "p", None),
            StandardizedStudy("s2", "S2", 0.9, 0.1, 100, 100, "m", "t", EffectSizeType.LOG_ODDS_RATIO, "p", None),
        ]
        
        n = self.detector.failsafe_n(studies)
        self.assertGreater(n, 0)


class TestSensitivityAnalysis(unittest.TestCase):
    
    def setUp(self):
        self.analyzer = SensitivityAnalyzer()
        
    def test_leave_one_out(self):
        studies = [
            StandardizedStudy("s1", "S1", 0.5, 0.1, 100, 100, "m", "t", EffectSizeType.LOG_ODDS_RATIO, "p", None),
            StandardizedStudy("s2", "S2", 0.5, 0.1, 100, 100, "m", "t", EffectSizeType.LOG_ODDS_RATIO, "p", None),
            StandardizedStudy("s3", "S3", 5.0, 0.1, 100, 100, "m", "t", EffectSizeType.LOG_ODDS_RATIO, "p", None) # Outlier
        ]
        
        results = self.analyzer.leave_one_out_analysis(studies)
        
        # Removing s3 should change effect drastically
        change_s3 = results["s3"]["relative_change"]
        change_s1 = results["s1"]["relative_change"]
        
        self.assertGreater(change_s3, change_s1)


class TestSimpsonsParadox(unittest.TestCase):
    
    def setUp(self):
        from src.meta_analysis.statistical.simpsons_detector import SimpsonsParadoxDetector
        self.detector = SimpsonsParadoxDetector()
        
    def test_paradox_detection(self):
        # Create a situation where overall effect is positive, but subgroups are negative
        # This usually happens with confounding by weighting
        # Example: 
        # Group A: Low conversions but huge volume (Negative effect)
        # Group B: High conversions but small volume (Negative effect)
        # But if combined naively? Actually, standard meta-analysis usually avoids this if properly weighted.
        # But we can simulate sign flipping.
        
        studies = [
            # Group A (Negative)
            StandardizedStudy("s1", "S1", -0.2, 0.1, 100, 100, "m", "t", EffectSizeType.LOG_ODDS_RATIO, "p", None, metadata={"segment": "A"}),
            StandardizedStudy("s2", "S2", -0.3, 0.1, 100, 100, "m", "t", EffectSizeType.LOG_ODDS_RATIO, "p", None, metadata={"segment": "A"}),
            
            # Group B (Positive and dominates overall due to some artifact or just different effect)
            # To simulate paradox in meta-analysis, we need overall to different from subgroups.
            # Let's say we have Group B with Positive effect.
            StandardizedStudy("s3", "S3", 0.5, 0.05, 100, 100, "m", "t", EffectSizeType.LOG_ODDS_RATIO, "p", None, metadata={"segment": "B"}),
            StandardizedStudy("s4", "S4", 0.6, 0.05, 100, 100, "m", "t", EffectSizeType.LOG_ODDS_RATIO, "p", None, metadata={"segment": "B"})
        ]
        
        # Wait, if Group A is negative and Group B is positive, and overall is positive, 
        # then Group A is "reversed" relative to overall. That IS the paradox definition used in the code.
        
        result = self.detector.detect_simpsons_paradox(studies, segment_key="segment")
        
        self.assertTrue(result.paradox_detected)
        self.assertEqual(len(result.reversal_details), 1)
        self.assertEqual(result.reversal_details[0]["subgroup"], "A")
        
    def test_no_paradox(self):
        studies = [
            StandardizedStudy("s1", "S1", 0.2, 0.1, 100, 100, "m", "t", EffectSizeType.LOG_ODDS_RATIO, "p", None, metadata={"segment": "A"}),
            StandardizedStudy("s3", "S3", 0.5, 0.05, 100, 100, "m", "t", EffectSizeType.LOG_ODDS_RATIO, "p", None, metadata={"segment": "B"})
        ]
        # Both positive, overall positive -> No paradox
        result = self.detector.detect_simpsons_paradox(studies, segment_key="segment")
        self.assertFalse(result.paradox_detected)

if __name__ == '__main__':
    unittest.main()
