
import sys
import os
import pandas as pd
import numpy as np
from loguru import logger

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


def test_imports():
    print("Testing Imports...")
    try:
        from src.meta_analysis.mcp_servers.csv_experiment_server import CSVExperimentMCP
        from src.meta_analysis.statistical.meta_analyzer import MetaAnalyzer
        from src.meta_analysis.synthetic.generators.ctgan_generator import CTGANGenerator
        from src.meta_analysis.utils.kaggle_importer import KaggleABTestImporter
        from src.meta_analysis.ui.components.simple_view_renderer import render_simple_view
        print("[OK] All modules imported successfully.")
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        sys.exit(1)

def test_statistical_engine():
    print("\nTesting Statistical Engine...")
    from src.meta_analysis.statistical.meta_analyzer import MetaAnalyzer
    from src.meta_analysis.mcp_servers.base_experiment_mcp import StandardizedStudy, MetricType, EffectSizeType
    
    analyzer = MetaAnalyzer()
    
    # Create dummy studies
    studies = [
        StandardizedStudy(
            study_id="1", study_name="A", effect_size=0.5, standard_error=0.1, 
            sample_size_control=100, sample_size_treatment=100, metric_name="conv"
        ),
        StandardizedStudy(
            study_id="2", study_name="B", effect_size=0.6, standard_error=0.1, 
            sample_size_control=100, sample_size_treatment=100, metric_name="conv"
        )
    ]
    
    res = analyzer.analyze(studies, model="fixed")
    print(f"[OK] Fixed Effects Pooled: {res.pooled_effect:.3f}")
    
    res_rand = analyzer.analyze(studies, model="random")
    print(f"[OK] Random Effects Pooled: {res_rand.pooled_effect:.3f}")

def test_synthetic_engine():
    print("\nTesting Synthetic Engine (Fallback Mode)...")
    from src.meta_analysis.synthetic.generators.ctgan_generator import CTGANGenerator
    from src.meta_analysis.synthetic.generators.base_generator import SyntheticConfig
    
    try:
        gen = CTGANGenerator("ecommerce")
        # Fake training
        gen._is_trained = True 
        
        config = SyntheticConfig(
            domain="ecommerce",
            num_experiments=5,
            model_type="ctgan",
            effect_size_range=(0.01, 0.05),
            heterogeneity_level="low",
            include_segments=False
        )
        
        df = gen.generate(config)
        print(f"[OK] Generated {len(df)} synthetic experiments.")
    except Exception as e:
        print(f"[FAIL] Synthetic generation failed: {e}")


if __name__ == "__main__":
    test_imports()
    test_statistical_engine()
    test_synthetic_engine()
