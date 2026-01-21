
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

def test_hartung_knapp():
    print("Testing Hartung-Knapp Adjustment...")
    from src.meta_analysis.statistical.meta_analyzer import MetaAnalyzer
    from src.meta_analysis.mcp_servers.base_experiment_mcp import StandardizedStudy
    
    analyzer = MetaAnalyzer()
    
    # Create small dummy studies (HK is useful for small K)
    studies = [
        StandardizedStudy(study_id="1", study_name="A", effect_size=0.5, standard_error=0.1, sample_size_control=50, sample_size_treatment=50, metric_name="conv"),
        StandardizedStudy(study_id="2", study_name="B", effect_size=0.6, standard_error=0.1, sample_size_control=50, sample_size_treatment=50, metric_name="conv"),
        StandardizedStudy(study_id="3", study_name="C", effect_size=0.4, standard_error=0.1, sample_size_control=50, sample_size_treatment=50, metric_name="conv")
    ]
    
    # Run regular Random Effects (DL)
    res_dl = analyzer.random_effects_dl(studies)
    width_dl = res_dl.confidence_interval[1] - res_dl.confidence_interval[0]
    
    # Run Hartung-Knapp
    res_hk = analyzer.random_effects_hartung_knapp(studies)
    width_hk = res_hk.confidence_interval[1] - res_hk.confidence_interval[0]
    
    print(f"[OK] DL Width: {width_dl:.4f}")
    print(f"[OK] HK Width: {width_hk:.4f}")
    
    if res_hk.model_type == "random_hk":
        print("[OK] Model type correctly set.")
    else:
        print("[FAIL] Incorrect model type.")

def test_visualizations():
    print("\nTesting Visualization Generation...")
    try:
        from src.meta_analysis.statistical.visualization import MetaAnalysisVisualizer
        from src.meta_analysis.statistical.meta_analyzer import MetaAnalyzer
        from src.meta_analysis.mcp_servers.base_experiment_mcp import StandardizedStudy
        
        analyzer = MetaAnalyzer()
        viz = MetaAnalysisVisualizer()
        
        studies = [
            StandardizedStudy(study_id="1", study_name="A", effect_size=0.5, standard_error=0.05, sample_size_control=100, sample_size_treatment=100, metric_name="conv"),
            StandardizedStudy(study_id="2", study_name="B", effect_size=0.6, standard_error=0.06, sample_size_control=100, sample_size_treatment=100, metric_name="conv"),
            StandardizedStudy(study_id="3", study_name="C", effect_size=0.2, standard_error=0.08, sample_size_control=100, sample_size_treatment=100, metric_name="conv")
        ]
        
        result = analyzer.analyze(studies, model="random")
        
        # Test Bubble Plot
        fig_bubble = viz.create_bubble_plot(result)
        if fig_bubble:
            print(f"[OK] Bubble Plot generated.")
            
        # Test Heterogeneity Chart
        fig_het = viz.create_heterogeneity_chart(result)
        if fig_het:
            print(f"[OK] Heterogeneity Chart generated.")
            
    except Exception as e:
        print(f"[FAIL] Visualization test failed: {e}")

if __name__ == "__main__":
    test_hartung_knapp()
    test_visualizations()
