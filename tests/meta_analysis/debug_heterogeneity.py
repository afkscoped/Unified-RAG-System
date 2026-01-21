from src.meta_analysis.statistical.meta_analyzer import MetaAnalyzer
from src.meta_analysis.mcp_servers.base_experiment_mcp import StandardizedStudy
import numpy as np

def test_high_heterogeneity():
    print("--- DIAGNOSTIC TEST: HIGH HETEROGENEITY ---")
    
    # Create studies with unequal precision
    # Study 1: Small, Imprecise (SE=0.5), Effect=0.1
    var1 = 0.25 
    s1 = StandardizedStudy(
        study_id="s1", 
        study_name="Study 1 (Small)",
        effect_size=0.1, 
        standard_error=np.sqrt(var1),
        sample_size_control=20,
        sample_size_treatment=20,
        metric_name="conversion"
    )
    
    # Study 2: Large, Precise (SE=0.05), Effect=0.9
    var2 = 0.0025
    s2 = StandardizedStudy(
        study_id="s2", 
        study_name="Study 2 (Large)",
        effect_size=0.9, 
        standard_error=np.sqrt(var2),
        sample_size_control=500,
        sample_size_treatment=500,

        metric_name="conversion"
    )
    
    studies = [s1, s2]
    
    analyzer = MetaAnalyzer()
    
    # 1. Calculate Heterogeneity
    het = analyzer.calculate_heterogeneity(studies)
    print(f"Q: {het['Q']}")
    print(f"df: {het['Q_df']}")
    print(f"I2: {het['I2']}%")
    print(f"tau2: {het['tau2']}")
    
    # 2. Run FE
    fe = analyzer.fixed_effects(studies, het)
    print(f"FE Pooled: {fe.pooled_effect}")
    print(f"FE SE: {fe.standard_error}")
    
    # 3. Run RE
    re = analyzer.random_effects_dl(studies, het)
    print(f"RE Pooled: {re.pooled_effect}")
    print(f"RE SE: {re.standard_error}")
    print(f"RE Weights: {list(re.study_weights.values())}")
    
    if abs(fe.standard_error - re.standard_error) < 1e-9:
        print("FAIL: FE and RE Standard Errors are identical!")
    else:
        print("PASS: Models diverged as expected.")

if __name__ == "__main__":
    test_high_heterogeneity()
