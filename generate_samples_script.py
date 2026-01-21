"""
Generate sample A/B test files for user testing.
"""
import os
import pandas as pd
from src.meta_analysis.utils.synthetic_generator import SyntheticABTestGenerator

def generate_samples():
    output_dir = "sample_data"
    os.makedirs(output_dir, exist_ok=True)
    
    generator = SyntheticABTestGenerator(random_seed=123)
    
    # 1. Marketing Campaigns (CSV)
    print("Generating marketing campaigns...")
    marketing_df = generator.generate(n_experiments=12, domain="marketing", method="random")
    marketing_path = os.path.join(output_dir, "marketing_campaigns_v2.csv")
    marketing_df.to_csv(marketing_path, index=False)
    print(f"Saved {marketing_path}")
    
    # 2. Product Features (Excel)
    print("Generating product features...")
    product_df = generator.generate(n_experiments=15, domain="product", method="random")
    product_path = os.path.join(output_dir, "new_feature_launches.xlsx")
    product_df.to_excel(product_path, index=False)
    print(f"Saved {product_path}")
    
    # 3. Email Subject Lines (CSV) - High Volume
    print("Generating email experiments...")
    email_df = generator.generate(n_experiments=20, domain="email", method="random")
    email_path = os.path.join(output_dir, "email_subject_tests.csv")
    email_df.to_csv(email_path, index=False)
    print(f"Saved {email_path}")
    
    # 4. UX Tests (CSV) - Mixed results
    print("Generating UX tests...")
    ux_df = generator.generate(n_experiments=8, domain="ux", method="random")
    ux_path = os.path.join(output_dir, "ux_redesign_tests.csv")
    ux_df.to_csv(ux_path, index=False)
    print(f"Saved {ux_path}")

if __name__ == "__main__":
    generate_samples()
