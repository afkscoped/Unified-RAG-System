import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
# Use absolute path relative to script execution location or define explicitly
BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "src", "meta_analysis", "sample_datasets")
CATALOG_FILE = "dataset_catalog.json"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper to generate dataset
def generate_dataset(domain, name, num_experiments=20, effect_size=0.02, base_rate=0.1, segments=None):
    data = []
    
    segment_vals = segments if segments else ["All"]
    
    for i in range(num_experiments):
        for segment in segment_vals:
            exp_name = f"{name}_Exp_{i+1}"
            if segment != "All":
                exp_name += f"_{segment}"
            
            # Random volumes
            c_visitors = np.random.randint(1000, 5000)
            v_visitors = np.random.randint(1000, 5000)
            
            # Conversions
            c_conv = np.random.binomial(c_visitors, base_rate)
            
            # Variant effect
            lift = np.random.normal(effect_size, 0.01) # Some variance
            v_rate = base_rate * (1 + lift)
            v_conv = np.random.binomial(v_visitors, v_rate)
            
            row = {
                "experiment_name": exp_name,
                "control_visitors": c_visitors,
                "control_conversions": c_conv,
                "variant_visitors": v_visitors,
                "variant_conversions": v_conv
            }
            
            if segments:
                row["segment"] = segment
                
            data.append(row)
            
    return pd.DataFrame(data)

# Define dataset specs
datasets_specs = [
    {
        "id": "ecom_checkout",
        "name": "E-commerce Checkout Optimization",
        "domain": "E-commerce",
        "description": "Experiments testing single-page vs multi-step checkout flows.",
        "params": {"effect_size": 0.05, "base_rate": 0.15},
        "filename": "ecom_checkout.csv"
    },
    {
        "id": "saas_pricing",
        "name": "SaaS Pricing Page Tests",
        "domain": "SaaS / Product",
        "description": "Impact of annual vs monthly toggle placement on signups.",
        "params": {"effect_size": 0.03, "base_rate": 0.02},
        "filename": "saas_pricing.csv"
    },
    {
        "id": "email_campaigns",
        "name": "Email Subject Line CTR",
        "domain": "Email campaigns",
        "description": "Open rates for personalized vs generic subject lines.",
        "params": {"effect_size": 0.10, "base_rate": 0.20},
        "filename": "email_ctr.csv"
    },
    {
        "id": "mobile_ux_onboarding",
        "name": "Mobile App Onboarding Flow",
        "domain": "Mobile UX",
        "description": "Completion rates for tutorial vs skip-to-action onboarding.",
        "params": {"effect_size": -0.01, "base_rate": 0.60},
        "filename": "mobile_onboarding.csv"
    },
    {
        "id": "healthcare_cta",
        "name": "Healthcare Appointment Booking",
        "domain": "Healthcare",
        "description": "Booking rates for 'Book Now' vs 'Schedule Consultation'.",
        "params": {"effect_size": 0.015, "base_rate": 0.05},
        "filename": "healthcare_booking.csv"
    },
    {
        "id": "edtech_course_enroll",
        "name": "EdTech Course Enrollment",
        "domain": "EdTech",
        "description": "Enrollment rates with video trailer vs text description.",
        "params": {"effect_size": 0.04, "base_rate": 0.08},
        "filename": "edtech_enrollment.csv"
    },
    {
        "id": "marketing_landing_page",
        "name": "Landing Page Hero Image",
        "domain": "Marketing",
        "description": "Conversion rates for product shot vs lifestyle image.",
        "params": {"effect_size": 0.02, "base_rate": 0.12},
        "filename": "marketing_hero.csv"
    },
    {
        "id": "product_feature_adoption",
        "name": "New Feature Discovery",
        "domain": "SaaS / Product",
        "description": "Adoption rates with tooltip vs modal announcement.",
        "params": {"effect_size": 0.06, "base_rate": 0.30},
        "filename": "feature_adoption.csv"
    },
    {
        "id": "cart_abandonment",
        "name": "Cart Abandonment Recovery",
        "domain": "E-commerce",
        "description": "Recovery rates for discount code vs urgency messaging.",
        "params": {"effect_size": 0.025, "base_rate": 0.09},
        "filename": "cart_recovery.csv"
    },
    {
        "id": "subscription_upsell",
        "name": "Premium Upsell Modal",
        "domain": "Mobile UX",
        "description": "Upsell conversion on free tier usage limits.",
        "params": {"effect_size": 0.01, "base_rate": 0.03},
        "filename": "premium_upsell.csv"
    },
    {
        "id": "geo_segmented",
        "name": "Global vs Local Messaging",
        "domain": "Marketing",
        "description": "Campaign performance across different regions.",
        "params": {"effect_size": 0.0, "base_rate": 0.10, "segments": ["NA", "EU", "APAC"]},
        "filename": "geo_messaging.csv"
    }
]

catalog = []

print(f"Generating datasets in {OUTPUT_DIR}...")

for spec in datasets_specs:
    df = generate_dataset(
        domain=spec["domain"],
        name=spec["name"],
        **spec.get("params", {})
    )
    
    file_path = os.path.join(OUTPUT_DIR, spec["filename"])
    df.to_csv(file_path, index=False)
    
    catalog.append({
        "dataset_id": spec["id"],
        "name": spec["name"],
        "domain": spec["domain"],
        "description": spec["description"],
        "file_path": spec["filename"], # Relative to sample_datasets folder
        "row_count": len(df)
    })
    print(f"Generated {spec['filename']} ({len(df)} rows)")

# Save catalog
catalog_path = os.path.join(OUTPUT_DIR, CATALOG_FILE)
with open(catalog_path, "w") as f:
    json.dump(catalog, f, indent=2)

print(f"Catalog saved to {catalog_path}")
