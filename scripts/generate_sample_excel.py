import pandas as pd
import os

data = [
    {
        "experiment_name": "Landing Page - Hero Image",
        "control_conversions": 340,
        "control_total": 10000,
        "treatment_conversions": 320,
        "treatment_total": 9800,
        "date": "2025-01-10",
        "metric": "Signups"
    },
    {
        "experiment_name": "Landing Page - Pricing Layout",
        "control_conversions": 45,
        "control_total": 1500,
        "treatment_conversions": 60,
        "treatment_total": 1480,
        "date": "2025-01-15",
        "metric": "Purchases"
    },
    {
        "experiment_name": "Mobile Nav Redesign",
        "control_conversions": 150,
        "control_total": 4500,
        "treatment_conversions": 180,
        "treatment_total": 4600,
        "date": "2025-02-01",
        "metric": "Navigation Usage"
    }
]

df = pd.DataFrame(data)
os.makedirs("data/samples", exist_ok=True)
df.to_excel("data/samples/website_experiments.xlsx", index=False)
print("Created data/samples/website_experiments.xlsx")
