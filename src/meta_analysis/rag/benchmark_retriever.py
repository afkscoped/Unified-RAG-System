"""
Benchmark Retriever

Retrieves industry benchmarks and historical comparisons.
"""

from typing import Dict, List, Optional, Any
from loguru import logger


# Built-in industry benchmarks by metric type and industry
INDUSTRY_BENCHMARKS = {
    "ecommerce": {
        "conversion_rate": {
            "median": 0.02,  # 2% typical conversion
            "effect_size_median": 0.05,  # 5% typical lift
            "effect_size_p25": 0.02,
            "effect_size_p75": 0.10,
            "sample_size_median": 50000,
            "common_tests": ["checkout flow", "product page", "pricing", "cart abandonment"]
        },
        "revenue_per_visitor": {
            "effect_size_median": 0.03,
            "effect_size_p25": 0.01,
            "effect_size_p75": 0.07,
            "sample_size_median": 30000
        },
        "add_to_cart": {
            "effect_size_median": 0.08,
            "effect_size_p25": 0.03,
            "effect_size_p75": 0.15,
            "sample_size_median": 25000
        }
    },
    "saas": {
        "signup_conversion": {
            "median": 0.03,
            "effect_size_median": 0.10,
            "effect_size_p25": 0.04,
            "effect_size_p75": 0.20,
            "sample_size_median": 10000,
            "common_tests": ["landing page", "pricing page", "onboarding", "trial activation"]
        },
        "trial_to_paid": {
            "effect_size_median": 0.08,
            "effect_size_p25": 0.03,
            "effect_size_p75": 0.15,
            "sample_size_median": 5000
        },
        "feature_adoption": {
            "effect_size_median": 0.15,
            "effect_size_p25": 0.05,
            "effect_size_p75": 0.25,
            "sample_size_median": 8000
        }
    },
    "mobile_games": {
        "retention_d1": {
            "median": 0.35,
            "effect_size_median": 0.02,
            "effect_size_p25": 0.01,
            "effect_size_p75": 0.05,
            "sample_size_median": 100000,
            "common_tests": ["tutorial", "first session", "push notifications"]
        },
        "retention_d7": {
            "median": 0.15,
            "effect_size_median": 0.03,
            "effect_size_p25": 0.01,
            "effect_size_p75": 0.06,
            "sample_size_median": 80000
        },
        "in_app_purchase": {
            "median": 0.02,
            "effect_size_median": 0.12,
            "effect_size_p25": 0.05,
            "effect_size_p75": 0.25,
            "sample_size_median": 150000
        }
    },
    "email_marketing": {
        "open_rate": {
            "median": 0.20,
            "effect_size_median": 0.10,
            "effect_size_p25": 0.05,
            "effect_size_p75": 0.20,
            "sample_size_median": 25000,
            "common_tests": ["subject line", "send time", "personalization"]
        },
        "click_rate": {
            "median": 0.025,
            "effect_size_median": 0.15,
            "effect_size_p25": 0.08,
            "effect_size_p75": 0.30,
            "sample_size_median": 25000
        },
        "unsubscribe_rate": {
            "median": 0.002,
            "effect_size_median": -0.10,  # Want to reduce
            "effect_size_p25": -0.20,
            "effect_size_p75": 0.05,
            "sample_size_median": 30000
        }
    }
}


class BenchmarkRetriever:
    """
    Retrieves industry benchmarks for comparison with meta-analysis results.
    
    Provides context for whether observed effects are typical,
    larger, or smaller than industry norms.
    """
    
    def __init__(self):
        """Initialize with built-in benchmarks."""
        self.benchmarks = INDUSTRY_BENCHMARKS
    
    def get_benchmark(
        self,
        industry: str,
        metric: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get benchmark for specific industry and metric.
        
        Args:
            industry: Industry name (e.g., "ecommerce", "saas")
            metric: Metric name (e.g., "conversion_rate")
            
        Returns:
            Benchmark data or None if not found
        """
        industry_data = self.benchmarks.get(industry.lower())
        if not industry_data:
            return None
        
        # Try exact match
        if metric.lower() in industry_data:
            return {
                "industry": industry.lower(),
                "metric": metric.lower(),
                **industry_data[metric.lower()]
            }
        
        # Try fuzzy match
        metric_lower = metric.lower().replace("_", " ")
        for key, data in industry_data.items():
            if metric_lower in key.replace("_", " ") or key.replace("_", " ") in metric_lower:
                return {
                    "industry": industry.lower(),
                    "metric": key,
                    **data
                }
        
        return None
    
    def compare_to_benchmark(
        self,
        effect_size: float,
        industry: str,
        metric: str
    ) -> Dict[str, Any]:
        """
        Compare an effect size to industry benchmarks.
        
        Args:
            effect_size: Observed effect size
            industry: Industry for comparison
            metric: Metric type
            
        Returns:
            Comparison results with interpretation
        """
        benchmark = self.get_benchmark(industry, metric)
        
        if not benchmark:
            return {
                "benchmark_found": False,
                "interpretation": f"No benchmark available for {industry}/{metric}"
            }
        
        median = benchmark.get("effect_size_median", 0)
        p25 = benchmark.get("effect_size_p25", median * 0.5)
        p75 = benchmark.get("effect_size_p75", median * 1.5)
        
        # Determine percentile category
        if effect_size <= p25:
            percentile = "below 25th percentile"
            category = "below_average"
            interpretation = "smaller than typical effects"
        elif effect_size <= median:
            percentile = "25th-50th percentile"
            category = "average"
            interpretation = "typical for the industry"
        elif effect_size <= p75:
            percentile = "50th-75th percentile"
            category = "above_average"
            interpretation = "larger than typical effects"
        else:
            percentile = "above 75th percentile"
            category = "exceptional"
            interpretation = "exceptionally large effects - verify validity"
        
        return {
            "benchmark_found": True,
            "observed_effect": effect_size,
            "benchmark_median": median,
            "benchmark_p25": p25,
            "benchmark_p75": p75,
            "percentile": percentile,
            "category": category,
            "interpretation": f"Effect size of {effect_size:.3f} is {interpretation} "
                             f"(benchmark median: {median:.3f})",
            "common_tests": benchmark.get("common_tests", []),
            "sample_size_recommendation": benchmark.get("sample_size_median")
        }
    
    def list_industries(self) -> List[str]:
        """List available industries."""
        return list(self.benchmarks.keys())
    
    def list_metrics(self, industry: str) -> List[str]:
        """List available metrics for an industry."""
        industry_data = self.benchmarks.get(industry.lower(), {})
        return list(industry_data.keys())
    
    def suggest_sample_size(
        self,
        industry: str,
        metric: str,
        minimum_detectable_effect: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Suggest sample size based on industry benchmarks.
        
        Args:
            industry: Industry name
            metric: Metric type
            minimum_detectable_effect: MDE (optional)
            
        Returns:
            Sample size recommendation
        """
        benchmark = self.get_benchmark(industry, metric)
        
        if not benchmark:
            return {
                "recommendation": 10000,
                "basis": "default",
                "note": "No benchmark available; using conservative default"
            }
        
        base_sample = benchmark.get("sample_size_median", 10000)
        
        # If MDE is smaller than median effect, need larger sample
        if minimum_detectable_effect:
            median_effect = benchmark.get("effect_size_median", 0.05)
            if minimum_detectable_effect < median_effect:
                # Scale up sample size
                scale_factor = (median_effect / minimum_detectable_effect) ** 2
                adjusted_sample = int(base_sample * min(scale_factor, 4))  # Cap at 4x
            else:
                adjusted_sample = base_sample
        else:
            adjusted_sample = base_sample
        
        return {
            "recommendation": adjusted_sample,
            "basis": "industry_benchmark",
            "benchmark_median": base_sample,
            "typical_effect": benchmark.get("effect_size_median"),
            "note": f"Based on {industry}/{metric} benchmarks"
        }
