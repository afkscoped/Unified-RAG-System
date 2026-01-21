"""
Quality Metrics for Synthetic Data Validation

Implements statistical tests to validate synthetic data quality:
- Kolmogorov-Smirnov test for distribution similarity
- Jensen-Shannon divergence for probability distributions
- Correlation similarity using Frobenius norm
- Domain constraint violation checking
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from loguru import logger

from src.meta_analysis.synthetic.generators.base_generator import ValidationReport


class QualityValidator:
    """
    Validate synthetic data quality against real data.
    
    Provides statistical tests to measure:
    - Distribution similarity (KS test, JS divergence)
    - Correlation preservation
    - Domain constraint compliance
    """
    
    @staticmethod
    def kolmogorov_smirnov(real: np.ndarray, synthetic: np.ndarray) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test for distribution similarity.
        
        Args:
            real: Real data values
            synthetic: Synthetic data values
            
        Returns:
            Tuple of (statistic, p_value)
            - Statistic: 0 = identical, 1 = completely different
            - Threshold: < 0.1 is good
        """
        # Remove NaN values
        real = real[~np.isnan(real)]
        synthetic = synthetic[~np.isnan(synthetic)]
        
        if len(real) == 0 or len(synthetic) == 0:
            return 1.0, 0.0
        
        statistic, p_value = stats.ks_2samp(real, synthetic)
        return float(statistic), float(p_value)
    
    @staticmethod
    def jensen_shannon_divergence(real: np.ndarray, synthetic: np.ndarray, 
                                   bins: int = 50) -> float:
        """
        Jensen-Shannon divergence for probability distributions.
        
        Args:
            real: Real data values
            synthetic: Synthetic data values
            bins: Number of histogram bins
            
        Returns:
            Divergence value: 0 = identical, 1 = no overlap
            Threshold: < 0.3 is acceptable
        """
        # Remove NaN values
        real = real[~np.isnan(real)]
        synthetic = synthetic[~np.isnan(synthetic)]
        
        if len(real) == 0 or len(synthetic) == 0:
            return 1.0
        
        # Determine shared bin edges
        all_data = np.concatenate([real, synthetic])
        min_val, max_val = np.min(all_data), np.max(all_data)
        
        if min_val == max_val:
            return 0.0  # All values are the same
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Compute histograms
        real_hist, _ = np.histogram(real, bins=bin_edges, density=True)
        synth_hist, _ = np.histogram(synthetic, bins=bin_edges, density=True)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        real_hist = real_hist + epsilon
        synth_hist = synth_hist + epsilon
        
        # Normalize to proper probability distributions
        real_hist = real_hist / real_hist.sum()
        synth_hist = synth_hist / synth_hist.sum()
        
        return float(jensenshannon(real_hist, synth_hist))
    
    @staticmethod
    def correlation_similarity(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
        """
        Compare correlation matrices using Frobenius norm.
        
        Args:
            real_df: Real data DataFrame
            synth_df: Synthetic data DataFrame
            
        Returns:
            Similarity: 1 = identical, 0 = completely different
            Threshold: > 0.8 is good
        """
        # Get common numeric columns
        real_numeric = real_df.select_dtypes(include=[np.number])
        synth_numeric = synth_df.select_dtypes(include=[np.number])
        
        common_cols = list(set(real_numeric.columns) & set(synth_numeric.columns))
        
        if len(common_cols) < 2:
            return 1.0  # Not enough columns to compare
        
        real_corr = real_numeric[common_cols].corr().fillna(0)
        synth_corr = synth_numeric[common_cols].corr().fillna(0)
        
        # Compute Frobenius norm of difference
        diff = real_corr.values - synth_corr.values
        frobenius_distance = np.linalg.norm(diff, 'fro')
        
        # Maximum possible distance (all correlations flip sign)
        max_distance = np.sqrt(2 * len(common_cols) ** 2)
        
        if max_distance == 0:
            return 1.0
        
        similarity = 1.0 - (frobenius_distance / max_distance)
        return max(0.0, min(1.0, similarity))
    
    @staticmethod
    def check_constraints(data: pd.DataFrame, domain: str) -> List[str]:
        """
        Check domain-specific constraints.
        
        Args:
            data: DataFrame to validate
            domain: Domain name
            
        Returns:
            List of constraint violations
        """
        from src.meta_analysis.synthetic.domains.domain_constraints import DomainConstraints
        return DomainConstraints.check_violations(data, domain)
    
    @staticmethod
    def column_statistics(real: pd.Series, synthetic: pd.Series) -> Dict[str, float]:
        """
        Compute comparison statistics for a single column.
        
        Args:
            real: Real data series
            synthetic: Synthetic data series
            
        Returns:
            Dict with comparison metrics
        """
        real_clean = real.dropna()
        synth_clean = synthetic.dropna()
        
        if len(real_clean) == 0 or len(synth_clean) == 0:
            return {"error": "No valid data"}
        
        return {
            "real_mean": float(real_clean.mean()) if np.issubdtype(real_clean.dtype, np.number) else None,
            "synth_mean": float(synth_clean.mean()) if np.issubdtype(synth_clean.dtype, np.number) else None,
            "real_std": float(real_clean.std()) if np.issubdtype(real_clean.dtype, np.number) else None,
            "synth_std": float(synth_clean.std()) if np.issubdtype(synth_clean.dtype, np.number) else None,
            "real_min": float(real_clean.min()) if np.issubdtype(real_clean.dtype, np.number) else None,
            "synth_min": float(synth_clean.min()) if np.issubdtype(synth_clean.dtype, np.number) else None,
            "real_max": float(real_clean.max()) if np.issubdtype(real_clean.dtype, np.number) else None,
            "synth_max": float(synth_clean.max()) if np.issubdtype(synth_clean.dtype, np.number) else None,
        }
    
    @classmethod
    def full_validation(cls, synthetic: pd.DataFrame, real: pd.DataFrame, 
                        domain: str) -> ValidationReport:
        """
        Run full validation suite on synthetic data.
        
        Args:
            synthetic: Generated synthetic data
            real: Original real data
            domain: Domain name for constraint checking
            
        Returns:
            ValidationReport with all metrics
        """
        logger.info(f"Running full validation on {len(synthetic)} synthetic rows")
        
        # Get common numeric columns
        real_numeric = real.select_dtypes(include=[np.number])
        synth_numeric = synthetic.select_dtypes(include=[np.number])
        common_cols = list(set(real_numeric.columns) & set(synth_numeric.columns))
        
        # Compute KS statistic (average across columns)
        ks_stats = []
        ks_pvals = []
        column_stats = {}
        
        for col in common_cols:
            ks_stat, ks_pval = cls.kolmogorov_smirnov(
                real_numeric[col].values, 
                synth_numeric[col].values
            )
            ks_stats.append(ks_stat)
            ks_pvals.append(ks_pval)
            
            # Column-level statistics
            if col in real.columns and col in synthetic.columns:
                column_stats[col] = cls.column_statistics(real[col], synthetic[col])
        
        avg_ks = np.mean(ks_stats) if ks_stats else 1.0
        avg_ks_pval = np.mean(ks_pvals) if ks_pvals else 0.0
        
        # Compute JS divergence (average across columns)
        js_divs = []
        for col in common_cols:
            js_div = cls.jensen_shannon_divergence(
                real_numeric[col].values,
                synth_numeric[col].values
            )
            js_divs.append(js_div)
        
        avg_js = np.mean(js_divs) if js_divs else 1.0
        
        # Compute correlation similarity
        corr_sim = cls.correlation_similarity(real, synthetic)
        
        # Check constraints
        violations = cls.check_constraints(synthetic, domain)
        
        # Create report
        report = ValidationReport(
            ks_statistic=float(avg_ks),
            ks_p_value=float(avg_ks_pval),
            js_divergence=float(avg_js),
            correlation_similarity=float(corr_sim),
            constraint_violations=violations,
            column_stats=column_stats
        )
        
        logger.info(f"Validation complete: {report.summary()}")
        return report


def validate_synthetic_quality(synthetic: pd.DataFrame, real: pd.DataFrame,
                                domain: str) -> ValidationReport:
    """
    Convenience function for full validation.
    
    Args:
        synthetic: Generated synthetic data
        real: Original real data
        domain: Domain name
        
    Returns:
        ValidationReport with quality metrics
    """
    return QualityValidator.full_validation(synthetic, real, domain)
