"""
Base Generator Interface

Abstract base class for synthetic A/B test data generators.
Provides common interface for CTGAN and CopulaGAN implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger


DomainType = Literal["ecommerce", "saas", "healthcare", "marketing", "edtech"]
ModelType = Literal["ctgan", "copulagan"]
HeterogeneityLevel = Literal["low", "medium", "high"]


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    domain: DomainType
    num_experiments: int = 50
    model_type: ModelType = "ctgan"
    sample_size_range: Tuple[int, int] = (1000, 50000)
    effect_size_range: Tuple[float, float] = (0.05, 0.20)
    heterogeneity_level: HeterogeneityLevel = "medium"
    include_segments: bool = True
    seed: int = 42
    epochs: int = 300
    batch_size: int = 500
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_experiments < 1:
            raise ValueError("num_experiments must be >= 1")
        if self.sample_size_range[0] > self.sample_size_range[1]:
            raise ValueError("sample_size_range min must be <= max")
        if self.effect_size_range[0] > self.effect_size_range[1]:
            raise ValueError("effect_size_range min must be <= max")


@dataclass
class ValidationReport:
    """Quality metrics for synthetic data validation."""
    ks_statistic: float = 0.0
    ks_p_value: float = 0.0
    js_divergence: float = 0.0
    correlation_similarity: float = 0.0
    constraint_violations: List[str] = field(default_factory=list)
    column_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    passes_quality_check: bool = False
    
    # Thresholds
    KS_THRESHOLD: float = 0.1
    JS_THRESHOLD: float = 0.3
    CORR_THRESHOLD: float = 0.8
    
    def __post_init__(self):
        """Determine if quality checks pass."""
        self.passes_quality_check = (
            self.ks_statistic < self.KS_THRESHOLD and
            self.js_divergence < self.JS_THRESHOLD and
            self.correlation_similarity > self.CORR_THRESHOLD and
            len(self.constraint_violations) == 0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ks_statistic": self.ks_statistic,
            "ks_p_value": self.ks_p_value,
            "js_divergence": self.js_divergence,
            "correlation_similarity": self.correlation_similarity,
            "constraint_violations": self.constraint_violations,
            "column_stats": self.column_stats,
            "passes_quality_check": self.passes_quality_check
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "✅ PASSED" if self.passes_quality_check else "❌ FAILED"
        lines = [
            f"Quality Check: {status}",
            f"  KS Statistic: {self.ks_statistic:.4f} (threshold < {self.KS_THRESHOLD})",
            f"  JS Divergence: {self.js_divergence:.4f} (threshold < {self.JS_THRESHOLD})",
            f"  Correlation Similarity: {self.correlation_similarity:.4f} (threshold > {self.CORR_THRESHOLD})"
        ]
        if self.constraint_violations:
            lines.append(f"  Violations: {len(self.constraint_violations)}")
            for v in self.constraint_violations[:5]:
                lines.append(f"    - {v}")
        return "\n".join(lines)


class BaseGenerator(ABC):
    """
    Abstract base class for synthetic A/B test generators.
    
    Implementations:
    - CTGANGenerator: Uses Conditional Tabular GAN
    - CopulaGANGenerator: Uses Gaussian Copula for correlations
    """
    
    def __init__(self, domain: DomainType, model_dir: Optional[Path] = None):
        """
        Initialize generator.
        
        Args:
            domain: Target domain (ecommerce, saas, healthcare, marketing, edtech)
            model_dir: Directory to save/load trained models
        """
        self.domain = domain
        self.model_dir = model_dir or Path("./data/trained_models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.metadata = None
        self._is_trained = False
        self._seed_data: Optional[pd.DataFrame] = None
        
        logger.info(f"Initialized {self.__class__.__name__} for domain: {domain}")
    
    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained
    
    @abstractmethod
    def fit(self, real_data: pd.DataFrame, epochs: int = 300, verbose: bool = False) -> None:
        """
        Train the generator on real data.
        
        Args:
            real_data: DataFrame with A/B test data
            epochs: Number of training epochs
            verbose: Whether to show training progress
        """
        pass
    
    @abstractmethod
    def generate(self, config: SyntheticConfig) -> pd.DataFrame:
        """
        Generate synthetic experiments.
        
        Args:
            config: Generation configuration
            
        Returns:
            DataFrame with synthetic A/B test data
        """
        pass
    
    @abstractmethod
    def validate(self, synthetic: pd.DataFrame, real: pd.DataFrame) -> ValidationReport:
        """
        Validate quality of synthetic data.
        
        Args:
            synthetic: Generated synthetic data
            real: Original real data
            
        Returns:
            ValidationReport with quality metrics
        """
        pass
    
    @abstractmethod
    def save_model(self, path: Optional[Path] = None) -> Path:
        """
        Save trained model to disk.
        
        Args:
            path: Optional custom path
            
        Returns:
            Path where model was saved
        """
        pass
    
    @abstractmethod
    def load_model(self, path: Path) -> bool:
        """
        Load trained model from disk.
        
        Args:
            path: Path to saved model
            
        Returns:
            True if loaded successfully
        """
        pass
    
    def get_model_path(self) -> Path:
        """Get default model save path."""
        return self.model_dir / f"{self.__class__.__name__.lower()}_{self.domain}.pkl"
    
    def _apply_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply domain-specific constraints to generated data.
        
        Args:
            df: Raw generated data
            
        Returns:
            Constrained data
        """
        from src.meta_analysis.synthetic.domains.domain_constraints import DomainConstraints
        return DomainConstraints.enforce_constraints(df, self.domain)
    
    def _generate_experiment_ids(self, n: int, prefix: str = "exp") -> List[str]:
        """Generate unique experiment IDs."""
        return [f"{prefix}_{self.domain}_{i+1:04d}" for i in range(n)]
