"""
Base Experiment MCP Server

Provides core data models and interfaces for experiment data ingestion:
- StandardizedStudy: Harmonized experiment format
- ExperimentMCPBase: Abstract base class for platform connectors
- ValidationReport: Data quality checks
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class EffectSizeType(Enum):
    """Types of effect sizes supported."""
    LOG_ODDS_RATIO = "log_odds_ratio"
    COHENS_D = "cohens_d"
    RISK_RATIO = "risk_ratio"
    RISK_DIFFERENCE = "risk_difference"
    MEAN_DIFFERENCE = "mean_difference"
    CORRELATION = "correlation"


class MetricType(Enum):
    """Types of metrics in A/B tests."""
    CONVERSION = "conversion"
    REVENUE = "revenue"
    ENGAGEMENT = "engagement"
    RETENTION = "retention"
    CLICK_THROUGH = "click_through"
    CUSTOM = "custom"


@dataclass
class StandardizedStudy:
    """
    Harmonized experiment format for meta-analysis.
    
    All experiments from different sources are converted to this format
    before meta-analysis is performed.
    
    Attributes:
        study_id: Unique identifier for the study
        study_name: Human-readable name
        effect_size: Standardized effect size (typically log odds ratio)
        standard_error: Standard error of the effect size
        sample_size_control: Number of observations in control group
        sample_size_treatment: Number of observations in treatment group
        metric_name: Name of the metric being measured
        metric_type: Type of metric (conversion, revenue, etc.)
        effect_size_type: Type of effect size measurement
        platform: Source platform (CSV, Optimizely, etc.)
        timestamp: When the experiment was conducted
        confidence_interval_lower: Lower bound of CI
        confidence_interval_upper: Upper bound of CI
        p_value: Statistical significance
        segments: Optional segment-level results
        metadata: Additional metadata
    """
    study_id: str
    study_name: str
    effect_size: float
    standard_error: float
    sample_size_control: int
    sample_size_treatment: int
    metric_name: str
    metric_type: MetricType = MetricType.CONVERSION
    effect_size_type: EffectSizeType = EffectSizeType.LOG_ODDS_RATIO
    platform: str = "csv"
    timestamp: Optional[datetime] = None
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    p_value: Optional[float] = None
    segments: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Calculate CI if not provided."""
        if self.confidence_interval_lower is None:
            self.confidence_interval_lower = self.effect_size - 1.96 * self.standard_error
        if self.confidence_interval_upper is None:
            self.confidence_interval_upper = self.effect_size + 1.96 * self.standard_error
    
    @property
    def total_sample_size(self) -> int:
        """Total sample size across both groups."""
        return self.sample_size_control + self.sample_size_treatment
    
    @property
    def variance(self) -> float:
        """Variance of the effect size."""
        return self.standard_error ** 2
    
    @property
    def weight(self) -> float:
        """Inverse variance weight for meta-analysis."""
        return 1.0 / self.variance if self.variance > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "study_id": self.study_id,
            "study_name": self.study_name,
            "effect_size": self.effect_size,
            "standard_error": self.standard_error,
            "sample_size_control": self.sample_size_control,
            "sample_size_treatment": self.sample_size_treatment,
            "metric_name": self.metric_name,
            "metric_type": self.metric_type.value,
            "effect_size_type": self.effect_size_type.value,
            "platform": self.platform,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "confidence_interval_lower": self.confidence_interval_lower,
            "confidence_interval_upper": self.confidence_interval_upper,
            "p_value": self.p_value,
            "segments": self.segments,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardizedStudy":
        """Create from dictionary representation."""
        return cls(
            study_id=data["study_id"],
            study_name=data["study_name"],
            effect_size=data["effect_size"],
            standard_error=data["standard_error"],
            sample_size_control=data["sample_size_control"],
            sample_size_treatment=data["sample_size_treatment"],
            metric_name=data["metric_name"],
            metric_type=MetricType(data.get("metric_type", "conversion")),
            effect_size_type=EffectSizeType(data.get("effect_size_type", "log_odds_ratio")),
            platform=data.get("platform", "csv"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            confidence_interval_lower=data.get("confidence_interval_lower"),
            confidence_interval_upper=data.get("confidence_interval_upper"),
            p_value=data.get("p_value"),
            segments=data.get("segments"),
            metadata=data.get("metadata")
        )


@dataclass
class ValidationIssue:
    """A single validation issue found in the data."""
    field: str
    issue_type: str  # "error", "warning", "info"
    message: str
    value: Optional[Any] = None


@dataclass
class ValidationReport:
    """
    Report on data quality and validation results.
    
    Attributes:
        is_valid: Whether the data passes all critical validations
        total_records: Total number of records processed
        valid_records: Number of records that passed validation
        issues: List of validation issues found
        warnings: List of warnings (non-critical issues)
        summary: Summary statistics of the validation
    """
    is_valid: bool
    total_records: int
    valid_records: int
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def error_count(self) -> int:
        """Number of errors found."""
        return len([i for i in self.issues if i.issue_type == "error"])
    
    @property
    def warning_count(self) -> int:
        """Number of warnings found."""
        return len(self.warnings)
    
    def add_issue(self, field: str, issue_type: str, message: str, value: Any = None):
        """Add a validation issue."""
        issue = ValidationIssue(field, issue_type, message, value)
        if issue_type == "warning":
            self.warnings.append(issue)
        else:
            self.issues.append(issue)
            if issue_type == "error":
                self.is_valid = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [
                {"field": i.field, "type": i.issue_type, "message": i.message, "value": i.value}
                for i in self.issues
            ],
            "warnings": [
                {"field": w.field, "type": w.issue_type, "message": w.message, "value": w.value}
                for w in self.warnings
            ],
            "summary": self.summary
        }


@dataclass
class ExperimentFilter:
    """Filters for querying experiments."""
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    metric_types: Optional[List[MetricType]] = None
    platforms: Optional[List[str]] = None
    min_sample_size: Optional[int] = None
    search_query: Optional[str] = None


class ExperimentMCPBase(ABC):
    """
    Abstract base class for experiment data MCP servers.
    
    All platform-specific connectors (CSV, Optimizely, LaunchDarkly, etc.)
    should inherit from this class and implement its abstract methods.
    """
    
    @abstractmethod
    def list_experiments(
        self, 
        filters: Optional[ExperimentFilter] = None
    ) -> List[Dict[str, Any]]:
        """
        List available experiments.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            List of experiment metadata dictionaries
        """
        pass
    
    @abstractmethod
    def get_experiment_details(
        self, 
        experiment_id: str
    ) -> Optional[StandardizedStudy]:
        """
        Get detailed results for a specific experiment.
        
        Args:
            experiment_id: Unique identifier of the experiment
            
        Returns:
            Standardized study data or None if not found
        """
        pass
    
    @abstractmethod
    def get_experiments(
        self, 
        experiment_ids: Optional[List[str]] = None,
        filters: Optional[ExperimentFilter] = None
    ) -> List[StandardizedStudy]:
        """
        Get multiple experiments.
        
        Args:
            experiment_ids: Specific IDs to fetch (optional)
            filters: Filters to apply (optional)
            
        Returns:
            List of standardized studies
        """
        pass
    
    @abstractmethod
    def validate_data(
        self, 
        experiments: List[StandardizedStudy]
    ) -> ValidationReport:
        """
        Validate experiment data quality.
        
        Args:
            experiments: List of studies to validate
            
        Returns:
            Validation report with issues and warnings
        """
        pass
    
    def get_segmented_results(
        self, 
        experiment_id: str, 
        segment_by: str
    ) -> Dict[str, StandardizedStudy]:
        """
        Get results segmented by a specific dimension.
        
        Args:
            experiment_id: ID of the experiment
            segment_by: Dimension to segment by (e.g., "device", "country")
            
        Returns:
            Dictionary mapping segment values to study results
        """
        # Default implementation - can be overridden by subclasses
        experiment = self.get_experiment_details(experiment_id)
        if experiment and experiment.segments and segment_by in experiment.segments:
            return experiment.segments[segment_by]
        return {}
