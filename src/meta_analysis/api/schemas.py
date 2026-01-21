"""
API Schemas

Pydantic models for Meta-Analysis API.
"""

from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime


# Shared Models

class ExperimentFilterSchema(BaseModel):
    search_query: Optional[str] = None
    metric_type: Optional[str] = None
    min_sample_size: Optional[int] = None
    year_from: Optional[int] = None
    platform: Optional[str] = None


class EffectSizeSchema(BaseModel):
    effect: float
    ci_lower: float
    ci_upper: float
    p_value: float
    model_type: str


class HeterogeneitySchema(BaseModel):
    i2: float
    tau2: float
    q: float
    q_p_value: float


class StudySchema(BaseModel):
    id: str
    name: str
    effect: float
    se: float
    weight: Optional[float] = None
    sample_size: int
    year: Optional[int] = None


# Request Models

class AnalysisRequest(BaseModel):
    query: str = Field(..., description="Natural language query describing the analysis request")
    filters: Optional[ExperimentFilterSchema] = None
    model_preference: str = Field("auto", description="Model preference: 'fixed', 'random', or 'auto'")
    include_visualizations: bool = True
    include_rag_insights: bool = True


class ValidationRequest(BaseModel):
    studies: List[Dict[str, Any]]
    metric_type: str = "conversion_rate"


# Response Models

class BiasResultSchema(BaseModel):
    detected: bool
    severity: str
    eggers_p: float
    trim_fill_k0: int
    interpretation: str


class SensitivityResultSchema(BaseModel):
    robust: bool
    influential_studies: List[str]
    interpretation: str


class RecommendationSchema(BaseModel):
    title: str
    priority: str
    message: str
    rationale: str
    action: str


class MetaAnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    timestamp: datetime
    
    # Results
    pooled_effect: Optional[EffectSizeSchema] = None
    heterogeneity: Optional[HeterogeneitySchema] = None
    studies: List[StudySchema] = []
    
    # Advanced Analysis
    publication_bias: Optional[BiasResultSchema] = None
    sensitivity: Optional[SensitivityResultSchema] = None
    
    # Insights
    interpretation: Optional[str] = None
    recommendations: List[RecommendationSchema] = []
    
    # Visualizations (JSON strings for Plotly)
    visualizations: Dict[str, str] = {}
    
    # Report
    report_markdown: Optional[str] = None
    
    warnings: List[str] = []
    error: Optional[str] = None
