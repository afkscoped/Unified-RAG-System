"""
Meta-Analysis API Router

FastAPI endpoints for the meta-analysis feature.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Request
from typing import List, Optional
import json
import uuid
from datetime import datetime

from src.meta_analysis.api.schemas import (
    AnalysisRequest,
    MetaAnalysisResponse,
    ValidationRequest
)
from src.meta_analysis.orchestration.meta_analysis_agent import MetaAnalysisAgent
from src.meta_analysis.mcp_servers.csv_experiment_server import CSVExperimentMCP
from src.meta_analysis.integration.rag_bridge import MetaAnalysisRAGBridge
from src.meta_analysis.integration.llm_bridge import MetaAnalysisLLMBridge

# Initialize router
router = APIRouter(prefix="/meta-analysis", tags=["Meta Analysis"])

# Dependency injection for Agent
def get_analysis_agent(request: Request = None):
    # In a real app, these would be initialized at startup and reused
    # For now, we instantiate them per request or use singletons
    
    llm_router = getattr(request.app.state, "llm_router", None) if request else None
    
    csv_mcp = CSVExperimentMCP(data_dir="./data/experiments")
    rag_bridge = MetaAnalysisRAGBridge()  # Will connect to main RAG if initialized
    llm_bridge = MetaAnalysisLLMBridge(llm_router=llm_router)  # Will connect to main LLM Router
    
    # Initialize bridges (safe to call multiple times)
    rag_bridge.initialize()
    llm_bridge.initialize()
    
    return MetaAnalysisAgent(
        csv_mcp=csv_mcp,
        rag_bridge=rag_bridge,
        llm_bridge=llm_bridge
    )


@router.post("/analyze", response_model=MetaAnalysisResponse)
async def run_analysis(
    request: AnalysisRequest,
    agent: MetaAnalysisAgent = Depends(get_analysis_agent)
):
    """
    Run a full meta-analysis based on the request.
    """
    try:
        # Run agent
        state = agent.run(
            query=request.query,
            config={
                "model_preference": request.model_preference
            }
        )
        
        # Map state to response schema
        response = MetaAnalysisResponse(
            analysis_id=str(uuid.uuid4()),
            status=state.current_stage.value,
            timestamp=datetime.now(),
            warnings=state.warnings,
            error=state.errors[0] if state.errors else None
        )
        
        # Populate results if successful
        if state.meta_result:
            response.pooled_effect = {
                "effect": state.meta_result.pooled_effect,
                "ci_lower": state.meta_result.confidence_interval[0],
                "ci_upper": state.meta_result.confidence_interval[1],
                "p_value": state.meta_result.p_value,
                "model_type": state.meta_result.model_type
            }
            
            response.heterogeneity = {
                "i2": state.heterogeneity_stats.get("I2", 0),
                "tau2": state.heterogeneity_stats.get("tau2", 0),
                "q": state.heterogeneity_stats.get("Q", 0),
                "q_p_value": state.heterogeneity_stats.get("Q_pvalue", 1)
            }
            
            response.studies = [
                {
                    "id": s.study_id,
                    "name": s.study_name,
                    "effect": s.effect_size,
                    "se": s.standard_error,
                    "sample_size": s.total_sample_size,
                    "weight": state.meta_result.study_weights.get(s.study_id),
                    "year": s.timestamp.year if s.timestamp else None
                }
                for s in state.standardized_studies
            ]
        
        # Advanced results
        if state.publication_bias_result:
            pbr = state.publication_bias_result
            response.publication_bias = {
                "detected": pbr.get("bias_detected", False),
                "severity": pbr.get("bias_severity", "unknown"),
                "eggers_p": pbr.get("eggers_test", {}).get("p_value", 1),
                "trim_fill_k0": pbr.get("trim_and_fill", {}).get("k0", 0),
                "interpretation": pbr.get("interpretation", "")
            }
            
        if state.sensitivity_result:
            sr = state.sensitivity_result
            response.sensitivity = {
                "robust": sr.get("robust", True),
                "influential_studies": sr.get("influential_studies", []),
                "interpretation": sr.get("interpretation", "")
            }
        
        # Insights
        response.interpretation = state.rag_context
        response.recommendations = state.recommendations
        response.report_markdown = state.final_report
        
        # Visualizations
        if request.include_visualizations:
            response.visualizations = state.visualizations
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_experiments(
    file: UploadFile = File(...),
    agent: MetaAnalysisAgent = Depends(get_analysis_agent)
):
    """
    Upload a CSV/Excel file containing experiment data.
    """
    try:
        # Save temp file
        import tempfile
        import os
        import shutil
        
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        try:
            # Process with MCP
            result = agent.csv_mcp.upload_file(tmp_path)
            # Clean up handled by CSVExperimentMCP if it moves file, otherwise delete
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
            return {"message": "File uploaded successfully", "details": result}
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise e
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/recommendations")
async def get_recommendations(
    agent: MetaAnalysisAgent = Depends(get_analysis_agent)
):
    """
    Get general recommendations and guidance for meta-analysis.
    """
    # This endpoint provides static or general guidance if no analysis context
    return {
        "guidelines": [
            {
                "topic": "Heterogeneity",
                "advice": "If I² > 50%, consider using random effects models and investigating subgroups."
            },
            {
                "topic": "Publication Bias",
                "advice": "Always check funnel plots if you have 10+ studies. Formal tests have low power with fewer studies."
            },
            {
                "topic": "Search Strategy",
                "advice": "Ensure your search for experiments is comprehensive to avoid selection bias."
            }
        ],
        "benchmarks": agent.rag_bridge.rag_system.get_benchmarks() if agent.rag_bridge and hasattr(agent.rag_bridge.rag_system, "get_benchmarks") else "Benchmarks available via RAG analysis"
    }
