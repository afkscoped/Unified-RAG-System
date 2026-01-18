"""
FastAPI REST API

Provides REST endpoints for the RAG system:
- POST /query - Query the system
- POST /ingest - Upload documents
- GET /metrics - System metrics
- POST /feedback - Submit feedback

Story Mode endpoints (under /story):
- POST /story/generate - Generate story segment
- POST /story/compare - Compare all approaches
- GET /story/characters - List characters
- GET /story/arc/{id} - Character arc
- GET /story/suggestions - Plot suggestions
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from src.core.rag_system import UnifiedRAGSystem, RAGConfig
from src.api.story_api import story_router, set_comparison_engine


# Global instances
rag_system: Optional[UnifiedRAGSystem] = None
comparison_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG system and Story Engine on startup."""
    global rag_system, comparison_engine
    
    config_path = os.getenv("RAG_CONFIG_PATH", "config/config.yaml")
    
    try:
        if os.path.exists(config_path):
            rag_system = UnifiedRAGSystem(config_path)
        else:
            rag_system = UnifiedRAGSystem()
        logger.info("RAG system initialized")
        
        # Initialize Story Comparison Engine
        try:
            from src.story.comparison_engine import StoryComparisonEngine
            from src.llm.llm_router import LLMRouter
            
            llm_router = LLMRouter()
            comparison_engine = StoryComparisonEngine(
                unified_rag_system=rag_system,
                llm_router=llm_router,
                embedding_manager=rag_system.embedding_manager if hasattr(rag_system, 'embedding_manager') else None
            )
            set_comparison_engine(comparison_engine)
            logger.info("Story Comparison Engine initialized")
        except Exception as e:
            logger.warning(f"Story engine initialization failed (optional): {e}")
            
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        rag_system = UnifiedRAGSystem()
    
    yield
    
    # Cleanup
    logger.info("Shutting down RAG system")


app = FastAPI(
    title="StoryWeaver Platform API",
    description="Dual-mode RAG system: Document Q&A + Story Generation with Graph RAG comparison",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Story Mode router
app.include_router(story_router, prefix="/story", tags=["Story Mode"])


# ─────────────────────────────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Query request model."""
    question: str = Field(..., description="The question to ask")
    top_k: int = Field(5, ge=1, le=20, description="Number of sources to retrieve")
    use_cache: bool = Field(True, description="Whether to use semantic cache")


class SourceInfo(BaseModel):
    """Source document info."""
    content: str
    score: float
    metadata: dict


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    sources: List[SourceInfo]
    cached: bool
    query_type: str
    semantic_weight: float
    lexical_weight: float


class FeedbackRequest(BaseModel):
    """Feedback request model."""
    question: str = Field(..., description="The original question")
    rating: int = Field(..., ge=1, le=5, description="Rating 1-5")


class IngestResponse(BaseModel):
    """Ingest response model."""
    status: str
    files_processed: int
    chunks_created: int
    message: str


class MetricsResponse(BaseModel):
    """Metrics response model."""
    queries: int
    cache_hits: int
    cache_hit_rate: float
    documents_ingested: int
    chunks_created: int


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Advanced RAG System",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    global rag_system
    
    return {
        "status": "healthy",
        "rag_initialized": rag_system is not None,
        "indexed": rag_system._indexed if rag_system else False,
        "document_count": len(rag_system.documents) if rag_system else 0
    }


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Query the RAG system.
    
    Returns an answer based on the indexed documents.
    """
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not rag_system._indexed:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload documents first.")
    
    try:
        result = await rag_system.aquery(
            question=request.question,
            top_k=request.top_k,
            use_cache=request.use_cache
        )
        
        return QueryResponse(
            answer=result.answer,
            sources=[
                SourceInfo(
                    content=src.content[:500],  # Truncate for response
                    score=src.score,
                    metadata=src.metadata
                )
                for src in result.sources
            ],
            cached=result.cached,
            query_type=result.query_type,
            semantic_weight=result.semantic_weight,
            lexical_weight=result.lexical_weight
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_documents(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and ingest documents.
    
    Supports PDF, TXT, and DOCX files.
    """
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    allowed_extensions = {".pdf", ".txt", ".docx", ".doc"}
    total_chunks = 0
    files_processed = 0
    
    # Create temp directory for uploads
    temp_dir = tempfile.mkdtemp()
    
    try:
        for file in files:
            ext = Path(file.filename).suffix.lower()
            
            if ext not in allowed_extensions:
                logger.warning(f"Skipping unsupported file: {file.filename}")
                continue
            
            # Save to temp file
            temp_path = os.path.join(temp_dir, file.filename)
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Ingest
            chunks = rag_system.ingest_file(temp_path)
            total_chunks += chunks
            files_processed += 1
        
        # Build index if we have documents
        if rag_system.documents:
            rag_system.build_index()
        
        return IngestResponse(
            status="success",
            files_processed=files_processed,
            chunks_created=total_chunks,
            message=f"Successfully ingested {files_processed} files into {total_chunks} chunks"
        )
        
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup temp files
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a query to improve search weights."""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    rag_system.submit_feedback(request.question, request.rating)
    
    return {"status": "success", "message": "Feedback recorded"}


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """Get system metrics."""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    metrics = rag_system.get_metrics()
    
    return MetricsResponse(
        queries=metrics.get("queries", 0),
        cache_hits=metrics.get("cache_hits", 0),
        cache_hit_rate=metrics.get("cache_hit_rate", 0),
        documents_ingested=metrics.get("documents_ingested", 0),
        chunks_created=metrics.get("chunks_created", 0)
    )


@app.get("/metrics/detailed", tags=["Monitoring"])
async def get_detailed_metrics():
    """Get detailed system metrics including memory and LLM stats."""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return rag_system.get_metrics()


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

