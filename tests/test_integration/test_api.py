"""
Integration Tests for FastAPI

Tests the REST API endpoints.
"""

import pytest
from fastapi.testclient import TestClient


class TestFastAPIApp:
    """Tests for FastAPI application."""
    
    def test_import(self):
        """Test that app can be imported."""
        from src.api.fastapi_app import app
        assert app is not None
    
    def test_health_check(self):
        """Test health check endpoint."""
        from src.api.fastapi_app import app
        
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_detailed_health(self):
        """Test detailed health endpoint."""
        from src.api.fastapi_app import app
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "rag_initialized" in data
    
    def test_query_without_index(self):
        """Test query without indexed documents returns error."""
        from src.api.fastapi_app import app
        
        client = TestClient(app)
        response = client.post(
            "/query",
            json={"question": "What is Python?", "top_k": 3}
        )
        
        # Should return 400 because no documents are indexed
        assert response.status_code in [400, 503]
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        from src.api.fastapi_app import app
        
        client = TestClient(app)
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "queries" in data
        assert "cache_hits" in data


class TestAPIModels:
    """Tests for API request/response models."""
    
    def test_query_request(self):
        """Test QueryRequest model."""
        from src.api.fastapi_app import QueryRequest
        
        request = QueryRequest(question="What is AI?", top_k=5, use_cache=True)
        
        assert request.question == "What is AI?"
        assert request.top_k == 5
        assert request.use_cache is True
    
    def test_feedback_request(self):
        """Test FeedbackRequest model."""
        from src.api.fastapi_app import FeedbackRequest
        
        request = FeedbackRequest(question="Test query", rating=4)
        
        assert request.question == "Test query"
        assert request.rating == 4

