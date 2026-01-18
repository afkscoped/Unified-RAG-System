"""Pytest configuration and fixtures."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_documents(temp_dir):
    """Create sample documents for testing."""
    docs = []
    
    # Create text files
    for i in range(3):
        path = os.path.join(temp_dir, f"doc_{i}.txt")
        content = f"""
        Document {i}: Test Content
        
        This is a sample document for testing the RAG system.
        It contains information about topic {i}.
        
        Key points:
        - Point A for document {i}
        - Point B for document {i}
        - Point C for document {i}
        
        The RAG system should be able to find and retrieve this content
        when asked about topic {i} or related concepts.
        """
        
        with open(path, 'w') as f:
            f.write(content)
        
        docs.append(path)
    
    return docs


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample config file."""
    config_path = os.path.join(temp_dir, "config.yaml")
    
    config_content = """
model:
  embedding: "BAAI/bge-small-en-v1.5"
  llm_provider: "groq"
  llm_model: "llama-3.1-8b-instant"
  device: "cpu"
  use_fp16: false

search:
  chunk_size: 500
  chunk_overlap: 50
  top_k: 3

cache:
  enabled: true
  similarity_threshold: 0.95
  max_items: 10

memory:
  max_ram_gb: 8.0
  max_vram_gb: 4.0
  embedding_batch_size: 8
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path


@pytest.fixture
def mock_groq_key(monkeypatch):
    """Set a mock GROQ API key."""
    monkeypatch.setenv("GROQ_API_KEY", "test_key_not_real")

