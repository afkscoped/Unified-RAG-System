import pytest
import os
import json
import pandas as pd
from unittest.mock import MagicMock
from src.meta_analysis.mcp_servers.csv_experiment_server import CSVExperimentMCP

# Use absolute path resolving logic
BASE_DIR = os.getcwd()
SAMPLE_DIR = os.path.join(BASE_DIR, "src", "meta_analysis", "sample_datasets")
CATALOG_PATH = os.path.join(SAMPLE_DIR, "dataset_catalog.json")

def test_catalog_exists():
    assert os.path.exists(CATALOG_PATH), "Dataset catalog.json not found"

def test_catalog_valid_json():
    with open(CATALOG_PATH, 'r') as f:
        catalog = json.load(f)
    assert isinstance(catalog, list), "Catalog should be a list"
    assert len(catalog) >= 10, "Should have at least 10 datasets"

def test_sample_files_exist():
    with open(CATALOG_PATH, 'r') as f:
        catalog = json.load(f)
    
    for entry in catalog:
        file_path = os.path.join(SAMPLE_DIR, entry['file_path'])
        assert os.path.exists(file_path), f"File {entry['file_path']} missing"

def test_sample_schema_valid():
    with open(CATALOG_PATH, 'r') as f:
        catalog = json.load(f)
    
    required_cols = {'experiment_name', 'control_visitors', 'control_conversions', 'variant_visitors', 'variant_conversions'}
    
    for entry in catalog:
        file_path = os.path.join(SAMPLE_DIR, entry['file_path'])
        df = pd.read_csv(file_path)
        
        # Check columns
        assert required_cols.issubset(df.columns), f"Missing columns in {entry['name']}"
        
        # Check data types (basic)
        assert pd.api.types.is_numeric_dtype(df['control_visitors']), f"control_visitors not numeric in {entry['name']}"
        assert pd.api.types.is_numeric_dtype(df['control_conversions']), f"control_conversions not numeric in {entry['name']}"

def test_mcp_ingestion_simulation():
    """Simulate MCP ingestion for a sample dataset."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        mcp = CSVExperimentMCP(data_dir=tmpdirname)
        
        # helper to mock upload
        with open(CATALOG_PATH, 'r') as f:
            catalog = json.load(f)
            
        entry = catalog[0] # Test first one
        file_path = os.path.join(SAMPLE_DIR, entry['file_path'])
        
        with open(file_path, 'rb') as f:
            content = f.read()
            
        # Mock upload
        mcp.upload_file(content, entry['file_path'])
        
        # Process
        studies = mcp.get_experiments()
        assert len(studies) > 0, "MCP failed to extract studies from sample"
        
        # Validate
        report = mcp.validate_data(studies)
        assert report.is_valid, f"Validation failed for {entry['name']}: {report.issues}"
