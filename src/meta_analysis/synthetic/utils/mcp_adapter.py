"""
MCP Adapter for Synthetic Data

Routes synthetic data through the MCP ingestion layer to ensure
consistent validation and standardization with real data.
"""

import io
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from loguru import logger

from src.meta_analysis.mcp_servers.base_experiment_mcp import (
    StandardizedStudy,
    ValidationReport as MCPValidationReport
)
from src.meta_analysis.mcp_servers.csv_experiment_server import CSVExperimentMCP


class MCPIngestionResult:
    """Result of MCP ingestion including studies and validation."""
    
    def __init__(
        self,
        studies: List[StandardizedStudy],
        validation_report: MCPValidationReport,
        source_type: str,
        server_name: str,
        timestamp: datetime
    ):
        self.studies = studies
        self.validation_report = validation_report
        self.source_type = source_type
        self.server_name = server_name
        self.timestamp = timestamp
        self.study_count = len(studies)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for session state storage."""
        return {
            "study_count": self.study_count,
            "source_type": self.source_type,
            "server_name": self.server_name,
            "timestamp": self.timestamp.isoformat(),
            "is_valid": self.validation_report.is_valid,
            "valid_records": self.validation_report.valid_records,
            "total_records": self.validation_report.total_records,
            "error_count": self.validation_report.error_count,
            "warning_count": self.validation_report.warning_count,
            "issues": [
                {"field": i.field, "type": i.issue_type, "message": i.message}
                for i in self.validation_report.issues[:10]  # Limit to first 10
            ]
        }


class SyntheticMCPAdapter:
    """
    Adapter to route synthetic data through MCP ingestion.
    
    Ensures synthetic data follows the same validation and standardization
    path as real uploaded or Kaggle-imported data.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize adapter with MCP server.
        
        Args:
            data_dir: Directory for MCP server (uses temp if not provided)
        """
        if data_dir is None:
            data_dir = tempfile.mkdtemp(prefix="synthetic_mcp_")
        
        self.data_dir = data_dir
        self.mcp_server = CSVExperimentMCP(data_dir=data_dir)
        self.server_name = "CSVExperimentMCP"
        
        logger.info(f"Initialized SyntheticMCPAdapter with data_dir: {data_dir}")
    
    def ingest_synthetic_data(
        self,
        df: pd.DataFrame,
        source_name: str = "synthetic_generator"
    ) -> MCPIngestionResult:
        """
        Ingest synthetic DataFrame through MCP server.
        
        Args:
            df: Synthetic data DataFrame
            source_name: Name for the synthetic source file
            
        Returns:
            MCPIngestionResult with studies and validation
        """
        timestamp = datetime.now()
        
        # 1. Convert DataFrame to CSV buffer
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        
        logger.info(f"Converting {len(df)} synthetic rows to CSV buffer")
        
        # 2. Upload to MCP server
        filename = f"{source_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
        saved_path = self.mcp_server.upload_file(csv_bytes, filename)
        
        logger.info(f"Uploaded synthetic data to MCP: {saved_path}")
        
        # 3. Parse through MCP to get StandardizedStudy objects
        studies = self.mcp_server.parse_ab_test_csv(saved_path)
        
        logger.info(f"MCP parsed {len(studies)} studies from synthetic data")
        
        # 4. Validate through MCP
        validation_report = self.mcp_server.validate_data(studies)
        
        logger.info(f"MCP validation: {validation_report.valid_records}/{validation_report.total_records} valid")
        
        return MCPIngestionResult(
            studies=studies,
            validation_report=validation_report,
            source_type="synthetic",
            server_name=self.server_name,
            timestamp=timestamp
        )
    
    def get_standardized_studies(self, df: pd.DataFrame) -> List[StandardizedStudy]:
        """
        Quick method to get standardized studies from synthetic data.
        
        Args:
            df: Synthetic DataFrame
            
        Returns:
            List of StandardizedStudy objects
        """
        result = self.ingest_synthetic_data(df)
        return result.studies
    
    def validate_synthetic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate synthetic data and return validation dict.
        
        Args:
            df: Synthetic DataFrame
            
        Returns:
            Validation result dictionary
        """
        result = self.ingest_synthetic_data(df)
        return result.to_dict()


def ingest_via_mcp(
    df: pd.DataFrame,
    data_dir: Optional[str] = None,
    source_name: str = "synthetic"
) -> MCPIngestionResult:
    """
    Convenience function to ingest synthetic data via MCP.
    
    Args:
        df: Synthetic DataFrame
        data_dir: Optional MCP data directory
        source_name: Source identifier
        
    Returns:
        MCPIngestionResult with studies and validation
    """
    adapter = SyntheticMCPAdapter(data_dir=data_dir)
    return adapter.ingest_synthetic_data(df, source_name)
