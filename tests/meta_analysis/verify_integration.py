"""
Feature 3: Meta-Analysis Integration Verification Script

This script simulates an end-to-end workflow:
1. Initialize the MetaAnalysisAgent (with mocked mocks/stubs where needed or real if possible).
2. Create a dummy experiment CSV file.
3. Upload/ingest this file via the CSVExperimentMCP.
4. Run a full analysis via the Agent.
5. Verify the output state and standard output.
"""

import os
import sys
import unittest
import pandas as pd
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.meta_analysis.orchestration.meta_analysis_agent import MetaAnalysisAgent, AnalysisStage
from src.meta_analysis.mcp_servers.csv_experiment_server import CSVExperimentMCP
from src.meta_analysis.integration.rag_bridge import MetaAnalysisRAGBridge
from src.meta_analysis.integration.llm_bridge import MetaAnalysisLLMBridge

class IntegrationVerify(unittest.TestCase):
    
    def setUp(self):
        # Setup temporary data directory
        self.test_dir = Path("tests/temp_data")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a dummy CSV file
        self.csv_path = self.test_dir / "test_campaigns.csv"
        data = {
            "experiment_name": ["Exp A", "Exp B", "Exp C"],
            "control_visitors": [1000, 1000, 1000],
            "control_conversions": [100, 100, 100],
            "treatment_visitors": [1000, 1000, 1000],
            "treatment_conversions": [110, 120, 130] # Increasing positive effect
        }
        pd.DataFrame(data).to_csv(self.csv_path, index=False)
        
        # Initialize components with test config
        self.mcp = CSVExperimentMCP(data_dir=str(self.test_dir))
        
        # We can use real bridges but they might fail if RAG/LLM not ready, 
        # but the code handles that gracefully (fallback mode)
        self.rag = MetaAnalysisRAGBridge()
        self.llm = MetaAnalysisLLMBridge()
        
        self.agent = MetaAnalysisAgent(
            csv_mcp=self.mcp,
            rag_bridge=self.rag,
            llm_bridge=self.llm
        )
        
    def tearDown(self):
        # Cleanup
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_full_flow(self):
        print("\n--- Starting Integration Verification ---")
        
        # 1. Ingest Data (Implicitly handled by Agent calling MCP, but let's ensure MCP sees it)
        experiments = self.mcp.list_experiments()
        print(f"Found {len(experiments)} experiments available.")
        self.assertTrue(len(experiments) >= 3, "Should find at least 3 experiments from the CSV")
        
        # 2. Run Agent
        query = "Run meta-analysis using random effects model"
        print(f"Running agent with query: '{query}'")
        
        state = self.agent.run(query=query)
        
        # 3. Verify Stages
        print(f"Final Stage: {state.current_stage}")
        self.assertEqual(state.current_stage, AnalysisStage.COMPLETE)
        
        # 4. Verify Results
        self.assertIsNotNone(state.meta_result)
        print(f"Pooled Effect: {state.meta_result.pooled_effect:.4f}")
        print(f"P-value: {state.meta_result.p_value:.4f}")
        
        # Check if effect is positive (as expected from data)
        self.assertGreater(state.meta_result.pooled_effect, 0)
        
        # 5. Verify Artifacts
        self.assertTrue(len(state.visualizations) > 0, "Should generate visualizations")
        self.assertTrue(len(state.final_report) > 0, "Should generate a report")
        
        print("--- Integration Verification Successful ---")

if __name__ == "__main__":
    unittest.main()
