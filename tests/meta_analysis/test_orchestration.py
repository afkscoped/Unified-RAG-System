"""
Tests for Orchestration
"""

import unittest
from unittest.mock import MagicMock, patch
from src.meta_analysis.orchestration.meta_analysis_agent import MetaAnalysisAgent, MetaAnalysisState, AnalysisStage
from src.meta_analysis.mcp_servers.base_experiment_mcp import StandardizedStudy, EffectSizeType


class TestOrchestrator(unittest.TestCase):
    
    def setUp(self):
        # Mock dependencies
        self.mock_csv_mcp = MagicMock()
        self.mock_rag = MagicMock()
        self.mock_llm = MagicMock()
        
        self.agent = MetaAnalysisAgent(
            csv_mcp=self.mock_csv_mcp,
            rag_bridge=self.mock_rag,
            llm_bridge=self.mock_llm
        )
        
        # Setup mock data return
        self.mock_studies = [
            StandardizedStudy("s1", "S1", 0.5, 0.1, 100, 100, "m", "t", EffectSizeType.LOG_ODDS_RATIO, "p", None),
            StandardizedStudy("s2", "S2", 0.6, 0.1, 100, 100, "m", "t", EffectSizeType.LOG_ODDS_RATIO, "p", None)
        ]
        self.mock_csv_mcp.get_experiments.return_value = self.mock_studies
        
        # Mock analysis components to avoid full execution
        # self.agent.meta_analyzer = MagicMock()
        # self.agent.meta_analyzer.analyze.return_value = MetaAnalysisResult(...)
        
    def test_workflow_execution(self):
        # Run workflow
        # Note: We let the real statistical components run as they are fast pure logic
        # But we mock the LLM/RAG parts
        
        self.mock_csv_mcp.validate_data.return_value = MagicMock(warnings=[])
        self.agent.meta_rag.enhance_statistical_results = MagicMock(return_value="Enhanced interpretation")
        self.agent.meta_rag.generate_recommendations = MagicMock(return_value=[{"title": "Rec", "message": "Msg", "action": "Act"}])
        
        state = self.agent.run(query="Analyze these experiments")
        
        self.assertEqual(state.current_stage, AnalysisStage.COMPLETE)
        self.assertEqual(len(state.standardized_studies), 2)
        self.assertIsNotNone(state.meta_result)
        self.assertEqual(state.rag_context, "Enhanced interpretation")
        
    def test_error_handling(self):
        # Simulate empty data
        self.mock_csv_mcp.get_experiments.return_value = []
        
        state = self.agent.run(query="Analyze")
        
        self.assertIn(AnalysisStage.ERROR, [state.current_stage])
        self.assertTrue(len(state.errors) > 0)

if __name__ == '__main__':
    unittest.main()
