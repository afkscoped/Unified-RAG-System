"""
Judicial Evaluation Module
Scores every response for faithfulness and relevance
"""

from typing import Dict, List
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import re


@dataclass
class EvaluationResult:
    """Evaluation scores for a response"""
    faithfulness_score: float  # 0-1: How well grounded in sources
    relevance_score: float     # 0-1: How relevant to query
    citation_coverage: float   # 0-1: % of claims with citations
    overall_score: float       # Weighted average
    detailed_metrics: Dict
    warnings: List[str]


class JudicialEvaluator:
    """
    Real-time response evaluation using:
    - Faithfulness: Claims grounded in sources
    - Relevance: Response addresses query
    - Citation coverage: Proper attribution
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._embedder = None
        self._embedding_model = embedding_model
        self._device = device
    
    @property
    def embedder(self):
        """Lazy load embedder"""
        if self._embedder is None:
            self._embedder = SentenceTransformer(self._embedding_model, device=self._device)
        return self._embedder
    
    def evaluate(
        self,
        query: str,
        response: str,
        sources: List[str]
    ) -> EvaluationResult:
        """
        Comprehensive evaluation of response quality
        """
        # Compute individual metrics
        faithfulness = self._evaluate_faithfulness(response, sources)
        relevance = self._evaluate_relevance(query, response)
        citation_cov = self._evaluate_citation_coverage(response)
        
        # Detect warnings
        warnings = self._detect_warnings(response, sources, faithfulness, relevance)
        
        # Compute overall score (weighted)
        overall = (
            faithfulness * 0.5 +
            relevance * 0.3 +
            citation_cov * 0.2
        )
        
        detailed = {
            'num_sources': len(sources),
            'response_length': len(response.split()),
            'query_length': len(query.split()),
            'citations_found': self._count_citations(response)
        }
        
        return EvaluationResult(
            faithfulness_score=faithfulness,
            relevance_score=relevance,
            citation_coverage=citation_cov,
            overall_score=overall,
            detailed_metrics=detailed,
            warnings=warnings
        )
    
    def _evaluate_faithfulness(self, response: str, sources: List[str]) -> float:
        """Measure how well response is grounded in sources"""
        if not sources or not response:
            return 0.0
        
        sentences = self._split_sentences(response)
        if not sentences:
            return 0.0
        
        # Encode sentences and sources
        sentence_embs = self.embedder.encode(sentences, convert_to_numpy=True)
        source_embs = self.embedder.encode(sources, convert_to_numpy=True)
        
        # For each sentence, find max similarity with any source
        faithfulness_scores = []
        for sent_emb in sentence_embs:
            similarities = [
                self._cosine_similarity(sent_emb, src_emb)
                for src_emb in source_embs
            ]
            max_sim = max(similarities) if similarities else 0.0
            faithfulness_scores.append(max_sim)
        
        return float(np.mean(faithfulness_scores))
    
    def _evaluate_relevance(self, query: str, response: str) -> float:
        """Measure how relevant response is to query"""
        if not query or not response:
            return 0.0
            
        query_emb = self.embedder.encode([query], convert_to_numpy=True)[0]
        response_emb = self.embedder.encode([response[:1000]], convert_to_numpy=True)[0]
        
        return self._cosine_similarity(query_emb, response_emb)
    
    def _evaluate_citation_coverage(self, response: str) -> float:
        """Measure what % of factual claims have citations"""
        sentences = self._split_sentences(response)
        if not sentences:
            return 0.0
        
        cited_sentences = sum(
            1 for sent in sentences
            if re.search(r'\[\d+\]', sent)
        )
        
        return cited_sentences / len(sentences)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def _count_citations(self, text: str) -> int:
        """Count number of citation markers"""
        citations = re.findall(r'\[(\d+)\]', text)
        return len(set(citations))
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between vectors"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    
    def _detect_warnings(
        self,
        response: str,
        sources: List[str],
        faithfulness: float,
        relevance: float
    ) -> List[str]:
        """Detect potential issues with response"""
        warnings = []
        
        if faithfulness < 0.5:
            warnings.append("⚠️ Low faithfulness - response may not be well-supported by sources")
        
        if relevance < 0.6:
            warnings.append("⚠️ Low relevance - response may not address the query directly")
        
        if not sources:
            warnings.append("⚠️ No sources provided - cannot verify claims")
        
        if self._count_citations(response) == 0 and len(response) > 100:
            warnings.append("⚠️ No citations found - claims are not attributed")
        
        if len(response.split()) < 20:
            warnings.append("⚠️ Very short response - may be incomplete")
        
        return warnings
