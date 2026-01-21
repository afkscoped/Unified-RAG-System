"""
[ELITE ARCHITECTURE] elite_rag.py
Unified Orchestrator for Advanced RAG + QLoRA Intelligence.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import yaml

from src.preprocessing.parsers.layout_parser import LayoutParser
from src.preprocessing.chunking.semantic_chunker import SemanticChunker
from src.retrieval.multi_vector.hyde_generator import HyDEGenerator
from src.retrieval.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.query_processing.intent_classifier import IntentClassifier, QueryIntent
from src.query_processing.query_decomposer import QueryDecomposer
from src.generation.cot.cot_generator import AdaptiveCoTGenerator
from src.generation.citation_manager import CitationManager
from src.evaluation.ragas_evaluator import EliteRagasEvaluator
from src.optimization.cache.multi_level_cache import MultiLevelCache
from src.production.security.pii_detector import PIIDetector
from src.knowledge_graph.graph_engine import KnowledgeGraphEngine
from src.production.security.encryption import VaultManager
from src.production.security.rate_limiter import RateLimiter
from src.production.monitoring.audit_logger import AuditLogger
from src.core.model_registry import ModelRegistry
from src.production.api.async_jobs import AsyncJobManager

from src.elite.triple_extractor import TripleExtractor
from src.elite.dialectical_engine import DialecticalSynthesizer

class EliteRAGSystem:
    """
    Main orchestrator for the Elite Document Analyzer.
    Integrates all Phase 1-4 modules into a seamless pipeline.
    """
    
    def __init__(self, config_path: str = "config/config.yaml", llm_router=None, embedding_manager=None, search_engine=None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.llm = llm_router
        self.emb = embedding_manager
        self.search = search_engine
        
        # Initialize Sub-modules
        self.parser = LayoutParser(self.config)
        self.chunker = SemanticChunker(self.emb, threshold=self.config['preprocessing']['chunking']['semantic_threshold'])
        self.hyde = HyDEGenerator(self.llm)
        self.reranker = CrossEncoderReranker(self.config['retrieval']['reranking']['model'])
        self.cot = AdaptiveCoTGenerator(self.llm)
        self.citations = CitationManager()
        self.evaluator = EliteRagasEvaluator(self.llm)
        self.cache = MultiLevelCache(self.config['paths']['cache'])
        self.pii = PIIDetector()
        self.kg = KnowledgeGraphEngine()
        
        # Elite Modules (New)
        self.triple_extractor = TripleExtractor(self.llm)
        self.dialectical = DialecticalSynthesizer(self.llm, self.search)
        
        # Strategic Intelligence
        self.intent = IntentClassifier(self.llm)
        self.decomposer = QueryDecomposer(self.llm)
        
        # Production Hardening
        self.vault = VaultManager()
        self.limiter = RateLimiter(max_requests=self.config['optimization'].get('rate_limit', 5))
        self.auditor = AuditLogger()
        self.registry = ModelRegistry()
        self.async_manager = AsyncJobManager()
        
        logger.success("Elite RAG System Orchestrator: INITIALIZED")

    def ingest_document(self, file_path: str):
        """
        Full ingestion pipeline: Parse -> Redact -> Chunk -> Index -> Graph.
        """
        # 1. Parse Layout
        elements = self.parser.parse_pdf(file_path)
        
        processed_chunks = []
        for element in elements:
            # 2. Privacy Redaction
            safe_content, count = self.pii.redact(element['content'])
            
            # 3. Semantic Chunking
            chunks = self.chunker.chunk_text(safe_content)
            
            # 4. Generate Embeddings & Index
            chunk_embeddings = self.emb.encode(chunks)
            
            for i, chunk_body in enumerate(chunks):
                # 5. Knowledge Graph Enrichment
                self.kg.extract_and_add(file_path, chunk_body, element['metadata'])
                
                # 6. Core Search Engine Indexing
                self.search.add_document(
                    content=chunk_body, 
                    metadata=element['metadata'],
                    embedding=chunk_embeddings[i]
                )
                processed_chunks.append(chunk_body)
                
        # 7. Background: Triple Extraction (Adversarial Prep)
        if processed_chunks:
            # We assume metadata is similar for all chunks in one file pass, or we pass specifics
            # For simplicity, pass the first metadata. Ideally, should be per-chunk.
            meta = elements[0]['metadata'] if elements else {}
            self.triple_extractor.extract_async(processed_chunks, meta)
                
        logger.info(f"Ingestion complete: {len(processed_chunks)} chunks indexed.")
        return len(processed_chunks)

    def get_corpus_data(self) -> Dict[str, Any]:
        """
        Returns all indexed data for visualization purposes.
        """
        # Logic depends on the underlying vector store (e.g., FAISS)
        # Assuming we store them in a list for this research implementation
        return {
            "embeddings": self.search.get_all_embeddings(), # Requires implementation in search.py
            "texts": self.search.get_all_texts(),
            "sources": self.search.get_all_sources()
        }

    def query(self, question: str, deep_audit: bool = False) -> Dict[str, Any]:
        """
        Full retrieval pipeline: Security -> Cache -> HyDE -> Hybrid Search -> Rerank -> CoT -> Eval.
        """
        # 1. Rate Limiting check
        if not self.limiter.allow_request():
            raise Exception("Rate Limit Protocol Active: Please wait before next request.")
            
        # 2. Audit Logging
        self.auditor.log_action("SYSTEM", "QUERY_START", {"query": question[:50]})
        
        # 3. Cache Check
        cached = self.cache.get(question)
        if cached: return {**cached, "is_cached": True}
        
        # 2. Strategic Routing (Intent & Decomposition)
        intent = self.intent.classify(question)
        sub_queries = self.decomposer.decompose(question) if intent == QueryIntent.CROSS_DOCUMENT else [question]
        
        logger.info(f"Intent Detetected: {intent.value} | Research Plan: {len(sub_queries)} steps.")
        
        all_candidates = []
        for sq in sub_queries:
            # 3. HyDE Expansion for each sub-query
            pseudo_doc = self.hyde.generate_pseudo_doc(sq)
            sq_embedding = self.emb.get_embeddings([pseudo_doc])[0]
            
            # 4. Hybrid Retrieval
            candidates = self.search.search(sq_embedding, k=15 if intent == QueryIntent.SUMMARIZATION else 10)
            all_candidates.extend(candidates)
            
        # Deduplicate candidates
        seen_chunks = set()
        unique_candidates = []
        for c in all_candidates:
            if c.content not in seen_chunks:
                unique_candidates.append(c)
                seen_chunks.add(c.content)
        
        # 5. Precision Reranking
        top_docs = self.reranker.rerank(question, unique_candidates, top_n=self.config['retrieval']['reranking']['top_n'])
        
        # 5. Adaptive Generation (CoT) OR Dialectical Debate
        response = {}
        debate_data = {}
        
        if deep_audit:
            # Run Adversarial Loop
            debate_result = self.dialectical.generate_debate(question, deep_audit=True)
            response['answer'] = debate_result['synthesis']
            response['thought'] = "Adversarial Debate Conducted (See Thought Trace)"
            debate_data = debate_result
        else:
            response = self.cot.generate_adaptive_response(question, top_docs)
        
        # 6. Citation Mapping
        citations = self.citations.resolve_citations(response['answer'], top_docs)
        
        # 7. Judicial Evaluation (RAGAS)
        eval_scores = self.evaluator.evaluate_response(question, response['answer'], top_docs)
        
        result = {
            "query": question,
            "answer": response['answer'],
            "thought": response['thought'],
            "citations": citations,
            "evaluation": eval_scores,
            "is_cached": False,
            "source_docs": top_docs,
            "debate_trace": debate_data
        }
        
        # 8. Store in Cache
        self.cache.set(question, result)
        
        return result
