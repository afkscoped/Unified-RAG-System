# Advanced Unified RAG System - Technical Documentation

## Abstract

This document presents a comprehensive technical overview of the Advanced Unified RAG (Retrieval-Augmented Generation) System, a production-ready implementation designed for efficient document question-answering on consumer hardware. The system integrates hybrid search combining lexical (BM25) and semantic (FAISS) retrieval, semantic response caching, and adaptive weight learning to continuously improve answer quality based on user feedback.

---

## 1. Introduction and Theoretical Foundation

### 1.1 The RAG Paradigm

Retrieval-Augmented Generation addresses a fundamental limitation of Large Language Models: their knowledge is frozen at training time and prone to hallucination. RAG systems augment LLM capabilities by retrieving relevant context from an external knowledge base before generation, grounding responses in factual source material.

The core pipeline follows three stages: **Indexing** (processing documents into searchable chunks), **Retrieval** (finding relevant context for a query), and **Generation** (producing answers conditioned on retrieved context).

### 1.2 Hybrid Search Theory

Traditional RAG systems employ either lexical or semantic search. Lexical methods like BM25 (Best Matching 25) excel at exact term matching—critical for technical documentation where specific identifiers, function names, or acronyms must be matched precisely. However, they fail to capture semantic similarity; "machine learning" and "ML algorithms" would not match despite conceptual equivalence.

Semantic search using dense vector embeddings addresses this by mapping text to high-dimensional vector spaces where semantically similar content clusters together. The limitation is that these methods may overlook exact keyword matches that carry high signal.

Our hybrid approach combines both methods using **Reciprocal Rank Fusion (RRF)**, a score aggregation technique that merges ranked result lists:

```
RRF(d) = Σ [weight / (k + rank(d))]
```

Where `k` is a smoothing constant (default 60) that prevents excessive weight on top-ranked items, and `weight` represents the contribution from each search method.

### 1.3 Adaptive Weight Learning

Static hybrid weights (e.g., 50% semantic, 50% lexical) assume uniform query characteristics. In practice, queries vary significantly:

- **Keyword queries** ("Python API authentication") benefit from lexical dominance
- **Conceptual queries** ("explain how neural networks learn") require semantic emphasis
- **Mixed queries** balance both approaches

Our adaptive system classifies queries and learns optimal weights per category through user feedback, implementing a rudimentary reinforcement signal where ratings above threshold reinforce current weights while poor ratings trigger adjustment.

### 1.4 Semantic Caching

LLM inference represents the primary latency and cost bottleneck. Semantic caching exploits the observation that users often ask semantically equivalent questions. Rather than string-matching cached queries, we compute embedding similarity—if a new query's embedding achieves ≥0.95 cosine similarity with a cached query, we return the cached response immediately, achieving sub-millisecond response times.

---

## 2. System Architecture

### 2.1 High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│              (Streamlit UI / FastAPI REST)                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                   UnifiedRAGSystem                              │
│                  (src/core/rag_system.py)                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • Document Ingestion & Chunking                        │   │
│  │  • Query Orchestration                                  │   │
│  │  • Metrics Collection                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────┬──────────────┬──────────────┬──────────────┬───────────┘
        │              │              │              │
   ┌────▼────┐   ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
   │Embedding│   │  Hybrid   │  │ Semantic  │  │   LLM     │
   │ Manager │   │  Search   │  │   Cache   │  │  Router   │
   │         │   │  Engine   │  │           │  │           │
   └─────────┘   └───────────┘  └───────────┘  └───────────┘
        │              │
   ┌────▼────┐   ┌─────▼─────┐
   │  FAISS  │   │   BM25    │
   │  Index  │   │   Index   │
   └─────────┘   └───────────┘
```

### 2.2 Directory Structure

```
unified-rag-system/
├── config/
│   └── config.yaml           # Centralized configuration
├── src/
│   ├── core/
│   │   ├── rag_system.py     # Main orchestrator (412 lines)
│   │   ├── embeddings.py     # GPU-optimized embedding manager
│   │   ├── hybrid_search.py  # BM25 + FAISS with RRF fusion
│   │   └── weight_manager.py # Adaptive query classification
│   ├── cache/
│   │   └── semantic_cache.py # Similarity-based response caching
│   ├── llm/
│   │   └── llm_router.py     # Groq/Ollama with fallback
│   ├── api/
│   │   ├── fastapi_app.py    # REST API endpoints
│   │   └── streamlit_app.py  # Interactive web UI
│   └── utils/
│       ├── memory_monitor.py # RAM/VRAM tracking
│       └── gpu_manager.py    # CUDA memory management
├── tests/                    # Pytest test suites
├── scripts/
│   └── download_models.py    # Setup automation
└── data/                     # Runtime storage
```

---

## 3. Core Component Implementation

### 3.1 EmbeddingManager (embeddings.py)

The embedding manager wraps SentenceTransformers with GPU optimization:

- **Model**: BAAI/bge-small-en-v1.5 (384 dimensions, 33M parameters)
- **Precision**: FP16 half-precision for 50% memory reduction
- **Batching**: Processes 32 texts per batch to prevent OOM
- **LangChain Integration**: `LangChainEmbeddingsWrapper` adapts our manager to LangChain's `Embeddings` interface for FAISS compatibility

Key method:
```python
def encode(self, texts: List[str]) -> np.ndarray:
    # Batch processing with autocast for mixed precision
    with torch.cuda.amp.autocast():
        embeddings = self.model.encode(batch, normalize_embeddings=True)
```

### 3.2 HybridSearchEngine (hybrid_search.py)

Implements dual-index architecture:

**BM25 Index**: Built using rank-bm25 library with basic tokenization. Excels at rare term matching but requires exact lexical overlap.

**FAISS Index**: Dense vector index using LangChain's FAISS wrapper. Enables approximate nearest neighbor search in embedding space.

**RRF Fusion**: Merges results by iterating both result sets, computing RRF scores, and returning top-k by combined score:

```python
for rank, (doc, score) in enumerate(semantic_results):
    rrf_score = semantic_weight / (self.rrf_k + rank + 1)
    combined_scores[doc_key] += rrf_score
```

### 3.3 AdaptiveWeightManager (weight_manager.py)

Query classification heuristics:
- **Keyword**: ≤3 words → favor lexical (0.3 semantic, 0.7 lexical)
- **Conceptual**: Contains "explain", "what is", "how does" → favor semantic (0.7, 0.3)
- **Mixed**: Default balanced weights (0.5, 0.5)

Feedback learning accumulates (rating, weight) tuples per query type. After 5+ samples, if average rating falls below 60%, weights shift ±0.1 toward the better-performing configuration.

### 3.4 SemanticCache (semantic_cache.py)

Implements an LRU cache with embedding-based lookup:

```python
def get(self, query: str, query_embedding: np.ndarray) -> Optional[str]:
    for cached_query, entry in self._cache.items():
        similarity = cosine_similarity(query_embedding, entry["embedding"])
        if similarity >= 0.95:
            self._cache.move_to_end(cached_query)  # LRU update
            return entry["response"]
```

Features TTL expiration (default 1 hour) and configurable capacity (default 100 entries).

### 3.5 LLMRouter (llm_router.py)

Implements provider abstraction with automatic fallback:

1. **Primary**: Groq API (llama-3.1-8b-instant) - fast cloud inference
2. **Fallback**: Ollama (local) - privacy-preserving alternative

Rate limiting prevents API quota exhaustion. Statistics tracking enables monitoring provider health.

---

## 4. Memory Management

The system enforces strict resource bounds for 16GB RAM / 6GB VRAM hardware:

| Resource | Limit | Enforcement |
|----------|-------|-------------|
| RAM | 12GB (75%) | MemoryMonitor triggers gc.collect() at 80% |
| VRAM | 4.8GB (80%) | torch.cuda.set_per_process_memory_fraction(0.8) |
| Batch Size | 32 | Prevents embedding OOM |

The MemoryMonitor runs periodic checks during query processing, triggering cleanup when thresholds exceed:

```python
if ram_percent > self.ram_threshold:
    gc.collect()
if vram_percent > self.vram_threshold:
    torch.cuda.empty_cache()
```

---

## 5. API Layer

### 5.1 FastAPI (fastapi_app.py)

RESTful endpoints:
- `POST /query` - Main question-answering endpoint
- `POST /ingest` - Document upload with multipart/form-data
- `POST /feedback` - Rating submission for weight learning
- `GET /metrics` - System telemetry

CORS middleware enables cross-origin requests for frontend integration.

### 5.2 Streamlit (streamlit_app.py)

Interactive UI featuring:
- File uploader supporting PDF, TXT, DOCX
- Real-time metrics dashboard (queries, cache hits, memory)
- Chat interface with streaming display
- Feedback slider for response rating

Session state persists the RAG system instance across reruns.

---

## 6. Conclusion

This Advanced RAG System demonstrates production-grade patterns for retrieval-augmented generation: hybrid search for robust retrieval, semantic caching for efficiency, and adaptive learning for continuous improvement. The modular architecture enables component substitution—alternative embedding models, vector stores, or LLM providers can be integrated with minimal refactoring.

The system achieves practical deployment on consumer hardware through aggressive memory optimization while maintaining sub-second query latency. User feedback drives weight adaptation, progressively tailoring retrieval behavior to specific document collections without manual tuning.

---

**Word Count**: ~1,480 words
