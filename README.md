# 🤖 Unified RAG System + StoryWeaver

A production-ready Retrieval-Augmented Generation platform with two powerful modes:
- **📄 Document Analyzer** - Hybrid search RAG for document Q&A
- **✍️ StoryWeaver** - Comparative story generation with Graph RAG

Optimized for consumer hardware (16GB RAM + 6GB VRAM).

## ✨ Features

### 📄 Document Analyzer Mode
- **Hybrid Search**: BM25 lexical + FAISS semantic with RRF fusion
- **Semantic Caching**: Cache similar queries (0.95 threshold)
- **Adaptive Weights**: Auto-adjust search based on feedback
- **Multi-format Support**: PDF, TXT, DOCX

### ✍️ StoryWeaver Mode
- **Three RAG Approaches**: Side-by-side comparison
  - Unified RAG (vector-based retrieval)
  - Graph RAG (knowledge graph + entity tracking)
  - Hybrid Fusion (combined approach)
- **Multi-dimensional Analysis**:
  - Coherence scoring (semantic, lexical, discourse, temporal)
  - Consistency checking with violation detection
  - Plot suggestions with actionable prompts
- **Visualization**:
  - Interactive knowledge graph (Plotly)
  - Plot timeline with structure analysis
- **Feedback System**: 5-dimension ratings with adaptive learning
- **Character Arc Tracking**: Automatic emotional state and goal extraction
- **Story Export**: Markdown export with chapter organization

### Core Platform
- **Dual API**: FastAPI REST + Streamlit Web UI
- **LLM Routing**: Groq (primary) + Ollama (fallback)
- **Memory Optimized**: 16GB RAM + RTX 4050 (6GB VRAM)
- **SpaCy NER**: Entity and relationship extraction

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure API Keys

```bash
copy .env.example .env
# Edit .env and add your Groq API key
# Get free key at: https://console.groq.com
```

### 3. Run the App

**Landing Page (Mode Selection):**
```bash
streamlit run src/ui/landing_page.py
```

**Document Analyzer Only:**
```bash
streamlit run src/ui/pages/document_app.py
```

**StoryWeaver Only:**
```bash
streamlit run src/ui/pages/story_app.py
```

**FastAPI REST:**
```bash
uvicorn src.api.fastapi_app:app --reload
```
Open http://localhost:8000/docs

---

## 📖 Usage

### StoryWeaver UI

1. **Generation Tab**: Enter prompts, compare three approaches
2. **Visualization Tab**: Interactive graph and timeline
3. **Feedback Tab**: Rate generations (5 dimensions)
4. **Analytics Tab**: View performance trends

### Document Analyzer

1. Upload documents (PDF, TXT, DOCX)
2. Click "Process Documents"
3. Ask questions in chat
4. Rate responses for weight adaptation

### Python SDK

```python
from src.core.rag_system import UnifiedRAGSystem

rag = UnifiedRAGSystem()
rag.ingest_file("document.pdf")
rag.build_index()
result = rag.query("What are the findings?")
print(result.answer)
```

### Story Generation API

```python
from src.story.comparison_engine import StoryComparisonEngine
from src.llm.llm_router import LLMRouter

engine = StoryComparisonEngine(llm_router=LLMRouter())
results = engine.generate_comparative("Elena discovers a hidden chamber...", chapter=1)

print(results["unified"]["text"])  # Unified RAG output
print(results["graph"]["text"])    # Graph RAG output
print(results["hybrid"]["text"])   # Hybrid output
```

---

## 📁 Project Structure

```
unified-rag-system/
├── config/config.yaml
├── src/
│   ├── core/                    # Document RAG
│   │   ├── rag_system.py
│   │   ├── hybrid_search.py
│   │   └── embeddings.py
│   ├── story/                   # StoryWeaver
│   │   ├── comparison_engine.py
│   │   ├── narrative_manager.py
│   │   ├── graph_rag/
│   │   │   ├── story_graph.py
│   │   │   ├── entity_extractor.py
│   │   │   ├── arc_tracker.py
│   │   │   └── relationship_discovery.py
│   │   ├── analysis/
│   │   │   ├── coherence_analyzer.py
│   │   │   ├── consistency_checker.py
│   │   │   └── plot_suggestion_engine.py
│   │   ├── visualization/
│   │   │   ├── graph_visualizer.py
│   │   │   └── plot_timeline.py
│   │   ├── feedback/
│   │   │   └── feedback_manager.py
│   │   ├── unified_rag/
│   │   │   └── story_adapter.py
│   │   └── fusion/
│   │       └── hybrid_generator.py
│   ├── ui/
│   │   ├── landing_page.py
│   │   └── pages/
│   │       ├── document_app.py
│   │       └── story_app.py
│   ├── api/
│   │   ├── fastapi_app.py
│   │   └── story_api.py
│   ├── llm/
│   │   └── llm_router.py
│   └── cache/
│       └── semantic_cache.py
├── tests/
└── data/
```

---

## 🔧 Configuration

Edit `config/config.yaml`:

```yaml
model:
  embedding: "BAAI/bge-small-en-v1.5"
  llm_provider: "groq"
  llm_model: "llama-3.1-8b-instant"
  device: "cuda"

story:
  entity_extraction: true
  arc_tracking: true

graph_rag:
  max_hops: 3
  similarity_threshold: 0.7
```

---

## 🧪 Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

---

## 🔧 Hardware Requirements

- **Minimum**: 8GB RAM, CPU only
- **Recommended**: 16GB RAM, NVIDIA GPU (6GB+ VRAM)
- **Dependencies**: SpaCy, NetworkX, Plotly, Streamlit

---

## 📄 License

MIT License

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - LLM orchestration
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [Sentence Transformers](https://sbert.net/) - Embeddings
- [Groq](https://groq.com/) - Fast LLM inference
- [SpaCy](https://spacy.io/) - NLP/NER
- [NetworkX](https://networkx.org/) - Graph processing
- [Plotly](https://plotly.com/) - Visualization
