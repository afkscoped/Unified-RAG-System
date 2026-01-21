# 🤖 Unified RAG System + StoryWeaver & QLoRA Studio

A production-ready Retrieval-Augmented Generation platform with a **High-Tech / Cyberpunk** aesthetic. Optimized for consumer hardware (16GB RAM + 6GB VRAM).

## ✨ Features

### 🎨 Cyberpunk / Glitch UI
- **Neon Glitch Aesthetic**: A dark, high-contrast "High-Tech, Low-Life" design system.
- **Interactive HUD**: Landing page with "Cyber Cards" and glitch hover effects.
- **Scanline Overlays**: CRT-monitor texture for an immersive terminal feel.

### 🧬 QLoRA Training Studio
- **Fine-Tune LLMs Locally**: Complete QLoRA fine-tuning pipeline for consumer GPUs.
- **Groq API Data Gen**: Instant high-quality dataset generation via Groq cloud.
- **Model Tester**: Side-by-side testing of base vs. fine-tuned models.
- **Export & Deploy**: One-click model merging and export.

### 📄 Document Analyzer Mode
- **Hybrid Search**: BM25 lexical + FAISS semantic with RRF fusion.
- **Semantic Caching**: Extreme latency reduction via similarity caching.
- **Persona Engine**: Adaptive response styles (General, Technical, Executive).
- **Multi-format Support**: PDF, TXT, DOCX, XLSX.

### ✍️ StoryWeaver Mode
- **Comparative RAG**: Compare Unified RAG, Graph RAG, and Hybrid Fusion levels.
- **Advanced Graph RAG**: Knowledge graph extraction with entity and arc tracking.
- **Narrative Analysis**: Coherence scoring, consistency checking, and plot hole detection.
- **Interactive Visualization**: Real-time Plotly-based story graphs and timelines.

### 🧪 Meta-Analysis Engine
- **A/B Test Analytics**: Statistical engine for aggregating experiment results.
- **Synthetic Data**: AI-powered simulation data generation (CTGAN).
- **Bias Detection**: Egger's regression and Funnel plots.

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure API Keys

Create a `.env` file in the root:
```env
GROQ_API_KEY=your_key_here
OLLAMA_HOST=http://localhost:11434
```

### 3. Run the Platform

**Main Entry Point (Landing Page):**
```bash
streamlit run src/ui/landing_page.py
```

**Background API (FastAPI):**
```bash
uvicorn src.api.fastapi_app:app --reload --port 8000
```

---

## 📁 Project Structure

```
unified-rag-system/
├── src/
│   ├── ui/
│   │   ├── landing_page.py      # Main Hub
│   │   ├── styles/               # Cyberpunk Theme & CSS
│   │   └── pages/                # App Modules
│   ├── elite_app/               # QLoRA Training Studio Core
│   ├── core/                    # RAG & Search Engines
│   ├── story/                   # StoryWeaver Engine
│   └── api/                     # FastAPI Backend
├── config/                      # System YAML configs
├── data/                        # Local database/document storage
└── cache/                       # LLM & Embedding caches
```

---

## 🔧 Hardware Requirements

- **Minimum**: 8GB RAM, CPU only
- **Recommended**: 16GB RAM, NVIDIA GPU (6GB+ VRAM)
- **Primary stack**: Python 3.10+, Streamlit, FastAPI, FAISS, PyTorch, Transformers.

## 📄 License

MIT License
