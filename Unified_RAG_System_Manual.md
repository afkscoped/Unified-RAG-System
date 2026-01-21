# Unified RAG System + StoryWeaver: User Manual

**Version:** 1.0
**Date:** January 2026

---

## 1. Introduction

The **Unified RAG System + StoryWeaver** is a cutting-edge, production-ready platform designed for three distinct purposes:
1.  **Document Analysis**: Intelligent Q&A on your documents using Hybrid Search RAG.
2.  **Creative Writing**: Comparative story generation using Graph RAG and Unified RAG.
3.  **Statistical Analysis**: A comprehensive A/B Test Meta-Analysis Engine for decision making.

This manual provides a guide to setting up, running, and understanding the core features of the system, with a special deep dive into the **Meta-Analysis Engine**.

---

## 2. Quick Start Guide

### 2.1 Hardware Requirements
- **RAM**: Minimum 8GB (16GB Recommended)
- **GPU**: NVIDIA GPU with 6GB+ VRAM recommended for optimal performance (local LLM/embeddings).
- **OS**: Windows, Linux, or macOS.

### 2.2 Installation
1.  **Clone/Download** the repository.
2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    # Activate:
    # Windows: venv\Scripts\activate
    # Linux/Mac: source venv/bin/activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

### 2.3 Configuration
Create a `.env` file in the root directory (copy `.env.example`) and add your API keys:
```env
GROQ_API_KEY=your_key_here
```

### 2.4 Running the Application
Launch the main landing page to access all modes:
```bash
streamlit run src/ui/landing_page.py
```

---

## 3. Core Features Overview

### 3.1 📄 Document Analyzer
Using **Hybrid Search** (BM25 + FAISS), this mode allows you to upload PDFs, TXT, or DOCX files and ask questions. It features **Semantic Caching** to speed up repeated queries and **Adaptive Weights** that learn from your feedback.

### 3.2 ✍️ StoryWeaver
An advanced creative writing assistant that generates stories using three approaches simultaneously for comparison:
*   **Unified RAG**: Standard vector-based retrieval.
*   **Graph RAG**: Knowledge graph-based generation offering better consistency.
*   **Hybrid**: A fusion of both.

Features include **Character Arc Tracking**, **Timeline Visualization**, and **Coherence Scoring**.

---

## 4. Deep Dive: A/B Test Meta-Analysis Engine

The **Meta-Analysis Engine** is a powerful statistical tool designed to aggregate results from multiple A/B tests (e.g., across different regions or time periods) to provide a single, trustworthy conclusion.

### 4.1 Concepts Explained
*   **A/B Testing**: Comparing two versions (Control vs. Variant) to see which performs better.
*   **Meta-Analysis**: A "super-study" that mathematically combines multiple A/B tests to increase accuracy and detect patterns that single tests might miss.

### 4.2 Data Import Modes
The system enables data ingestion via four methods:

1.  **📁 Upload Files**:
    *   Upload your own `.csv` or `.xlsx` files.
    *   **Required Columns**: `control_conversions`, `control_total`, `treatment_conversions`, `treatment_total`.
2.  **🧪 Synthesize Data**:
    *   Generate realistic "fake" data using AI models (**CTGAN** or **CopulaGAN**). Ideal for testing and learning.
3.  **🌐 Import from Kaggle**:
    *   Directly download datasets (e.g., "marketing-ab-testing") from Kaggle.
4.  **📚 Sample Datasets**:
    *   One-click load of pre-built examples.

### 4.3 Understanding the Visualizations

#### **Forest Plot ("The Tree of Truth")**
The central visualization of meta-analysis.
*   **Rows**: Each line represents one experiment.
*   **The Diamond**: Located at the bottom, it represents the **Combined Result**.
    *   If the diamond is to the right of the center line: **Variant Wins**.
    *   If it touches the center line: **Inconclusive**.

#### **Funnel Plot ("The Triangle of Fairness")**
Checks for **Publication Bias** (e.g., hiding failed experiments).
*   **Interpretation**: You want the dots to look like a balanced, upside-down funnel. An asymmetrical plot suggests bias.

#### **Heterogeneity Statistics**
*   **I² (Disagreement Score)**:
    *   **0%**: Perfect agreement between studies.
    *   **>50%**: Moderate disagreement (High Heterogeneity).
*   **Tau-Squared (τ²)**: The actual variance between study effects.

### 4.4 Simulation & Statistical Models
*   **Random-Effects Model (Default)**: Assumes experiments are similar but not identical (e.g., different populations). Safer for most real-world data.
*   **Fixed-Effects Model**: Assumes experiments are identical. Use only when conditions are strictly controlled.
*   **Egger’s Test**: A mathematical "lie detector" for bias. Low p-values indicate potential missing data.
*   **Simpson’s Paradox Detector**: Warns you if the aggregate result conflicts with individual group results (a common statistical trap).

### 4.5 Simple View vs. Advanced View
*   **Simple View**: Provides a "Traffic Light" decision (Green/Red/Yellow) and a plain-English summary.
*   **Advanced View**: Shows full statistical dashboards, interactive graphs, and detailed metrics.

---

## 5. Troubleshooting & FAQ

*   **"Missing Columns" Error**: Ensure your CSV has headers that match the required format (e.g., `conversions`, `total`). The **MCP Server** attempts to map columns automatically, but distinct headers help.
*   **Application Crashing**: Check your GPU VRAM usage. If running out of memory, try switching to a smaller model in `config.yaml` or running in CPU-only mode.

---

**Unified RAG System** | Developed by Antigravity
