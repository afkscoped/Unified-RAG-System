# A/B Test Meta-Analysis Engine

## Overview
The **A/B Test Meta-Analysis Engine** is a powerful statistical tool integrated into the Unified RAG System. It allows users to aggregate results from multiple A/B tests (e.g., matching studies from different sources or time periods) to determine the true underlying effect, detect biases, and generate actionable recommendations.

## Features
- **Statistical Engine**: 
  - Fixed Effects & Random Effects models (DerSimonian-Laird)
  - Heterogeneity quantification (I², τ², Cochran's Q)
  - Automatic model selection
- **Bias Detection**:
  - Funnel plots
  - Egger's regression test
  - Trim-and-fill method for publication bias adjustment
  - Simpson's Paradox detection across subgroups
- **Sensitivity Analysis**:
  - Leave-one-out cross-validation
  - Influence diagnostics (Cook's distance, DFBETAS)
- **AI-Powered Insights**:
  - RAG-enhanced interpretation of statistical results
  - Context-aware recommendations
  - Benchmark comparison against industry standards

## Usage Guide

### 1. Data Ingestion
Upload your experiment data using the CSV/Excel uploader in the sidebar.
Required columns (auto-detected aliases available):
- `experiment_name`
- `control_visitors` / `control_total`
- `control_conversions`
- `treatment_visitors` / `treatment_total`
- `treatment_conversions`

### 2. Running Analysis
1. Select your preferred model (Auto, Fixed, or Random).
2. Set the confidence level (default 95%).
3. Click **Run Meta-Analysis**.

### 3. Interpreting Results
- **Forest Plot**: Visualizes the effect size and confidence interval for each study and the pooled result.
- **Funnel Plot**: Checks for publication bias. Asymmetry suggests missing studies.
- **I² (Heterogeneity)**:
  - < 25%: Low heterogeneity (Studies are consistent)
  - > 50%: High heterogeneity (Consider Random Effects or Subgroup Analysis)

### 4. API Reference
The engine exposes a REST API at `/meta-analysis`:
- `POST /meta-analysis/analyze`: Run full analysis
- `GET /meta-analysis/recommendations`: Get general guidelines

## Architecture
The system is built with a modular architecture:
- **`mcp_servers/`**: Data ingestion layer (CSV, Platform APIs)
- **`statistical/`**: Core statistical logic (Pure Python/NumPy)
- **`orchestration/`**: LangGraph-based workflow management
- **`rag/`**: Knowledge retrieval and LLM integration
- **`ui/`**: Streamlit interface

## Quick Start

### 1. Launch the Application
You can access the tool via the Unified RAG landing page or directly:

```bash
# Option A: Full User Interface
streamlit run src/ui/landing_page.py

# Option B: Meta-Analysis Integration Verification
python tests/meta_analysis/verify_integration.py
```

### 2. Start the API Server
To access the REST API endpoints:
```bash
uvicorn src.api.fastapi_app:app --reload
```
API Documentation will be available at `http://localhost:8000/docs`.

## Development
To run the full test suite:
```bash
python -m unittest discover tests/meta_analysis
```
