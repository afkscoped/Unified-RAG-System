"""
Meta-Analysis UI App

Main entry point for the Streamlit UI of the Meta-Analysis feature.
Enhanced with multi-file validation, synthetic data generation, and Kaggle import.
"""

import streamlit as st
import pandas as pd
import json
import hashlib
import plotly.io as pio
from datetime import datetime
from typing import Optional

from src.meta_analysis.orchestration.meta_analysis_agent import MetaAnalysisAgent
from src.meta_analysis.mcp_servers.csv_experiment_server import CSVExperimentMCP
from src.meta_analysis.integration.rag_bridge import MetaAnalysisRAGBridge
from src.meta_analysis.integration.llm_bridge import MetaAnalysisLLMBridge
from src.meta_analysis.utils.synthetic_generator import SyntheticABTestGenerator, DOMAIN_CONFIGS
from src.meta_analysis.utils.kaggle_importer import KaggleABTestImporter
from src.meta_analysis.ui.components.simple_view_renderer import render_simple_view

# --- ERROR FIX: Define MCP Badge locally to avoid UnboundLocalError ---
def render_mcp_badge(show: bool = True):
    """Render the Processed via MCP Server badge."""
    if show:
        st.markdown("""
        <div style="display: inline-block; 
                    background: linear-gradient(90deg, #2196F3 0%, #1976D2 100%);
                    color: white; 
                    padding: 6px 12px; 
                    border-radius: 16px; 
                    font-size: 14px;
                    font-weight: 500;
                    margin: 8px 0;">
            ✓ Processed via MCP Server
        </div>
        """, unsafe_allow_html=True)

def update_mcp_session_state(
    server_name: str = "CSVExperimentMCP",
    studies_count: int = 0,
    source_type: str = "none",
    validation_report: Optional[dict] = None
):
    """Update MCP session state."""
    st.session_state.mcp_active_server = server_name
    st.session_state.mcp_last_ingestion = datetime.now()
    st.session_state.mcp_studies_ingested = studies_count
    st.session_state.mcp_source_type = source_type
    st.session_state.mcp_validation_report = validation_report or {}
# ---------------------------------------------------------------------


def render_meta_analysis_page():
    """Render the main meta-analysis page."""
    
    st.title("🧪 A/B Test Meta-Analysis Engine")
    st.markdown("Aggregating insights from multiple experiments to find the true effect.")
    
    # Force reset on version upgrade (to apply hotfixes)
    CURRENT_VERSION = "1.0.2"
    if st.session_state.get("app_version") != CURRENT_VERSION:
        st.session_state.clear()
        st.session_state.app_version = CURRENT_VERSION
        st.rerun()
    
    # Initialize Agent (singleton-like pattern using session state)
    if 'agent' not in st.session_state:
        initialize_agent()
        
    agent = st.session_state.agent
    
    # Initialize session state for multi-file validation
    if 'file_hashes' not in st.session_state:
        st.session_state.file_hashes = {}
    if 'results_updated_message' not in st.session_state:
        st.session_state.results_updated_message = False
    if 'synthetic_data' not in st.session_state:
        st.session_state.synthetic_data = None
    if 'kaggle_data' not in st.session_state:
        st.session_state.kaggle_data = None
    if 'run_analysis_trigger' not in st.session_state:
        st.session_state.run_analysis_trigger = False
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Step 1: Data Source")
        
        # Mode selection
        data_mode = st.radio(
            "Select Data Source",
            ["📁 Upload Files", "🧪 Synthesize Data", "📚 Use Sample Datasets", "🌐 Import from Kaggle"],
            help="Choose how to provide experiment data"
        )
        
        st.divider()
        
        # Mode-specific UI
        if data_mode == "📁 Upload Files":
            render_upload_mode(agent)
        elif data_mode == "🧪 Synthesize Data":
            render_synthesize_mode(agent)
        elif data_mode == "📚 Use Sample Datasets":
            render_sample_datasets_mode(agent)
        else:
            render_kaggle_mode(agent)
        
        st.divider()
        
        st.header("Step 2: Analysis Settings")
        
        model_pref = st.selectbox(
            "Statistical Model",
            ["Auto (Recommended)", "Random Effects (DL)", "Random Effects (Hartung-Knapp)", "Fixed Effects"],
            help="Random effects assumes true effects vary across studies. Fixed effects assumes one true effect. Hartung-Knapp is more robust for small samples."
        )
        
        confidence = st.slider(
            "Confidence Level",
            min_value=0.90, max_value=0.99, value=0.95, step=0.01
        )
        
        st.divider()
        
        run_btn = st.button("Run Meta-Analysis", type="primary", use_container_width=True)
        reset_btn = st.button("Reset Analysis", use_container_width=True)
        
        if reset_btn:
            st.session_state.analysis_state = None
            st.session_state.results_updated_message = False
            st.rerun()
    
    # Show results updated message
    if st.session_state.results_updated_message:
        st.success("✅ Results updated based on the newly uploaded dataset.")
        st.session_state.results_updated_message = False

    # MCP Activity Panel & Data Flow Overview
    from src.meta_analysis.ui.components.mcp_activity_panel import (
        render_mcp_activity_panel, 
        render_data_flow_overview
    )
    
    # Determine active step based on session state
    active_step = None
    if st.session_state.get('mcp_source_type'):
        if st.session_state.get('analysis_state'):
            active_step = "engine"
        else:
            active_step = "mcp"
    
    render_mcp_activity_panel()
    render_data_flow_overview(active_step=active_step)

    # Main Content Area
    if 'analysis_state' not in st.session_state or st.session_state.analysis_state is None:
        render_landing_state(agent)
    else:
        render_results_state(st.session_state.analysis_state)
        
    # Trigger analysis (either from button or auto-trigger)
    if run_btn or st.session_state.run_analysis_trigger:
        # Reset trigger
        st.session_state.run_analysis_trigger = False
        
        with st.spinner("Running comprehensive meta-analysis..."):
            # Map UI settings to config
            model_map = {
                "Auto (Recommended)": "auto", 
                "Random Effects (DL)": "random", 
                "Random Effects (Hartung-Knapp)": "random_hk",
                "Fixed Effects": "fixed"
            }
            
            # Check if we have data first
            status = agent.csv_mcp.get_file_status()
            valid_count = sum(s['experiments_count'] for s in status)
            if valid_count < 2:
                st.error(f"Need at least 2 valid studies for meta-analysis. Found {valid_count}.")
                # Debug info
                st.warning(f"Looking in: {agent.csv_mcp.data_dir}")
                return

            # Run Agent
            state = agent.run(
                query=f"Run meta-analysis with {model_map[model_pref]} model",
                config={"model_preference": model_map[model_pref]}
            )
            
            st.session_state.analysis_state = state
            st.rerun()


def render_plots_tab(state):
    """Render visualizations."""
    st.header("Visualizations")
    
    viz_type = st.radio(
        "Select Plot", 
        ["Forest Plot", "Funnel Plot", "Bubble Plot", "Influence Plot", "Heterogeneity Chart"], 
        horizontal=True
    )
    
    if viz_type == "Forest Plot":
        if "forest_plot" in state.visualizations:
            st.plotly_chart(
                pio.from_json(state.visualizations["forest_plot"]), 
                use_container_width=True
            )
        else:
            st.info("Forest plot not available.")
            
    elif viz_type == "Funnel Plot":
        if "funnel_plot" in state.visualizations:
            st.plotly_chart(
                pio.from_json(state.visualizations["funnel_plot"]), 
                use_container_width=True
            )
            
            # Add interpretation helper
            with st.expander("How to read this plot"):
                st.markdown("""
                **Funnel Plots** are used to detect publication bias.
                - **Symmetry**: If the plot looks like an inverted funnel, bias is likely low.
                - **Asymmetry**: If points are missing from one bottom corner, small studies with negative/null results might be missing (publication bias).
                """)
        else:
            st.info("Funnel plot not available.")

    elif viz_type == "Bubble Plot":
        if "bubble_plot" in state.visualizations:
            st.plotly_chart(
                pio.from_json(state.visualizations["bubble_plot"]),
                use_container_width=True
            )
            st.caption("Visualizes the relationship between precision (1/SE) and effect size. Larger bubbles = more weight.")
        else:
            st.info("Bubble plot not available (requires > 2 studies).")

    elif viz_type == "Influence Plot":
        if "influence_plot" in state.visualizations:
            st.plotly_chart(
                pio.from_json(state.visualizations["influence_plot"]),
                use_container_width=True
            )
            st.caption("Identify influential studies and outliers. Points outside red dotted lines are potential outliers.")
        else:
            st.info("Influence plot not available.")

    elif viz_type == "Heterogeneity Chart":
        if "heterogeneity_chart" in state.visualizations:
             st.plotly_chart(
                pio.from_json(state.visualizations["heterogeneity_chart"]),
                use_container_width=True
             )
        else:
            st.info("Heterogeneity chart not available.")


def render_upload_mode(agent):
    """Render the upload files mode UI."""
    # Global import used
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Experiment Data (CSV/Excel)", 
        type=['csv', 'xlsx'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        new_files = process_uploads_with_validation(uploaded_files, agent)
        if new_files:
            st.success(f"Loaded {len(uploaded_files)} files")
            
            # Update MCP session state for uploaded files
            all_studies = agent.csv_mcp.get_experiments()
            validation_report = agent.csv_mcp.validate_data(all_studies)
            
            update_mcp_session_state(
                server_name="CSVExperimentMCP",
                studies_count=len(all_studies),
                source_type="upload",
                validation_report=validation_report.to_dict() if validation_report else {}
            )
    
    if st.button("🗑️ Clear All Data", help="Delete all uploaded files"):
        count = agent.csv_mcp.clear_all_files()
        if 'processed_files' in st.session_state:
            st.session_state.processed_files.clear()
        st.session_state.file_hashes.clear()
        st.session_state.analysis_state = None
        
        # Reset MCP session state
        update_mcp_session_state(
            server_name="CSVExperimentMCP",
            studies_count=0,
            source_type="none",
            validation_report={}
        )
        
        st.success(f"Deleted {count} files")
        st.rerun()


def render_synthesize_mode(agent):
    """Render the synthesize data mode UI."""
    from src.meta_analysis.synthetic.generators.ctgan_generator import CTGANGenerator
    from src.meta_analysis.synthetic.generators.copulagan_generator import CopulaGANGenerator
    from src.meta_analysis.synthetic.generators.base_generator import SyntheticConfig
    from src.meta_analysis.synthetic.domains.domain_constraints import DOMAIN_CONFIGS
    from src.meta_analysis.synthetic.generators.base_generator import SyntheticConfig
    from src.meta_analysis.synthetic.domains.domain_constraints import DOMAIN_CONFIGS
    from src.meta_analysis.synthetic.utils.schema_mapper import SchemaMapper
    # Global import used instead to avoid UnboundLocalError

    st.subheader("🧪 Generate Synthetic Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        domain_key = st.selectbox(
            "Domain",
            list(DOMAIN_CONFIGS.keys()),
            format_func=lambda x: DOMAIN_CONFIGS[x].display_name,
            help="Domain affects realistic parameter ranges"
        )
        
        n_experiments = st.slider(
            "Number of Experiments",
            min_value=5, max_value=200, value=50,
            help="Number of synthetic A/B tests to generate"
        )
        
        model_type = st.radio(
            "Generation Model",
            ["CTGAN", "CopulaGAN"],
            help="CTGAN: Better for complex distributions. CopulaGAN: Preserves correlations."
        )

    with col2:
        # Domain config to set defaults
        dom_conf = DOMAIN_CONFIGS[domain_key]
        
        st.caption("Generation Parameters")
        
        # Effect size slider
        es_min, es_max = dom_conf.effect_size_range
        effect_range = st.slider(
            "Effect Size Range (Relative Lift)", 
            min_value=0.01, max_value=0.50, 
            value=(max(0.01, float(es_min)), min(0.50, float(es_max))),
            step=0.01
        )
        
        heterogeneity = st.select_slider(
            "Heterogeneity Level", 
            options=["low", "medium", "high"], 
            value="medium",
            help="Variance in effect sizes across experiments"
        )
        
        include_segments = st.checkbox(
            "Include Segments", 
            value=True,
            help="Generate breakdowns by device, region, etc."
        )

    # Training & Generation
    if st.button("🚀 Generate Synthetic Data", type="primary", use_container_width=True):
        
        with st.status(f"Generating {n_experiments} {domain_key} experiments...", expanded=True) as status:
            try:
                # 1. Initialize Generator
                st.write(f"Initializing {model_type} generator...")
                if model_type == "CTGAN":
                    generator = CTGANGenerator(domain_key)
                else:
                    generator = CopulaGANGenerator(domain_key)
                
                # 2. Check if trained, otherwise train
                train_needed = not generator.is_trained
                if train_needed:
                    st.write("Loading seed data and training model (this may take a moment)...")
                    # In a real app we'd load specific seed file. For now using built-in logic via fit.
                    # We need to load the seed dataset we created
                    seed_path = f"src/meta_analysis/synthetic/data/seed_datasets/{domain_key}_tests.csv"
                    try:
                        seed_df = pd.read_csv(seed_path)
                        generator.fit(seed_df, epochs=200, verbose=False) # Reduced epochs for UI responsiveness
                    except FileNotFoundError:
                        st.warning(f"Seed file not found at {seed_path}. Using fallback generation.")
                        # Hack to force fallback mode if seed missing
                        generator._is_trained = True 
                
                # 3. Generate
                st.write("Generating experiments...")
                config = SyntheticConfig(
                    domain=domain_key,
                    num_experiments=n_experiments,
                    model_type=model_type.lower(),
                    effect_size_range=effect_range,
                    heterogeneity_level=heterogeneity,
                    include_segments=include_segments
                )
                
                synthetic_df = generator.generate(config)
                
                # 4. Map to standardized format
                st.write("Standardizing data format...")
                standardized_df = SchemaMapper.from_domain_format(synthetic_df, domain_key)
                
                # 5. Validate with generator
                st.write("Validating data quality...")
                if generator._seed_data is not None:
                    report = generator.validate(synthetic_df, generator._seed_data)
                else:
                    report = None
                
                # 6. Send to MCP Server (using agent's MCP server to ensure same data directory)
                st.write("📡 Sending dataset to MCP server...")
                # Global import used instead
                import io
                
                # Convert to CSV and upload to AGENT's MCP server (not a new temp one)
                csv_buffer = io.StringIO()
                standardized_df.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue().encode('utf-8')
                
                filename = f"synth_{domain_key}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                agent.csv_mcp.upload_file(csv_bytes, filename)
                
                # FORCE CLEAR CACHE to ensure new file is seen
                agent.csv_mcp.experiments_cache.clear()
                agent.csv_mcp._file_hashes.clear()
                
                # Get all studies from agent's MCP
                all_studies = agent.csv_mcp.get_experiments()
                mcp_validation = agent.csv_mcp.validate_data(all_studies)
                
                # Update MCP session state (Global Panel)
                update_mcp_session_state(
                    server_name="CSVExperimentMCP",
                    studies_count=len(all_studies),
                    source_type="synthetic",
                    validation_report=mcp_validation.to_dict() if mcp_validation else {}
                )
                
                # Save state for local display
                st.session_state.synthetic_data = synthetic_df
                st.session_state.mcp_studies = all_studies
                st.session_state.mcp_validation = mcp_validation
                st.session_state.analysis_state = None
                st.session_state.results_updated_message = True
                st.success(f"Generated {len(synthetic_df)} experiments")
                
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # --- PERSISTENT RESULTS DISPLAY ---
    # content is shown if data exists in session state, regardless of button click
    if st.session_state.get('synthetic_data') is not None:
        synthetic_df = st.session_state.synthetic_data
        all_studies = st.session_state.get('mcp_studies', [])
        mcp_validation = st.session_state.get('mcp_validation')
        
        st.divider()
        st.subheader("📊 Generation Results")
        
        # MCP Badge
        render_mcp_badge(show=True)
        
        # MCP Validation Summary
        mcp_col1, mcp_col2 = st.columns(2)
        mcp_col1.metric("Studies Ingested (Total)", len(all_studies))
        mcp_col2.metric("Valid Studies", f"{mcp_validation.valid_records}/{mcp_validation.total_records}" if mcp_validation else "N/A")
        
        # Preview
        with st.expander("Preview Data & Download", expanded=True):
            st.dataframe(synthetic_df.head(10), use_container_width=True)
            
            # Download button
            csv_download = synthetic_df.to_csv(index=False).encode('utf-8')
            col_d1, col_d2 = st.columns([1, 2])
            with col_d1:
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_download,
                    file_name=f"synthetic_data_{datetime.now().strftime('%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_synthetic_main"
                )
        
        # Analysis Actions
        st.info(f"Ready to analyze {len(all_studies)} studies from repository.")
        
        auto_analyze = st.checkbox("🔄 Auto-run meta-analysis", value=True, key="auto_run_check")
        
        if st.button("▶️ Run Meta-Analysis Now", type="primary", use_container_width=True):
            st.session_state.run_analysis_trigger = True
            st.rerun()
        
        if st.button("🗑️ Clear Generated Data"):
            st.session_state.synthetic_data = None
            st.session_state.mcp_studies = None
            st.rerun()

            st.rerun()

    # DEBUG: List actual files in data directory
    with st.expander("🕵️ Debug: File System Status"):
        import os
        cwd = os.getcwd()
        abs_data_dir = os.path.join(cwd, "data", "experiments")
        
        st.write(f"**Current Working Dir:** `{cwd}`")
        st.write(f"**Target Data Dir:** `{abs_data_dir}`")
        
        # Check Agent Config
        if 'agent' in st.session_state:
            agent_dir = st.session_state.agent.csv_mcp.data_dir
            st.write(f"**Agent MCP Data Dir:** `{agent_dir}`")
            if str(agent_dir) != str(abs_data_dir):
                st.error("⚠️ DATA DIR MISMATCH! Agent is looking in the wrong place.")
        
        if os.path.exists(abs_data_dir):
            files = os.listdir(abs_data_dir)
            st.write(f"**Files Found ({len(files)}):**")
            for f in files:
                st.code(f)
                # Show size
                size = os.path.getsize(os.path.join(abs_data_dir, f))
                st.write(f"- Size: {size} bytes")
        else:
            st.error(f"Directory `{abs_data_dir}` does not exist!")



def render_sample_datasets_mode(agent):
    """Render the sample datasets mode UI."""
    st.subheader("📚 Sample Datasets Library")
    
    import os
    import json
    import pandas as pd
    
    # Load catalog
    base_dir = os.path.dirname(os.path.abspath(__file__))
    catalog_path = os.path.join(base_dir, "../sample_datasets/dataset_catalog.json")
    
    if not os.path.exists(catalog_path):
        st.error("Sample dataset catalog not found!")
        return

    with open(catalog_path, 'r') as f:
        catalog = json.load(f)

    # Filter domains
    domains = sorted(list(set(d['domain'] for d in catalog)))
    selected_domain = st.selectbox("Filter by Domain", ["All"] + domains)
    
    filtered_catalog = catalog if selected_domain == "All" else [d for d in catalog if d['domain'] == selected_domain]
    
    # Select Dataset
    dataset_opts = {d['dataset_id']: f"{d['name']} ({d['row_count']} rows)" for d in filtered_catalog}
    selected_id = st.selectbox("Select Dataset", list(dataset_opts.keys()), format_func=lambda x: dataset_opts[x])
    
    # Get details
    dataset_info = next(d for d in filtered_catalog if d['dataset_id'] == selected_id)
    st.info(dataset_info['description'])
    
    # Preview
    file_path = os.path.join(base_dir, "../sample_datasets", dataset_info['file_path'])
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        st.write("Preview:")
        st.dataframe(df.head(5), use_container_width=True)
        
        # Load Action
        if st.button("🚀 Load & Analyze This Dataset", type="primary"):
            import io
            with st.spinner("Ingesting into MCP..."):
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                agent.csv_mcp.upload_file(csv_bytes, dataset_info['file_path'])
                
                # Clear cache & Validate
                agent.csv_mcp.experiments_cache.clear()
                agent.csv_mcp._file_hashes.clear()
                
                all_studies = agent.csv_mcp.get_experiments()
                validation_report = agent.csv_mcp.validate_data(all_studies)
                
                update_mcp_session_state(
                    server_name="CSVExperimentMCP",
                    studies_count=len(all_studies),
                    source_type="sample_dataset",
                    validation_report=validation_report.to_dict() if validation_report else {}
                )
                
                st.session_state.run_analysis_trigger = True
                st.success(f"Loaded {len(df)} experiments from {dataset_info['name']}")
                st.rerun()

    # Show Badge if loaded
    if st.session_state.get('mcp_source_type') == "sample_dataset":
        st.divider()
        render_mcp_badge(show=True)
        
        # MCP Validation Summary
        mcp_validation = st.session_state.get('mcp_validation_report', {})
        valid_count = mcp_validation.get("valid_records", 0)
        total_count = mcp_validation.get("total_records", 0)
        
        mcp_col1, mcp_col2 = st.columns(2)
        mcp_col1.metric("Studies Ingested", total_count)
        mcp_col2.metric("Valid Studies", f"{valid_count}/{total_count}")


def render_kaggle_mode(agent):
    """Render the Kaggle import mode UI."""
    # Global import used
    
    st.subheader("🌐 Import from Kaggle")
    
    # Quick Import Selection
    samples = KaggleABTestImporter.get_sample_datasets()
    quick_opts = {s['id']: f"{s['name']} ({s['description']})" for s in samples}
    
    col_mode, col_input = st.columns([1, 2])
    
    with col_mode:
        import_type = st.radio("Source", ["⚡ Quick Import", "🔍 Custom ID"], label_visibility="collapsed")
    
    dataset_id = ""
    if import_type == "⚡ Quick Import":
        with col_input:
            dataset_id = st.selectbox(
                "Select a Curated Dataset", 
                list(quick_opts.keys()),
                format_func=lambda x: quick_opts.get(x, x)
            )
            st.caption("🚀 These datasets are cached and load instantly.")
    else:
        with col_input:
            dataset_id = st.text_input(
                "Enter Kaggle Dataset ID",
                placeholder="e.g., zhangluyuan/ab-testing"
            )
            st.caption("ℹ️ First time downloads may take a moment.")

    # Fast Mode Toggle
    st.write("")
    fast_mode = st.toggle("🚀 Fast Mode (Download single CSV only)", value=True, help="Skip large files and archives. Recommended.")

    # Metadata & Warnings
    if dataset_id and import_type == "🔍 Custom ID":
        importer = KaggleABTestImporter()
        # Only fetch metadata if not a known quick/cached one (optional optimization)
        with st.status("Checking dataset...", expanded=False) as status:
            meta = importer.fetch_metadata(dataset_id)
            if "error" not in meta:
                status.update(label="Dataset Info", state="complete")
                
                c1, c2 = st.columns(2)
                c1.metric("Size", f"{meta['total_size_mb']} MB")
                c2.metric("Files", f"{meta['csv_count']} CSVs")
                
                if meta['total_size_mb'] > 50:
                    st.warning(f"⚠️ Large dataset (>50MB). Fast Mode will skip download of full archive.")
            else:
                 status.update(label="Could not fetch metadata", state="error")

    if st.button("Import Dataset", type="primary", use_container_width=True, disabled=not dataset_id):
        with st.status("Processing Kaggle Dataset...", expanded=True) as status:
            st.write("Initializing download...")
            importer = KaggleABTestImporter()
            
            # Pass fast_mode
            success, message, df = importer.import_dataset(dataset_id, fast_mode=fast_mode)
            
            if success and df is not None:
                st.write(f"Loaded {len(df)} rows. Extracting columns...")
                st.session_state.kaggle_data = df
                
                # Save to MCP
                st.write("Ingesting into MCP System...")
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', newline='') as f:
                    df.to_csv(f, index=False)
                    temp_path = f.name
                
                with open(temp_path, 'rb') as f:
                    # Clean filename
                    filename = f"kaggle_{dataset_id.replace('/', '_')}.csv"
                    agent.csv_mcp.upload_file(f.read(), filename)
                
                status.update(label="Import Complete!", state="complete", expanded=False)
                
                # Update MCP session state for Kaggle import
                all_studies = agent.csv_mcp.get_experiments()
                validation_report = agent.csv_mcp.validate_data(all_studies)
                
                update_mcp_session_state(
                    server_name="CSVExperimentMCP",
                    studies_count=len(all_studies),
                    source_type="kaggle",
                    validation_report=validation_report.to_dict() if validation_report else {}
                )
                
                st.session_state.analysis_state = None
                st.success(f"Successfully imported {dataset_id}")
                st.rerun()
            else:
                status.update(label="Import Failed", state="error", expanded=True)
                st.error(f"Error: {message}")
    
    # Show preview if data exists
    if st.session_state.kaggle_data is not None:
        with st.expander("Preview Imported Data", expanded=False):
            st.dataframe(st.session_state.kaggle_data.head(10), use_container_width=True)


def process_uploads_with_validation(files, agent) -> bool:
    """Process uploaded files with hash-based change detection."""
    import tempfile
    import os
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    
    new_files_detected = False
    
    for file in files:
        # Calculate file hash
        file.seek(0)
        content = file.read()
        file_hash = hashlib.md5(content).hexdigest()
        
        # Check if this is a new or changed file
        old_hash = st.session_state.file_hashes.get(file.name)
        if old_hash != file_hash:
            new_files_detected = True
            st.session_state.file_hashes[file.name] = file_hash
            
            # Clear previous analysis when new data arrives
            if st.session_state.get('analysis_state'):
                st.session_state.analysis_state = None
                st.session_state.results_updated_message = True
            
            # Upload to MCP
            try:
                agent.csv_mcp.upload_file(content, file.name)
                st.session_state.processed_files.add(file.name)
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
    
    return new_files_detected


def initialize_agent():
    """Initialize the meta-analysis agent."""
    import os
    # Use ABSOLUTE CURRENT WORKING DIRECTORY path
    cwd = os.getcwd()
    abs_data_dir = os.path.join(cwd, "data", "experiments")
    os.makedirs(abs_data_dir, exist_ok=True)
    
    print(f"Initializing Agent with Data Dir: {abs_data_dir}")
    csv_mcp = CSVExperimentMCP(data_dir=abs_data_dir)
    
    # Initialize Core Systems if available
    rag_system = None
    llm_router = None
    
    try:
        from src.core.rag_system import UnifiedRAGSystem
        rag_system = UnifiedRAGSystem()
    except Exception as e:
        print(f"Failed to initialize RAG System: {e}")
        
    try:
        from src.llm.llm_router import LLMRouter
        llm_router = LLMRouter()
    except Exception as e:
        print(f"Failed to initialize LLM Router: {e}")
    
    rag_bridge = MetaAnalysisRAGBridge(rag_system=rag_system)
    llm_bridge = MetaAnalysisLLMBridge(llm_router=llm_router)
    
    # Initialize bridges
    rag_bridge.initialize()
    llm_bridge.initialize()
    
    st.session_state.agent = MetaAnalysisAgent(
        csv_mcp=csv_mcp,
        rag_bridge=rag_bridge,
        llm_bridge=llm_bridge
    )


def process_uploads(files, agent):
    """Process uploaded files."""
    import tempfile
    import os
    
    # Check if we've already processed these specific files to avoid reloading
    # (Simple check could be refined)
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
        
    for file in files:
        if file.name not in st.session_state.processed_files:
            # Save to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            
            # Use MCP to ingest
            try:
                # Read file content
                file.seek(0)
                content = file.read()
                agent.csv_mcp.upload_file(content, file.name)
                st.session_state.processed_files.add(file.name)
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)


def render_landing_state(agent):
    """Render the initial landing state."""
    st.info("👈 Upload your experiment data in the sidebar to get started.")
    
    # Show available studies if any
    # Show available studies and file status
    try:
        st.subheader("Data Status")
        status_list = agent.csv_mcp.get_file_status()
        
        valid_studies = []
        for s in status_list:
            if s["status"] == "Error":
                 st.error(f"❌ **{s['file']}**: {s['error']}")
            elif s["status"] == "Warning":
                 st.warning(f"⚠️ **{s['file']}**: {s['error']}")
            else:
                 st.success(f"✅ **{s['file']}**: {s['experiments_count']} valid experiments")
        
        # Reload valid studies
        studies = agent.csv_mcp.get_experiments()
        
        if studies:
            st.divider()
            st.subheader(f"Available Experiments ({len(studies)})")
            
            # Convert to dataframe for display
            
            # Convert to dataframe for display
            df = pd.DataFrame([
                {
                    "Study ID": s.study_id,
                    "Name": s.study_name,
                    "Effect Size": f"{s.effect_size:.3f}",
                    "Sample Size": s.total_sample_size,
                    "Platform": s.platform
                }
                for s in studies
            ])
            st.dataframe(df, use_container_width=True)
        else:
            st.markdown("""
            ### What is this?
            
            This tool performs **statistical meta-analysis** on A/B test results to:
            - **Aggregate** results from multiple experiments
            - **Detect** publication bias and outliers
            - **Provide** AI-powered recommendations
            
            ### Data Format
            Upload simple CSV or Excel files with columns like:
            - `experiment_name`
            - `control_visitors`, `control_conversions`
            - `variant_visitors`, `variant_conversions`
            """)
    except Exception as e:
        st.error(f"Error loading studies: {e}")


def render_results_state(state):
    """Render the analysis results."""
    
    if state.current_stage.name == "ERROR":
        st.error(f"Analysis failed: {state.errors[0] if state.errors else 'Unknown error'}")
        return

    # View mode toggle
    view_mode = st.radio(
        "View Mode",
        ["📊 Technical View", "😊 Simple View"],
        horizontal=True,
        help="Simple View provides plain-language explanations for non-technical users"
    )
    
    if view_mode == "😊 Simple View":
        render_simple_view(state)
        return

    # Tabs for different result sections (Technical View)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Plots", 
        "🧠 AI Insights", 
        "⚖️ Bias & Sensitivity",
        "📑 Report"
    ])
    
    # --- Tab 1: Overview ---
    with tab1:
        render_overview_tab(state)
        
    # --- Tab 2: Plots ---
    with tab2:
        render_plots_tab(state)
        
    # --- Tab 3: Insights ---
    with tab3:
        render_insights_tab(state)
        
    # --- Tab 4: Bias & Sensitivity ---
    with tab4:
        render_bias_tab(state)
        
    # --- Tab 5: Report ---
    with tab5:
        render_report_tab(state)


def render_overview_tab(state):
    """Render overview statistics."""
    st.header("Meta-Analysis Results")
    
    if not state.meta_result:
        st.warning("No results available.")
        return

    # Top level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Pooled Effect", f"{state.meta_result.pooled_effect:.3f}")
    with col2:
        p_val = state.meta_result.p_value
        p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"
        st.metric("P-value", p_str, delta="Significant" if p_val < 0.05 else "Not Sig", delta_color="inverse")
    with col3:
        i2 = state.heterogeneity_stats.get("I2", 0)
        st.metric("Heterogeneity (I²)", f"{i2:.1f}%", help="Variance due to heterogeneity")
    with col4:
        st.metric("Studies Included", state.meta_result.n_studies)
        
    # Confidence Interval
    ci = state.meta_result.confidence_interval
    st.info(f"**95% Confidence Interval:** {ci[0]:.3f} to {ci[1]:.3f}")
    
    # Warnings
    if state.warnings:
        with st.expander("⚠️ Analysis Warnings", expanded=True):
            for w in state.warnings:
                st.write(f"- {w}")
    
    # Model Comparison Panel
    with st.expander("⚖️ Model Comparison (Fixed vs Random Effects)"):
        # We need to construct this from the state or re-calculate. 
        # Since state doesn't have it explicitly stored in this format, let's just use what we have or placeholder.
        # Ideally, the agent run would populate this.
        # For now, explain the current model choice.
        
        current_model = state.meta_result.model_type
        other_model = "Fixed Effects" if "random" in current_model else "Random Effects"
        
        st.write(f"**Selected Model:** {current_model.replace('_', ' ').title()}")
        if state.heterogeneity_stats.get("I2", 0) > 25 and "fixed" in current_model:
            st.warning("⚠️ High heterogeneity detected but Fixed Effects model was used. Random Effects is recommended.")
        elif state.heterogeneity_stats.get("I2", 0) < 25 and "random" in current_model:
            st.info("ℹ️ Low heterogeneity detected. Fixed Effects model might be sufficient.")
        
        st.markdown("""
        - **Fixed Effects**: Assumes there is one true effect size shared by all studies. best for exact replications.
        - **Random Effects**: Assumes true effect sizes vary across studies. Best when studies differ in populations or methods.
        - **Hartung-Knapp**: An adjustment to Random Effects that is more robust for small number of studies (< 10).
        """)


def render_plots_tab(state):
    """Render visualizations."""
    st.header("Visualizations")
    
    viz_type = st.radio("Select Plot", ["Forest Plot", "Funnel Plot"], horizontal=True)
    
    if viz_type == "Forest Plot":
        if "forest_plot" in state.visualizations:
            st.plotly_chart(
                pio.from_json(state.visualizations["forest_plot"]), 
                use_container_width=True
            )
        else:
            st.info("Forest plot not available.")
            
    elif viz_type == "Funnel Plot":
        if "funnel_plot" in state.visualizations:
            st.plotly_chart(
                pio.from_json(state.visualizations["funnel_plot"]), 
                use_container_width=True
            )
            
            # Add interpretation helper
            with st.expander("How to read this plot"):
                st.markdown("""
                **Funnel Plots** are used to detect publication bias.
                - **Symmetry**: If the plot looks like an inverted funnel, bias is likely low.
                - **Asymmetry**: If points are missing from one bottom corner, small studies with negative/null results might be missing (publication bias).
                """)
        else:
            st.info("Funnel plot not available.")


def render_insights_tab(state):
    """Render AI recommendations and insights."""
    st.header("🧠 AI Insights & Recommendations")
    
    if state.recommendations:
        for rec in state.recommendations:
            priority_color = {
                "high": "red",
                "medium": "orange",
                "low": "blue"
            }.get(rec.get("priority", "medium").lower(), "blue")
            
            with st.container(border=True):
                st.markdown(f"### :{priority_color}[{rec.get('title')}]")
                st.write(rec.get("message"))
                st.markdown(f"**Action:** {rec.get('action')}")
                st.caption(f"Rationale: {rec.get('rationale')}")
    else:
        st.info("No specific recommendations generated.")
        
    if state.rag_context:
        st.divider()
        st.subheader("Statistical Interpretation")
        st.markdown(state.rag_context)


def render_bias_tab(state):
    """Render bias and sensitivity details."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Publication Bias")
        if state.publication_bias_result:
            pbr = state.publication_bias_result
            
            st.write(f"**Detection Status:** {pbr.get('bias_detected')}")
            st.write(f"**Severity:** {pbr.get('bias_severity', 'Unknown').title()}")
            st.write(f"**Egger's Test p-value:** {pbr.get('eggers_test', {}).get('p_value', 1.0):.3f}")
            
            k0 = pbr.get('trim_and_fill', {}).get('k0', 0)
            if k0 > 0:
                st.warning(f"Trim-and-fill suggests {k0} missing studies.")
                orig = pbr.get('trim_and_fill', {}).get('original_effect', 0)
                adj = pbr.get('trim_and_fill', {}).get('adjusted_effect', 0)
                st.metric("Effect Size Adjustment", f"{adj:.3f}", f"{adj-orig:.3f}")
            else:
                st.success("No missing studies imputed by Trim-and-fill.")
        else:
            st.info("Publication bias analysis not available.")
            
    with col2:
        st.subheader("Sensitivity Analysis")
        if state.sensitivity_result:
            sen = state.sensitivity_result
            
            is_robust = sen.get('robust', False)
            if is_robust:
                st.success("✅ Results are robust to leave-one-out analysis.")
            else:
                st.error("⚠️ Results are sensitive to individual studies.")
                
            influential = sen.get('influential_studies', [])
            if influential:
                st.write("**Influential Studies:**")
                for study in influential:
                    st.code(study)
            else:
                st.write("No single study dominates the results.")
        else:
            st.info("Sensitivity analysis not available.")


def render_report_tab(state):
    """Render the final report."""
    st.header("Final Report")
    
    if state.final_report:
        st.markdown(state.final_report)
        
        st.download_button(
            "Download Report (Markdown)",
            state.final_report,
            file_name=f"meta_analysis_report_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )
    else:
        st.info("Report generation pending.")
