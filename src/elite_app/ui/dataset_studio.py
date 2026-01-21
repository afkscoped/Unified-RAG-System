"""
Dataset Studio - Upload, validate, and manage training datasets
With Groq API integration for instant dataset generation
"""

import streamlit as st
import json
from pathlib import Path
import pandas as pd

import os
try:
    from src.elite_app.groq_integration import GroqDatasetGenerator, GROQ_AVAILABLE
except ImportError:
    GROQ_AVAILABLE = False


def render_dataset_studio():
    """Render dataset management interface"""
    st.title("Dataset Studio")
    st.markdown("Upload, validate, and prepare training datasets for fine-tuning")
    
    # Create tabs - add Groq tab
    tab1, tab2, tab3, tab4 = st.tabs(["⚡ Groq API", "Upload Dataset", "Auto-Generate", "Dataset Info"])
    
    # Tab 1: Groq API (NEW - Priority tab)
    with tab1:
        render_groq_tab()
    
    # Tab 2: Upload Dataset
    with tab2:
        render_upload_tab()
    
    # Tab 3: Auto-Generate
    with tab3:
        render_autogen_tab()
    
    # Tab 4: Dataset Info
    with tab4:
        render_info_tab()


def render_groq_tab():
    """Render Groq API dataset generation interface"""
    st.subheader("⚡ Instant Dataset Generation with Groq")
    
    st.markdown("""
    **Generate high-quality training data in seconds!**
    - No local model loading required
    - Ultra-fast: 2-5 samples per second
    - Free tier: 30 requests/minute
    """)
    
    if not GROQ_AVAILABLE:
        st.error("Groq library not installed. Run: `pip install groq`")
        st.code("pip install groq", language="bash")
        return
    
    # API Key section
    st.markdown("---")

    env_key = None
    try:
        env_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    except Exception:
        env_key = os.environ.get("GROQ_API_KEY")

    use_env_key = st.checkbox(
        "Use GROQ_API_KEY from environment/.env",
        value=True if env_key else False
    )

    if use_env_key:
        api_key = env_key
        if api_key:
            st.success("Using GROQ_API_KEY from Streamlit secrets")
        else:
            api_key = st.text_input(
                "Groq API Key",
                type="password",
                help="Set GROQ_API_KEY in Streamlit secrets or paste your key here"
            )
    else:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Get free key at console.groq.com"
        )
    
    # Test API button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Test API"):
            try:
                generator = GroqDatasetGenerator(api_key)
                if generator.test_api_key():
                    st.success("API key valid!")
                else:
                    st.error("Invalid API key")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # Generation options
    st.subheader("Generation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Template selection
        templates = [
            "General QA",
            "Coding Assistant", 
            "Customer Support",
            "Educational Content",
            "Creative Writing",
            "Business & Marketing",
            "Technical Documentation"
        ]
        
        template = st.selectbox(
            "Template",
            templates,
            help="Pre-configured generation template"
        )
        
        num_samples = st.slider("Number of samples", 10, 200, 50)
    
    with col2:
        # Model selection
        models = {
            "llama-3.1-8b (Fastest)": "llama-3.1-8b",
            "llama-3.1-70b (Best Quality)": "llama-3.1-70b",
            "llama-3.3-70b (Latest)": "llama-3.3-70b",
            "mixtral-8x7b (Balanced)": "mixtral-8x7b",
            "gemma2-9b (Efficient)": "gemma2-9b"
        }
        
        model_display = st.selectbox("Model", list(models.keys()))
        model = models[model_display]
        
        st.info(f"Est. time: ~{num_samples // 2} seconds")
    
    # Custom prompt option
    with st.expander("Custom Generation Prompt (Optional)"):
        custom_prompt = st.text_area(
            "Custom instructions for generation",
            placeholder="E.g., Generate Q&A pairs about machine learning concepts...",
            height=100
        )
    
    # Generate button
    st.markdown("---")
    
    if st.button("⚡ Generate Dataset", type="primary", use_container_width=True):
        try:
            generator = GroqDatasetGenerator(api_key)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total):
                progress_bar.progress(current / total)
                status_text.text(f"Generating sample {current}/{total}...")
            
            with st.spinner("Generating dataset with Groq..."):
                if custom_prompt:
                    dataset = generator.generate_custom_samples(
                        custom_prompt=custom_prompt,
                        num_samples=num_samples,
                        model=model,
                        progress_callback=update_progress
                    )
                else:
                    dataset = generator.generate_from_template(
                        template_name=template,
                        num_samples=num_samples,
                        model=model,
                        progress_callback=update_progress
                    )
            
            progress_bar.progress(1.0)
            status_text.text("Complete!")
            
            if dataset:
                # Load into processor
                processor = st.session_state.dataset_processor
                loaded_dataset = processor.load_from_dict_list(dataset)
                st.session_state.training_dataset = loaded_dataset
                
                st.success(f"Generated {len(dataset)} samples!")
                
                # Preview
                st.subheader("Preview Generated Samples")
                for i, sample in enumerate(dataset[:3]):
                    with st.expander(f"Sample {i+1}"):
                        st.markdown("**Instruction:**")
                        st.text(sample.get("instruction", "")[:500])
                        st.markdown("**Output:**")
                        st.text(sample.get("output", "")[:500])
                
                # Quick save option
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    save_name = st.text_input("Save as", f"groq_{template.lower().replace(' ', '_')}")
                with col2:
                    if st.button("Save Dataset"):
                        save_path = Path(f"./data/training_datasets/{save_name}.jsonl")
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(save_path, 'w') as f:
                            for item in dataset:
                                json.dump(item, f)
                                f.write('\n')
                        st.success(f"Saved to {save_path}")
            else:
                st.warning("No samples generated. Try again.")
                
        except Exception as e:
            st.error(f"Generation error: {str(e)}")
            st.info("Check your API key and try again")


def render_upload_tab():
    """Render upload dataset interface"""
    st.subheader("Upload Training Dataset")
    st.markdown("""
    **Supported Formats:**
    - **JSONL** (recommended): Each line is `{"instruction": "...", "output": "..."}`
    - **CSV**: Must have `instruction` and `output` columns
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a dataset file",
        type=['jsonl', 'csv'],
        help="Upload JSONL or CSV file with instruction-output pairs"
    )
    
    if uploaded_file is not None:
        try:
            processor = st.session_state.dataset_processor
            
            # Save uploaded file temporarily
            file_ext = uploaded_file.name.split('.')[-1].lower()
            temp_path = Path(f"./data/training_datasets/{uploaded_file.name}")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # Load based on file type
            if file_ext == 'jsonl':
                dataset = processor.load_from_jsonl(str(temp_path))
            elif file_ext == 'csv':
                dataset = processor.load_from_csv(str(temp_path))
            else:
                st.error(f"Unsupported file type: {file_ext}")
                return
            
            # Store in session state
            st.session_state.training_dataset = dataset
            
            # Show success
            st.success(f"Dataset loaded: {len(dataset)} samples")
            
            # Show statistics
            stats = processor.validate_dataset()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Samples", stats.total_samples)
            col2.metric("Avg Instruction Length", f"{stats.avg_instruction_length} words")
            col3.metric("Avg Output Length", f"{stats.avg_output_length} words")
            
            if stats.issues:
                st.warning(f"Issues found: {len(stats.issues)}")
                with st.expander("View Issues"):
                    for issue in stats.issues[:10]:
                        st.text(f"- {issue}")
            
            # Preview samples
            st.subheader("Sample Preview")
            samples = processor.get_samples(5)
            
            for i, sample in enumerate(samples):
                with st.expander(f"Sample {i+1}"):
                    st.markdown("**Instruction:**")
                    st.text(sample.get("instruction", "")[:500])
                    st.markdown("**Output:**")
                    st.text(sample.get("output", "")[:500])
        
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    
    # Manual entry option
    st.markdown("---")
    st.subheader("Manual Entry")
    
    with st.expander("Add samples manually"):
        instruction = st.text_area("Instruction", height=100, key="manual_instruction")
        output = st.text_area("Output", height=150, key="manual_output")
        
        if st.button("Add Sample"):
            if instruction and output:
                if 'manual_samples' not in st.session_state:
                    st.session_state.manual_samples = []
                
                st.session_state.manual_samples.append({
                    "instruction": instruction,
                    "output": output
                })
                st.success(f"Sample added! Total: {len(st.session_state.manual_samples)}")
            else:
                st.warning("Both instruction and output are required")
        
        if st.session_state.get('manual_samples'):
            st.write(f"Manual samples: {len(st.session_state.manual_samples)}")
            
            if st.button("Load Manual Samples as Dataset"):
                processor = st.session_state.dataset_processor
                dataset = processor.load_from_dict_list(st.session_state.manual_samples)
                st.session_state.training_dataset = dataset
                st.success(f"Loaded {len(dataset)} manual samples")
                st.rerun()


def render_autogen_tab():
    """Render auto-generation interface"""
    st.subheader("Auto-Generate Dataset")
    st.markdown("""
    Generate training data automatically from text documents.
    Upload plain text files and the system will create instruction-output pairs.
    """)
    
    uploaded_files = st.file_uploader(
        "Upload text files",
        type=['txt', 'md'],
        accept_multiple_files=True,
        help="Upload text files to generate training data from"
    )
    
    if uploaded_files:
        all_text = ""
        for f in uploaded_files:
            content = f.read().decode('utf-8')
            all_text += content + "\n\n"
        
        st.info(f"Loaded {len(uploaded_files)} files, {len(all_text)} characters total")
        
        # Generation options
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input("Chunk size (characters)", 200, 2000, 500)
        with col2:
            max_samples = st.number_input("Max samples", 10, 500, 100)
        
        generation_type = st.selectbox(
            "Generation Type",
            ["Q&A from chunks", "Summarization", "Completion"]
        )
        
        if st.button("Generate Dataset", type="primary"):
            with st.spinner("Generating dataset..."):
                # Split into chunks
                chunks = []
                for i in range(0, len(all_text), chunk_size):
                    chunk = all_text[i:i+chunk_size].strip()
                    if len(chunk) > 50:  # Minimum chunk size
                        chunks.append(chunk)
                
                # Generate based on type
                data = []
                for chunk in chunks[:max_samples]:
                    if generation_type == "Q&A from chunks":
                        data.append({
                            "instruction": f"Based on the following text, provide a detailed explanation:\n\n{chunk[:200]}...",
                            "output": chunk
                        })
                    elif generation_type == "Summarization":
                        data.append({
                            "instruction": f"Summarize the following text:\n\n{chunk}",
                            "output": f"Summary: {chunk[:150]}..."
                        })
                    else:  # Completion
                        half = len(chunk) // 2
                        data.append({
                            "instruction": f"Continue the following text:\n\n{chunk[:half]}",
                            "output": chunk[half:]
                        })
                
                if data:
                    processor = st.session_state.dataset_processor
                    dataset = processor.load_from_dict_list(data)
                    st.session_state.training_dataset = dataset
                    
                    st.success(f"Generated {len(data)} samples!")
                    
                    # Show preview
                    st.subheader("Generated Samples Preview")
                    for i, sample in enumerate(data[:3]):
                        with st.expander(f"Sample {i+1}"):
                            st.markdown("**Instruction:**")
                            st.text(sample["instruction"][:300])
                            st.markdown("**Output:**")
                            st.text(sample["output"][:300])
                else:
                    st.error("Could not generate any samples")


def render_info_tab():
    """Render dataset information tab"""
    st.subheader("Current Dataset Information")
    
    # Clear dataset button at the top
    col_info, col_clear = st.columns([3, 1])
    with col_clear:
        if st.button("🗑️ Clear Dataset", type="secondary", use_container_width=True):
            st.session_state.training_dataset = None
            st.session_state.manual_samples = []
            if hasattr(st.session_state, 'dataset_processor'):
                st.session_state.dataset_processor.dataset = None
            st.success("Dataset cleared!")
            st.rerun()
    
    if st.session_state.training_dataset is None:
        st.info("No dataset loaded. Upload a dataset or generate one.")
        return
    
    dataset = st.session_state.training_dataset
    processor = st.session_state.dataset_processor
    stats = processor.validate_dataset()
    
    # Overview
    st.markdown("### Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", stats.total_samples)
    col2.metric("Avg Instruction", f"{stats.avg_instruction_length} words")
    col3.metric("Avg Output", f"{stats.avg_output_length} words")
    col4.metric("Valid Format", "Yes" if stats.format_valid else "No")
    
    # Distribution chart
    st.markdown("### Length Distribution")
    try:
        import plotly.graph_objects as go
        
        inst_lengths = [len(str(dataset[i].get("instruction", "")).split()) for i in range(len(dataset))]
        out_lengths = [len(str(dataset[i].get("output", "")).split()) for i in range(len(dataset))]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=inst_lengths, name="Instruction Length", opacity=0.7))
        fig.add_trace(go.Histogram(x=out_lengths, name="Output Length", opacity=0.7))
        fig.update_layout(
            barmode='overlay',
            xaxis_title="Word Count",
            yaxis_title="Frequency",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("Install plotly for distribution charts: pip install plotly")
    
    # Full dataset viewer
    st.markdown("### Dataset Viewer")
    
    page_size = st.selectbox("Samples per page", [5, 10, 25, 50], index=1)
    total_pages = (len(dataset) + page_size - 1) // page_size
    page = st.number_input("Page", 1, max(1, total_pages), 1) - 1
    
    start_idx = page * page_size
    end_idx = min(start_idx + page_size, len(dataset))
    
    for i in range(start_idx, end_idx):
        sample = dataset[i]
        with st.expander(f"Sample {i+1}: {str(sample.get('instruction', ''))[:50]}..."):
            st.markdown("**Instruction:**")
            st.text(sample.get("instruction", ""))
            st.markdown("**Output:**")
            st.text(sample.get("output", ""))
    
    # Export options
    st.markdown("---")
    st.markdown("### Export Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_name = st.text_input("Export filename", "training_data")
        
    with col2:
        export_format = st.selectbox("Format", ["JSONL", "CSV"])
    
    if st.button("Export Dataset"):
        try:
            export_path = Path(f"./data/training_datasets/{export_name}")
            
            if export_format == "JSONL":
                export_path = export_path.with_suffix(".jsonl")
                processor.save_dataset(dataset, str(export_path))
            else:
                export_path = export_path.with_suffix(".csv")
                data = [dataset[i] for i in range(len(dataset))]
                df = pd.DataFrame(data)
                df.to_csv(export_path, index=False)
            
            st.success(f"Dataset exported to {export_path}")
            
            # Download button
            with open(export_path, 'rb') as f:
                st.download_button(
                    "Download File",
                    f,
                    file_name=export_path.name,
                    mime="application/octet-stream"
                )
        except Exception as e:
            st.error(f"Export error: {str(e)}")
