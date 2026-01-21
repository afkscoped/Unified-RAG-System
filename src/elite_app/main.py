"""
Elite App - QLoRA Training Studio
Complete fine-tuning interface optimized for consumer hardware
"""

import streamlit as st
import sys
from pathlib import Path
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def configure_page():
    """Configure Streamlit page settings (must be first Streamlit command)."""
    st.set_page_config(
        page_title="QLoRA Training Studio",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Import components
try:
    from src.elite_app.core.model_manager import ModelManager
    from src.elite_app.core.dataset_processor import DatasetProcessor
    from src.elite_app.ui.dataset_studio import render_dataset_studio
    from src.elite_app.ui.training_dashboard import render_training_dashboard
    from src.elite_app.ui.model_tester import render_model_tester
    from src.elite_app.ui.export_manager import render_export_manager
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)


def init_session_state():
    """Initialize session state variables"""
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    
    if 'dataset_processor' not in st.session_state:
        st.session_state.dataset_processor = DatasetProcessor()
    
    if 'trainer' not in st.session_state:
        st.session_state.trainer = None
    
    if 'inference_engine' not in st.session_state:
        st.session_state.inference_engine = None
    
    if 'current_model_loaded' not in st.session_state:
        st.session_state.current_model_loaded = None
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    
    if 'training_dataset' not in st.session_state:
        st.session_state.training_dataset = None
    
    if 'training_active' not in st.session_state:
        st.session_state.training_active = False
    
    if 'training_metrics' not in st.session_state:
        st.session_state.training_metrics = []


def render_sidebar():
    """Render sidebar with system info and navigation"""
    st.sidebar.title("🧬 QLoRA Training Studio")
    st.sidebar.markdown("*Fine-tune LLMs on consumer hardware*")
    st.sidebar.markdown("---")
    
    # System info
    st.sidebar.subheader("System Status")
    
    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        used_vram = torch.cuda.memory_allocated(0) / 1e9
        
        st.sidebar.success(f"**GPU:** {gpu_name}")
        st.sidebar.metric("VRAM", f"{used_vram:.2f} / {total_vram:.1f} GB")
        
        # VRAM bar
        vram_percent = (used_vram / total_vram) * 100
        st.sidebar.progress(vram_percent / 100, text=f"{vram_percent:.1f}% used")
    else:
        st.sidebar.warning("**No GPU detected** - Training will be slow")
    
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.subheader("Navigation")
    page = st.sidebar.radio(
        "Select workspace",
        ["Dataset Studio", "Training Dashboard", "Model Tester", "Export & Deploy"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Quick stats
    st.sidebar.subheader("Quick Stats")
    col1, col2 = st.sidebar.columns(2)
    
    model_loaded = st.session_state.current_model_loaded is not None
    dataset_loaded = st.session_state.training_dataset is not None
    
    with col1:
        if model_loaded:
            st.success("Model: ✓")
        else:
            st.error("Model: ✗")
    
    with col2:
        if dataset_loaded:
            st.success("Dataset: ✓")
        else:
            st.error("Dataset: ✗")
    
    # Show loaded model name
    if model_loaded:
        st.sidebar.caption(f"Loaded: {st.session_state.current_model_loaded}")
    
    # Show dataset info
    if dataset_loaded:
        dataset = st.session_state.training_dataset
        st.sidebar.caption(f"Samples: {len(dataset)}")
    
    st.sidebar.markdown("---")
    
    # Help section
    with st.sidebar.expander("Help"):
        st.markdown("""
        **Quick Start:**
        1. Upload dataset in Dataset Studio
        2. Load model in Training Dashboard
        3. Configure and start training
        4. Test model in Model Tester
        5. Export in Export & Deploy
        
        **Tips:**
        - Use batch size 1 for 4GB VRAM
        - Start with TinyLlama for testing
        - Monitor VRAM during training
        """)
    
    return page


def main():
    """Main application entry point"""
    configure_page()
    render_app(set_page_config=False)


def render_app(set_page_config: bool = False):
    """Render the Elite QLoRA Training Studio.

    Args:
        set_page_config: If True, calls st.set_page_config(). Keep False when
        embedding inside another Streamlit entrypoint that already configured the page.
    """
    if set_page_config:
        configure_page()

    # Check imports
    if not IMPORTS_OK:
        st.error("Import Error")
        st.code(IMPORT_ERROR)
        st.info("Please ensure all dependencies are installed:")
        st.code("pip install torch transformers peft bitsandbytes datasets streamlit plotly pandas")
        st.stop()
    
    # Initialize session state
    init_session_state()
    
    # Ensure data directories exist
    Path("./data/training_datasets").mkdir(parents=True, exist_ok=True)
    Path("./data/fine_tuned_models").mkdir(parents=True, exist_ok=True)
    Path("./data/exports").mkdir(parents=True, exist_ok=True)
    Path("./cache/model_cache").mkdir(parents=True, exist_ok=True)
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render selected page
    if page == "Dataset Studio":
        render_dataset_studio()
    elif page == "Training Dashboard":
        render_training_dashboard()
    elif page == "Model Tester":
        render_model_tester()
    elif page == "Export & Deploy":
        render_export_manager()


if __name__ == "__main__":
    main()
