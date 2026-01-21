"""
Training Dashboard - Real-time training interface with live metrics
"""

import streamlit as st
import torch
import time
from pathlib import Path
from datetime import datetime


def render_training_dashboard():
    """Render training interface with live monitoring"""
    st.title("Training Dashboard")
    st.markdown("Configure and execute QLoRA fine-tuning with real-time monitoring")
    
    # Check prerequisites
    model_loaded = st.session_state.current_model_loaded is not None
    dataset_loaded = st.session_state.training_dataset is not None
    
    if not model_loaded or not dataset_loaded:
        st.warning("Prerequisites Required")
        
        col1, col2 = st.columns(2)
        with col1:
            status = "Loaded" if model_loaded else "Not loaded"
            st.info(f"**Model:** {status}")
            if not model_loaded:
                st.markdown("Load a model in the configuration below")
        
        with col2:
            status = "Loaded" if dataset_loaded else "Not loaded"
            st.info(f"**Dataset:** {status}")
            if not dataset_loaded:
                st.markdown("Upload dataset in Dataset Studio")
        
        st.markdown("---")
    
    # Configuration section
    st.subheader("Model & Training Configuration")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Base Model Selection")
        
        # Get available models
        model_manager = st.session_state.model_manager
        models = model_manager.list_models()
        
        model_options = list(models.keys())
        selected_model = st.selectbox(
            "Choose base model",
            model_options,
            help="Select a model optimized for your VRAM"
        )
        
        # Show model info
        if selected_model:
            info = models[selected_model]
            st.info(f"""
            **Size:** {info['size']}  
            **VRAM Required:** {info['vram_required']}  
            **Context:** {info['context_length']} tokens  
            **Best for:** {info['recommended_for']}
            """)
        
        # Load model button
        if st.button("Load Model", disabled=st.session_state.get('training_active', False)):
            with st.spinner(f"Loading {selected_model}..."):
                try:
                    model, tokenizer = model_manager.load_model(selected_model)
                    st.session_state.current_model_loaded = selected_model
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.success(f"{selected_model} loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
    
    with col2:
        st.markdown("#### Training Parameters")
        
        # Training config
        num_epochs = st.number_input("Number of Epochs", 1, 10, 3, help="Training passes over dataset")
        batch_size = st.number_input("Batch Size", 1, 4, 1, help="Samples per batch (1 recommended for 4GB VRAM)")
        gradient_accumulation = st.number_input("Gradient Accumulation Steps", 1, 16, 4, help="Effective batch size = batch_size x this")
        learning_rate = st.number_input("Learning Rate", 0.00001, 0.001, 0.0002, format="%.5f")
        
        # LoRA config
        with st.expander("Advanced LoRA Settings"):
            lora_r = st.slider("LoRA Rank (r)", 4, 64, 16, help="Higher = more parameters, better fit")
            lora_alpha = st.slider("LoRA Alpha", 8, 64, 32, help="Scaling factor, usually 2x rank")
            lora_dropout = st.slider("LoRA Dropout", 0.0, 0.3, 0.05, help="Regularization")
            max_seq_length = st.number_input("Max Sequence Length", 128, 2048, 512, help="Maximum tokens per sample")
    
    st.markdown("---")
    
    # Training execution
    if model_loaded and dataset_loaded:
        st.subheader("Training Execution")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if not st.session_state.get('training_active', False):
                if st.button("Start Training", type="primary", use_container_width=True):
                    start_training(
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        gradient_accumulation=gradient_accumulation,
                        learning_rate=learning_rate,
                        lora_r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        max_seq_length=max_seq_length
                    )
            else:
                st.button("Training in Progress...", disabled=True, use_container_width=True)
        
        with col2:
            if st.button("Clear Training State", use_container_width=True):
                st.session_state.training_metrics = []
                st.session_state.training_active = False
                st.success("Cleared")
        
        with col3:
            if st.button("Unload Model", use_container_width=True):
                st.session_state.model_manager.unload_model()
                st.session_state.current_model_loaded = None
                st.session_state.model = None
                st.session_state.tokenizer = None
                st.success("Model unloaded")
                st.rerun()
        
        # Training metrics display
        render_training_metrics()


def render_training_metrics():
    """Render training metrics and charts"""
    if 'training_metrics' not in st.session_state or not st.session_state.training_metrics:
        return
    
    st.markdown("---")
    st.subheader("Training Metrics")
    
    metrics = st.session_state.training_metrics
    
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    if metrics:
        latest = metrics[-1]
        col1.metric("Current Step", latest.get('step', 0))
        col2.metric("Training Loss", f"{latest.get('loss', 0):.4f}")
        col3.metric("Learning Rate", f"{latest.get('learning_rate', 0):.2e}")
        
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1e9
            col4.metric("VRAM Usage", f"{vram:.2f} GB")
    
    # Loss curve
    if len(metrics) > 1:
        st.markdown("#### Loss Curve")
        
        try:
            import plotly.graph_objects as go
            
            steps = [m.get('step', 0) for m in metrics]
            losses = [m.get('loss', 0) for m in metrics]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=steps,
                y=losses,
                mode='lines+markers',
                name='Training Loss',
                line=dict(color='#FF6B6B', width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                xaxis_title="Training Step",
                yaxis_title="Loss",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            # Fallback without plotly
            st.line_chart({"Loss": [m.get('loss', 0) for m in metrics]})


def start_training(**config):
    """Initialize and start training"""
    from src.elite_app.core.qlora_trainer import QLoRATrainer, TrainingConfig
    
    # Create training config
    training_config = TrainingConfig(
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation'],
        learning_rate=config['learning_rate'],
        lora_r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        max_seq_length=config['max_seq_length']
    )
    
    # Initialize trainer
    trainer = QLoRATrainer(
        model=st.session_state.model,
        tokenizer=st.session_state.tokenizer,
        config=training_config
    )
    
    # Prepare dataset
    processor = st.session_state.dataset_processor
    formatted_dataset = processor.format_for_training(st.session_state.training_dataset)
    
    # Initialize metrics storage
    st.session_state.training_metrics = []
    st.session_state.training_active = True
    
    # Progress callback
    def progress_callback(step, loss, logs):
        metrics_entry = {
            'step': step,
            'loss': loss,
            'learning_rate': logs.get('learning_rate', 0),
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.training_metrics.append(metrics_entry)
    
    # Start training
    progress_bar = st.progress(0, text="Preparing training...")
    status_text = st.empty()
    
    try:
        status_text.text("Training in progress...")
        result = trainer.train(formatted_dataset, progress_callback=progress_callback)
        
        # Save model
        timestamp = int(time.time())
        output_path = f"./data/fine_tuned_models/{st.session_state.current_model_loaded}_{timestamp}"
        trainer.save_model(output_path)
        
        st.session_state.trainer = trainer
        st.session_state.last_trained_model_path = output_path
        st.session_state.training_active = False
        
        progress_bar.progress(100, text="Training complete!")
        st.success(f"Training complete! Model saved to {output_path}")
        
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        st.session_state.training_active = False
