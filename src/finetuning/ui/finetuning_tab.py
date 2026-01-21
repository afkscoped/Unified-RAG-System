"""
[ELITE ARCHITECTURE] finetuning_tab.py
Streamlit Lab Interface for Training and Monitoring.
"""

import streamlit as st
import pandas as pd
import os
import time
import threading
import tempfile
from pathlib import Path
from src.finetuning.memory_optimizer import MemoryOptimizer
from src.finetuning.dataset_generator import DatasetGenerator

def render_finetuning_tab(system=None):
    st.header("🧬 QLoRA_LAB_01: EVOLUTIONARY_FINETUNING")
    
    # 0. Lab Manual (NEW)
    with st.expander("📖 LAB_MANUAL: HOW TO WORK", expanded=False):
        st.markdown("""
        ### Welcome to the QLoRA Lab
        This interface allows you to fine-tune Large Language Models on your own document corpus.
        
        #### Phase 1: Dataset Synthesis ([CONFIG_SETUP])
        1. **Upload Documents**: Use the "Direct Lab Upload" section below to add documents.
        2. **Process & Index**: Click "🚀 Process & Index Documents" to prepare them.
        3. **Generate Synthetic Dataset**: Click the synthesis button to create Q&A pairs.
        
        #### Phase 2: Configuration
        1. Set your **LoRA Rank (r)**: Higher values allow more complex adaptation but use more VRAM (16 is recommended).
        2. Select your **Target Model**: Phi-3-mini is optimized for the GTX 1650.
        
        #### Phase 3: Ignition ([INITIATE_TRAIN])
        1. Review the dataset path.
        2. Click **IGNITE TRAINING SEQUENCE**. 
        3. Monitoring: Watch the **VRAM_ALLOCATED** metric and the **Loss Curve**.
        
        > [!WARNING]
        > Training uses 100% of GPU resources. Avoid running video players or games during ignition.
        """)

    st.markdown("---")
    
    # 1. Hardware Stats
    col1, col2, col3 = st.columns(3)
    stats = MemoryOptimizer.get_vram_stats()
    with col1: st.metric("HARDWARE_ID", stats['device'])
    with col2: st.metric("VRAM_ALLOCATED", f"{stats.get('allocated_gb', 0):.2f} GB")
    with col3: st.metric("VRAM_RESERVED", f"{stats.get('reserved_gb', 0):.2f} GB")

    st.subheader("Training Control Plane")
    tab_train, tab_setup, tab_eval = st.tabs(["[INITIATE_TRAIN]", "[CONFIG_SETUP]", "[MODEL_METRICS]"])
    
    # Session State for Training Logs
    if 'train_logs' not in st.session_state:
        st.session_state.train_logs = []
    if 'train_metrics' not in st.session_state:
        st.session_state.train_metrics = []
    if 'is_training' not in st.session_state:
        st.session_state.is_training = False

    # 2. Setup Tab
    with tab_setup:
        st.info("Configure LoRA Hyperparameters for target hardware.")
        
        # 2.1 Direct Lab Upload (CRITICAL FIX)
        st.markdown("### 📤 Direct Lab Upload")
        lab_files = st.file_uploader("Upload training source documents", type=['pdf', 'txt', 'docx'], accept_multiple_files=True, key="lab_uploader_fixed")
        
        if lab_files and st.button("🚀 Process & Index Documents", type="primary"):
            if system:
                with st.spinner("Processing documents for fine-tuning..."):
                    total = 0
                    for file in lab_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                            tmp.write(file.read())
                            tmp_path = tmp.name
                        try:
                            # Handle both EnhancedAnalyzer (with .rag) and EliteRAGSystem (direct)
                            if hasattr(system, 'ingest_document'):
                                chunks = system.ingest_document(tmp_path)
                            elif hasattr(system, 'rag'):
                                chunks = system.rag.ingest_file(tmp_path, original_name=file.name)
                            else:
                                st.error("Unsupported RAG system type.")
                                break
                            total += chunks
                        finally:
                            os.unlink(tmp_path)
                    
                    if total > 0:
                        # EliteRAG handles indexing internally, but Unified needs build_index
                        if hasattr(system, 'rag') and hasattr(system.rag, 'build_index'):
                            system.rag.build_index()
                        st.success(f"Successfully indexed {total} chunks from {len(lab_files)} documents.")
            else:
                st.error("RAG System not available for ingestion.")
        
        st.divider()

        r = st.slider("LoRA Rank (r)", 4, 128, 16)
        alpha = st.slider("LoRA Alpha", 8, 256, 32)
        target_model = st.select_slider("Target Model", ["microsoft/Phi-3-mini-4k-instruct", "mistralai/Mistral-7B-v0.1"])
        
        if st.button("Generate Synthetic Dataset from Corpus"):
            if system:
                with st.spinner("Synthesizing Q&A pairs in optimized batches..."):
                    try:
                        corpus = system.get_corpus_data()
                        texts = corpus.get("texts", [])
                        if not texts:
                            st.error("No documents indexed. Please upload documents in the '📤 Direct Lab Upload' section above.")
                        else:
                            gen = DatasetGenerator()
                            dataset = gen.generate_qa_triplets(texts[:10], samples_per_chunk=2)
                            path = gen.export_dataset(dataset)
                            st.success(f"Dataset generated at {path} ({len(dataset)} samples)")
                            st.session_state['qlora_dataset_path'] = path
                            
                            # UI Enhancement: Preview & Download
                            if dataset:
                                st.subheader("🔍 Dataset Preview")
                                st.json(dataset[:3], expanded=False) # Show first 3 samples
                                
                                with open(path, "r") as f:
                                    json_content = f.read()
                                
                                st.download_button(
                                    label="📥 Download Dataset JSON",
                                    data=json_content,
                                    file_name=os.path.basename(path),
                                    mime="application/json"
                                )
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
            else:
                st.error("System corpus not accessible.")

    # 3. Training Tab
    with tab_train:
        default_path = st.session_state.get('qlora_dataset_path', "data/finetuning/datasets/train_v1.json")
        dataset_path = st.text_input("Dataset Path", default_path)
        
        file_exists = os.path.exists(dataset_path)
        if not file_exists:
            st.warning("⚠️ Dataset definition not found. Use [CONFIG_SETUP] or upload documents.")
        else:
            st.success("✅ Dataset loaded. Ready for ignition.")
            
        if st.button("IGNITE TRAINING SEQUENCE", disabled=not file_exists or st.session_state.is_training, use_container_width=True):
            from src.finetuning.qlora_trainer import QLoRATrainer
            from datasets import load_dataset
            
            st.session_state.is_training = True
            st.session_state.train_logs = ["Engine warm-up..."]
            st.session_state.train_metrics = []
            
            def training_thread():
                try:
                    trainer = QLoRATrainer()
                    dataset = load_dataset("json", data_files=dataset_path, split="train")
                    
                    def log_callback(step, max_steps, loss):
                        st.session_state.train_metrics.append({"Step": step, "Loss": loss})
                        st.session_state.train_logs.append(f"Step {step}/{max_steps} | Loss: {loss:.4f}")

                    trainer.run_training(dataset, progress_callback=log_callback)
                    st.session_state.train_logs.append("FINETUNING_COMPLETE: Adapter saved.")
                except Exception as e:
                    st.session_state.train_logs.append(f"CRITICAL_FAILURE: {str(e)}")
                finally:
                    st.session_state.is_training = False

            thread = threading.Thread(target=training_thread)
            thread.start()

        if st.session_state.is_training or st.session_state.train_logs:
            st.markdown("---")
            st.code("\n".join(st.session_state.train_logs[-10:]), language="bash")
            if st.session_state.train_metrics:
                st.line_chart(pd.DataFrame(st.session_state.train_metrics), x="Step", y="Loss")

    # 4. Metrics Tab
    with tab_eval:
        if st.session_state.train_metrics:
            st.dataframe(pd.DataFrame(st.session_state.train_metrics), use_container_width=True)
        else:
            st.info("No active training metrics.")

if __name__ == "__main__":
    render_finetuning_tab()
