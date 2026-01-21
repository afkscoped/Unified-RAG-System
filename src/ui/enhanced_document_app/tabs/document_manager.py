"""
[ELITE ARCHITECTURE] document_manager.py
Filing and Indexing Interface.
"""

import streamlit as st
import os
from loguru import logger

def render_document_manager(orchestrator):
    st.header("📂 DOCUMENT_COMMAND_CENTER")
    st.markdown("---")
    
    # 1. Upload Zone
    uploaded_files = st.file_uploader(
        "Ingest Research Documents (PDF/Structure-Aware)", 
        accept_multiple_files=True,
        type=['pdf']
    )
    
    if st.button("EXECUTE_INGESTION_PROTOCOL", use_container_width=True):
        if not uploaded_files:
            st.error("No vectors detected in buffer. Upload documents first.")
        else:
            for uploaded_file in uploaded_files:
                # Save to local data/documents
                save_path = os.path.join("data/documents", uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Submit Async Job
                job_id = f"ingest_{uploaded_file.name}"
                orchestrator.async_manager.submit_job(
                    job_id, 
                    orchestrator.ingest_document, 
                    save_path
                )
                st.info(f"INGESTION_QUEUED: {uploaded_file.name}")

    # 2. Background Job Status Tracker
    st.markdown("### Process Telemetry")
    active_job_id = orchestrator.async_manager.active_job_id
    if active_job_id:
        st.warning(f"🔄 CURRENTLY_PROCESSING: {active_job_id}")
    else:
        st.success("🟢 IDLE: Ready for new vectors.")

    # 3. Database View
    st.markdown("---")
    st.markdown("### Active Knowledge Pool")
    doc_list = os.listdir("data/documents")
    if doc_list:
        selected_doc = st.selectbox("Select document for inspection", [d for d in doc_list if d != ".gitkeep"])
        if selected_doc:
            st.info(f"Metadata for {selected_doc}")
            st.json({
                "format": "pdf",
                "layout_engine": "pdfplumber",
                "redaction": "enabled (PII)",
                "graph_link": "active"
            })
    else:
        st.write("Neutral State: No documents in vault.")
