"""
MCP Activity Panel Component

Renders MCP server activity information in the Streamlit UI.
Shows ingestion stats, validation summary, and server status.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional


def render_mcp_activity_panel():
    """
    Render the MCP Server Activity panel.
    
    Reads from st.session_state and displays:
    - Active server name
    - Studies ingested count
    - Source type (upload/kaggle/synthetic)
    - Validation summary
    - Last ingestion timestamp
    """
    with st.expander("📡 MCP Server Activity", expanded=False):
        # Get MCP state values
        server_name = st.session_state.get("mcp_active_server", "CSVExperimentMCP")
        last_ingestion = st.session_state.get("mcp_last_ingestion")
        studies_count = st.session_state.get("mcp_studies_ingested", 0)
        source_type = st.session_state.get("mcp_source_type", "none")
        validation = st.session_state.get("mcp_validation_report", {})
        
        # Server info
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Active Server:** `{server_name}`")
            st.markdown(f"**Studies Ingested:** {studies_count}")
        
        with col2:
            source_icons = {
                "upload": "📤 File Upload",
                "kaggle": "📊 Kaggle Import",
                "synthetic": "🧪 Synthetic Generator",
                "none": "⏳ No data yet"
            }
            st.markdown(f"**Source:** {source_icons.get(source_type, source_type)}")
            
            if last_ingestion:
                if isinstance(last_ingestion, str):
                    ts_str = last_ingestion
                else:
                    ts_str = last_ingestion.strftime("%Y-%m-%d %H:%M:%S")
                st.markdown(f"**Last Ingestion:** {ts_str}")
        
        # Validation summary
        if validation:
            st.divider()
            st.markdown("**Validation Summary**")
            
            valid_count = validation.get("valid_records", 0)
            total_count = validation.get("total_records", 0)
            error_count = validation.get("error_count", 0)
            warning_count = validation.get("warning_count", 0)
            is_valid = validation.get("is_valid", True)
            
            # Metrics row
            m1, m2, m3 = st.columns(3)
            m1.metric("Valid Records", f"{valid_count}/{total_count}")
            m2.metric("Errors", error_count, delta_color="inverse")
            m3.metric("Warnings", warning_count, delta_color="off")
            
            # Status indicator
            if is_valid:
                st.success("✅ Data passed MCP validation")
            else:
                st.error("❌ Validation issues detected")
            
            # Show issues if any
            issues = validation.get("issues", [])
            if issues:
                with st.expander(f"View {len(issues)} Issues"):
                    for issue in issues:
                        icon = "❌" if issue.get("type") == "error" else "⚠️"
                        st.markdown(f"{icon} **{issue.get('field')}**: {issue.get('message')}")


def render_data_flow_overview(active_step: Optional[str] = None):
    """
    Render a visual data flow diagram showing the pipeline.
    
    Args:
        active_step: Currently active step to highlight
    """
    with st.expander("🔄 Data Flow Overview", expanded=False):
        # Define flow steps
        steps = [
            ("source", "📥 Data Source", "Upload / Kaggle / Synthetic"),
            ("mcp", "📡 MCP Server", "Ingestion & Validation"),
            ("harmonizer", "🔧 Harmonizer", "StandardizedStudy Conversion"),
            ("engine", "📊 Meta-Analysis Engine", "Statistical Analysis"),
            ("rag", "🤖 RAG Interpreter", "AI-Powered Insights")
        ]
        
        # Render flow
        for i, (step_id, title, description) in enumerate(steps):
            is_active = active_step == step_id
            
            if is_active:
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%); 
                            color: white; padding: 12px; border-radius: 8px; margin: 4px 0;">
                    <strong>{title}</strong><br/>
                    <small>{description}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #f0f2f6; padding: 12px; border-radius: 8px; margin: 4px 0;">
                    <strong>{title}</strong><br/>
                    <small style="color: #666;">{description}</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Arrow between steps
            if i < len(steps) - 1:
                st.markdown("<div style='text-align: center; color: #888;'>↓</div>", unsafe_allow_html=True)


def render_mcp_badge(show: bool = True):
    """
    Render the "Processed via MCP Server ✓" badge.
    
    Args:
        show: Whether to show the badge
    """
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
    validation_report: Optional[Dict[str, Any]] = None
):
    """
    Update MCP-related session state values.
    
    Args:
        server_name: Name of active MCP server
        studies_count: Number of studies ingested
        source_type: Source of data (upload/kaggle/synthetic)
        validation_report: Validation result dictionary
    """
    st.session_state.mcp_active_server = server_name
    st.session_state.mcp_last_ingestion = datetime.now()
    st.session_state.mcp_studies_ingested = studies_count
    st.session_state.mcp_source_type = source_type
    st.session_state.mcp_validation_report = validation_report or {}
