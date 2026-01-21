"""
Simple View Renderer

Renders meta-analysis results in plain language for non-technical users.
Uses color-coded indicators and simple summaries.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Any, Dict, Optional
import numpy as np


def get_result_color(pooled_effect: float, p_value: float, ci_lower: float, ci_upper: float) -> str:
    """
    Determine result color based on effect and significance.
    
    Returns:
        'green' = Variant performs better (significant positive effect)
        'red' = Variant performs worse (significant negative effect)
        'yellow' = Inconclusive
    """
    if p_value >= 0.05:
        return "yellow"  # Not significant
    
    if ci_lower > 0 and ci_upper > 0:
        return "green"  # Significant positive
    elif ci_lower < 0 and ci_upper < 0:
        return "red"  # Significant negative
    else:
        return "yellow"  # CI crosses zero


def get_confidence_level(p_value: float) -> str:
    """Convert p-value to plain language confidence level."""
    if p_value < 0.001:
        return "Very High"
    elif p_value < 0.01:
        return "High"
    elif p_value < 0.05:
        return "Moderate"
    elif p_value < 0.10:
        return "Low"
    else:
        return "Very Low"


def get_heterogeneity_description(i2: float) -> str:
    """Convert I² to plain language description."""
    if i2 < 25:
        return "Low (experiments show consistent results)"
    elif i2 < 50:
        return "Moderate (some variation between experiments)"
    elif i2 < 75:
        return "Substantial (notable differences between experiments)"
    else:
        return "High (experiments show very different results)"


def render_simple_view(state: Any) -> None:
    """
    Render meta-analysis results in simple, non-technical language.
    
    Args:
        state: MetaAnalysisState object with analysis results
    """
    if not state.meta_result:
        st.warning("No analysis results available.")
        return
    
    st.header("📊 Results Summary")
    
    # Extract key values
    pooled_effect = state.meta_result.pooled_effect
    p_value = state.meta_result.p_value
    ci = state.meta_result.confidence_interval
    i2 = state.heterogeneity_stats.get("I2", 0) if state.heterogeneity_stats else 0
    n_studies = state.meta_result.n_studies
    
    # Determine overall result
    result_color = get_result_color(pooled_effect, p_value, ci[0], ci[1])
    
    # Color mapping for Streamlit
    color_map = {
        "green": "🟢",
        "red": "🔴",
        "yellow": "🟡"
    }
    
    # Main result indicator
    st.markdown("---")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if result_color == "green":
            st.markdown(f"# {color_map[result_color]}")
            st.markdown("### Variant Wins!")
        elif result_color == "red":
            st.markdown(f"# {color_map[result_color]}")
            st.markdown("### Control Wins")
        else:
            st.markdown(f"# {color_map[result_color]}")
            st.markdown("### Inconclusive")
    
    with col2:
        # Simple summary paragraph
        if result_color == "green":
            summary = f"""
            **The variant (test version) performed better than the control.**
            
            Based on {n_studies} experiments, the variant shows a positive effect.
            We are {get_confidence_level(p_value).lower()} confident in this result.
            """
        elif result_color == "red":
            summary = f"""
            **The control (original version) performed better than the variant.**
            
            Based on {n_studies} experiments, the variant shows a negative effect.
            We are {get_confidence_level(p_value).lower()} confident in this result.
            """
        else:
            summary = f"""
            **We cannot determine a clear winner.**
            
            Based on {n_studies} experiments, the difference between control and variant 
            is not statistically significant. More data may be needed.
            """
        
        st.markdown(summary)
    
    st.markdown("---")
    
    # Key questions answered
    st.subheader("Key Insights")
    
    q1, q2 = st.columns(2)
    
    with q1:
        st.markdown("#### Did the variant win?")
        if result_color == "green":
            st.success("✅ Yes, the variant performed better")
        elif result_color == "red":
            st.error("❌ No, the control performed better")
        else:
            st.warning("❓ We can't be sure yet")
    
    with q2:
        st.markdown("#### How confident is the result?")
        conf_level = get_confidence_level(p_value)
        if conf_level in ["Very High", "High"]:
            st.success(f"✅ {conf_level} confidence")
        elif conf_level == "Moderate":
            st.info(f"ℹ️ {conf_level} confidence")
        else:
            st.warning(f"⚠️ {conf_level} confidence")
    
    q3, q4 = st.columns(2)
    
    with q3:
        st.markdown("#### Is there inconsistency between experiments?")
        het_desc = get_heterogeneity_description(i2)
        if i2 < 25:
            st.success(f"✅ {het_desc}")
        elif i2 < 50:
            st.info(f"ℹ️ {het_desc}")
        elif i2 < 75:
            st.warning(f"⚠️ {het_desc}")
        else:
            st.error(f"❌ {het_desc}")
    
    with q4:
        st.markdown("#### Is there any sign of bias?")
        bias_detected = False
        if state.publication_bias_result:
            bias_detected = state.publication_bias_result.get("bias_detected", False)
        
        if bias_detected:
            st.warning("⚠️ Some bias indicators detected")
        else:
            st.success("✅ No major bias detected")
    
    st.markdown("---")
    
    # Simple bar chart
    st.subheader("📈 Visual Comparison")
    
    # Create simple comparison from studies
    if state.standardized_studies:
        total_control_conv = sum(
            s.metadata.get("control_conversions", 0) 
            for s in state.standardized_studies 
            if s.metadata
        )
        total_treatment_conv = sum(
            s.metadata.get("treatment_conversions", 0) 
            for s in state.standardized_studies 
            if s.metadata
        )
        total_control_n = sum(s.sample_size_control for s in state.standardized_studies)
        total_treatment_n = sum(s.sample_size_treatment for s in state.standardized_studies)
        
        control_rate = (total_control_conv / total_control_n * 100) if total_control_n > 0 else 0
        treatment_rate = (total_treatment_conv / total_treatment_n * 100) if total_treatment_n > 0 else 0
        
        fig = go.Figure(data=[
            go.Bar(
                name='Control',
                x=['Conversion Rate'],
                y=[control_rate],
                marker_color='#636EFA',
                text=[f"{control_rate:.2f}%"],
                textposition='auto'
            ),
            go.Bar(
                name='Variant',
                x=['Conversion Rate'],
                y=[treatment_rate],
                marker_color='#00CC96' if treatment_rate > control_rate else '#EF553B',
                text=[f"{treatment_rate:.2f}%"],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Control vs Variant Conversion Rates",
            yaxis_title="Conversion Rate (%)",
            barmode='group',
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # What this means section
    st.markdown("---")
    with st.expander("💡 What This Means", expanded=True):
        st.markdown("""
        **Understanding the Results**
        
        - **Pooled Effect**: We combined all experiments to get one overall answer. 
          A positive effect means the variant did better; negative means the control did better.
        
        - **Confidence**: This tells us how sure we can be about the result. 
          Low confidence means we might need more data before making a decision.
        
        - **Inconsistency (Heterogeneity)**: If experiments show very different results, 
          we should be careful about combining them. It might mean different user segments 
          respond differently.
        
        - **Bias**: Publication bias happens when only "successful" experiments are reported. 
          If detected, the true effect might be smaller than shown.
        
        **Recommendation**: {}
        """.format(
            "Consider implementing the variant based on these results." 
            if result_color == "green" 
            else "Consider keeping the control version." 
            if result_color == "red" 
            else "Gather more data before making a decision."
        ))


def render_simple_comparison_panel(
    results_history: list,
    current_file: str
) -> None:
    """
    Render a comparison panel for multi-file results.
    
    Args:
        results_history: List of previous analysis results
        current_file: Name of currently analyzed file
    """
    if not results_history:
        return
    
    st.subheader("📁 File Comparison")
    
    comparison_data = []
    for result in results_history:
        if result.get("meta_result"):
            comparison_data.append({
                "File": result.get("filename", "Unknown"),
                "Experiments": result["meta_result"].n_studies,
                "Pooled Effect": f"{result['meta_result'].pooled_effect:.3f}",
                "I²": f"{result.get('heterogeneity', {}).get('I2', 0):.1f}%",
                "Bias": "⚠️" if result.get("bias_detected") else "✅"
            })
    
    if comparison_data:
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
