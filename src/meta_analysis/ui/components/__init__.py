"""
Meta-Analysis UI Components

Reusable UI components for the meta-analysis Streamlit app.
"""

from src.meta_analysis.ui.components.simple_view_renderer import (
    render_simple_view,
    render_simple_comparison_panel,
    get_result_color,
    get_confidence_level,
    get_heterogeneity_description
)

__all__ = [
    "render_simple_view",
    "render_simple_comparison_panel",
    "get_result_color",
    "get_confidence_level",
    "get_heterogeneity_description"
]
