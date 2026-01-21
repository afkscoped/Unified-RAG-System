"""
Visualization Module

Creates statistical visualizations for meta-analysis:
- Forest plots
- Funnel plots
- Heterogeneity charts
- L'Abbé plots
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

from src.meta_analysis.mcp_servers.base_experiment_mcp import StandardizedStudy
from src.meta_analysis.statistical.meta_analyzer import MetaAnalysisResult


class MetaAnalysisVisualizer:
    """
    Creates visualizations for meta-analysis results.
    
    All plots are created using Plotly for interactivity.
    """
    
    def __init__(
        self,
        forest_height: int = 600,
        funnel_height: int = 500,
        theme: str = "plotly_white"
    ):
        """
        Initialize visualizer.
        
        Args:
            forest_height: Height of forest plots in pixels
            funnel_height: Height of funnel plots in pixels
            theme: Plotly theme to use
        """
        self.forest_height = forest_height
        self.funnel_height = funnel_height
        self.theme = theme
    
    def create_forest_plot(
        self,
        result: MetaAnalysisResult,
        title: str = "Forest Plot",
        show_weights: bool = True,
        sort_by: str = "effect"
    ) -> go.Figure:
        """
        Create a forest plot showing individual and pooled effects.
        
        Args:
            result: MetaAnalysisResult from meta-analysis
            title: Plot title
            show_weights: Whether to show study weights
            sort_by: How to sort studies ("effect", "weight", "name", "se")
            
        Returns:
            Plotly Figure object
        """
        studies = result.studies
        
        if not studies:
            logger.warning("No studies to plot")
            return go.Figure()
        
        # Sort studies
        if sort_by == "effect":
            studies = sorted(studies, key=lambda s: s.effect_size, reverse=True)
        elif sort_by == "weight":
            studies = sorted(studies, key=lambda s: result.study_weights.get(s.study_id, 0), reverse=True)
        elif sort_by == "se":
            studies = sorted(studies, key=lambda s: s.standard_error)
        else:  # by name
            studies = sorted(studies, key=lambda s: s.study_name)
        
        n_studies = len(studies)
        
        # Y positions (bottom to top, with space for summary)
        y_positions = list(range(n_studies + 2, 1, -1))  # Leave space at bottom for summary
        
        fig = go.Figure()
        
        # Add vertical line at null effect (0)
        fig.add_shape(
            type="line",
            x0=0, x1=0,
            y0=0, y1=n_studies + 3,
            line=dict(color="gray", dash="dash", width=1)
        )
        
        # Add individual study effects
        for i, (study, y_pos) in enumerate(zip(studies, y_positions)):
            weight = result.study_weights.get(study.study_id, 0)
            
            # Effect point
            fig.add_trace(go.Scatter(
                x=[study.effect_size],
                y=[y_pos],
                mode="markers",
                marker=dict(
                    size=np.sqrt(weight) * 3 + 5,  # Size proportional to weight
                    color="steelblue",
                    symbol="square"
                ),
                name="",
                showlegend=False,
                hovertemplate=(
                    f"<b>{study.study_name}</b><br>"
                    f"Effect: {study.effect_size:.3f}<br>"
                    f"95% CI: ({study.confidence_interval_lower:.3f}, {study.confidence_interval_upper:.3f})<br>"
                    f"Weight: {weight:.1f}%<br>"
                    f"N: {study.total_sample_size}"
                    "<extra></extra>"
                )
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=[study.confidence_interval_lower, study.confidence_interval_upper],
                y=[y_pos, y_pos],
                mode="lines",
                line=dict(color="steelblue", width=2),
                showlegend=False,
                hoverinfo="skip"
            ))
        
        # Add pooled effect (diamond shape)
        pooled_y = 1
        ci_lower, ci_upper = result.confidence_interval
        
        # Diamond vertices
        diamond_x = [
            ci_lower,
            result.pooled_effect,
            ci_upper,
            result.pooled_effect,
            ci_lower
        ]
        diamond_y = [
            pooled_y,
            pooled_y + 0.3,
            pooled_y,
            pooled_y - 0.3,
            pooled_y
        ]
        
        fig.add_trace(go.Scatter(
            x=diamond_x,
            y=diamond_y,
            fill="toself",
            fillcolor="darkred",
            line=dict(color="darkred", width=2),
            name="Pooled Effect",
            hovertemplate=(
                f"<b>Pooled Effect ({result.model_type.title()})</b><br>"
                f"Effect: {result.pooled_effect:.3f}<br>"
                f"95% CI: ({ci_lower:.3f}, {ci_upper:.3f})<br>"
                f"p-value: {result.p_value:.4f}<br>"
                f"I²: {result.heterogeneity['I2']:.1f}%"
                "<extra></extra>"
            )
        ))
        
        # Build y-axis labels
        y_labels = [s.study_name[:30] + ("..." if len(s.study_name) > 30 else "") for s in studies]
        y_labels.append("")  # Spacer
        y_labels.append(f"Pooled ({result.model_type})")
        
        # Add weight annotations
        if show_weights:
            for i, (study, y_pos) in enumerate(zip(studies, y_positions)):
                weight = result.study_weights.get(study.study_id, 0)
                fig.add_annotation(
                    x=1.02,
                    y=y_pos,
                    xref="paper",
                    text=f"{weight:.1f}%",
                    showarrow=False,
                    font=dict(size=10),
                    align="left"
                )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Effect Size (Log Odds Ratio)",
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(1, n_studies + 3)),
                ticktext=y_labels[::-1],
                showgrid=False
            ),
            template=self.theme,
            height=max(self.forest_height, n_studies * 25 + 150),
            showlegend=False,
            margin=dict(l=200, r=80 if show_weights else 40)
        )
        
        # Add heterogeneity info annotation
        fig.add_annotation(
            x=0.5,
            y=-0.1,
            xref="paper",
            yref="paper",
            text=(
                f"Heterogeneity: I² = {result.heterogeneity['I2']:.1f}%, "
                f"τ² = {result.heterogeneity['tau2']:.4f}, "
                f"Q = {result.heterogeneity['Q']:.2f} (p = {result.heterogeneity['Q_pvalue']:.3f})"
            ),
            showarrow=False,
            font=dict(size=11)
        )
        
        return fig
    
    def create_funnel_plot(
        self,
        result: MetaAnalysisResult,
        title: str = "Funnel Plot",
        show_contours: bool = True
    ) -> go.Figure:
        """
        Create a funnel plot for publication bias assessment.
        
        Args:
            result: MetaAnalysisResult
            title: Plot title
            show_contours: Whether to show significance contours
            
        Returns:
            Plotly Figure object
        """
        studies = result.studies
        
        if not studies:
            return go.Figure()
        
        effects = np.array([s.effect_size for s in studies])
        standard_errors = np.array([s.standard_error for s in studies])
        
        pooled = result.pooled_effect
        
        fig = go.Figure()
        
        # Add pseudo confidence interval contours
        if show_contours:
            se_range = np.linspace(0.001, np.max(standard_errors) * 1.2, 100)
            
            for alpha, color, width in [(0.05, "lightgray", 1), (0.01, "darkgray", 1)]:
                z = 1.96 if alpha == 0.05 else 2.576
                lower = pooled - z * se_range
                upper = pooled + z * se_range
                
                fig.add_trace(go.Scatter(
                    x=lower,
                    y=se_range,
                    mode="lines",
                    line=dict(color=color, width=width, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip"
                ))
                fig.add_trace(go.Scatter(
                    x=upper,
                    y=se_range,
                    mode="lines",
                    line=dict(color=color, width=width, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip"
                ))
        
        # Add vertical line at pooled effect
        fig.add_shape(
            type="line",
            x0=pooled, x1=pooled,
            y0=0, y1=np.max(standard_errors) * 1.1,
            line=dict(color="darkred", width=2)
        )
        
        # Add individual studies
        fig.add_trace(go.Scatter(
            x=effects,
            y=standard_errors,
            mode="markers",
            marker=dict(
                size=10,
                color="steelblue",
                opacity=0.7
            ),
            text=[s.study_name for s in studies],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Effect: %{x:.3f}<br>"
                "SE: %{y:.3f}"
                "<extra></extra>"
            ),
            name="Studies"
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Effect Size",
            yaxis_title="Standard Error",
            yaxis=dict(
                autorange="reversed"  # Invert y-axis (larger SE at bottom)
            ),
            template=self.theme,
            height=self.funnel_height,
            showlegend=False
        )
        
        return fig
    
    def create_heterogeneity_chart(
        self,
        result: MetaAnalysisResult,
        title: str = "Heterogeneity Analysis"
    ) -> go.Figure:
        """
        Create a chart visualizing heterogeneity statistics.
        
        Args:
            result: MetaAnalysisResult
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        i2 = result.heterogeneity["I2"]
        tau2 = result.heterogeneity["tau2"]
        q = result.heterogeneity["Q"]
        q_pvalue = result.heterogeneity["Q_pvalue"]
        
        # Create gauge chart for I²
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "indicator"}, {"type": "bar"}]],
            column_widths=[0.5, 0.5]
        )
        
        # I² gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=i2,
                title={"text": "I² (Heterogeneity)"},
                gauge=dict(
                    axis=dict(range=[0, 100]),
                    bar=dict(color="steelblue"),
                    steps=[
                        {"range": [0, 25], "color": "lightgreen"},
                        {"range": [25, 50], "color": "lightyellow"},
                        {"range": [50, 75], "color": "lightsalmon"},
                        {"range": [75, 100], "color": "lightcoral"}
                    ],
                    threshold=dict(
                        line=dict(color="red", width=4),
                        thickness=0.75,
                        value=i2
                    )
                ),
                number={"suffix": "%"}
            ),
            row=1, col=1
        )
        
        # Statistics bar chart
        fig.add_trace(
            go.Bar(
                x=["τ²", "τ", "Q/df"],
                y=[tau2, np.sqrt(tau2), q / max(1, result.n_studies - 1)],
                marker_color=["steelblue", "steelblue", "coral"],
                text=[f"{tau2:.4f}", f"{np.sqrt(tau2):.4f}", f"{q / max(1, result.n_studies - 1):.2f}"],
                textposition="outside",
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            template=self.theme,
            height=400
        )
        
        # Add Q-test result annotation
        sig_text = "significant" if q_pvalue < 0.05 else "not significant"
        fig.add_annotation(
            x=0.75,
            y=-0.15,
            xref="paper",
            yref="paper",
            text=f"Q-test: Q = {q:.2f}, df = {result.n_studies - 1}, p = {q_pvalue:.3f} ({sig_text})",
            showarrow=False,
            font=dict(size=11)
        )
        
        return fig
    
    def create_leave_one_out_plot(
        self,
        loo_results: Dict[str, Dict[str, float]],
        original_effect: float,
        original_ci: Tuple[float, float],
        title: str = "Leave-One-Out Sensitivity Analysis"
    ) -> go.Figure:
        """
        Create a forest-style plot for leave-one-out analysis.
        
        Args:
            loo_results: Results from leave_one_out_analysis
            original_effect: Original pooled effect
            original_ci: Original confidence interval
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        if not loo_results:
            return go.Figure()
        
        study_ids = list(loo_results.keys())
        n = len(study_ids)
        
        fig = go.Figure()
        
        # Add vertical line at original effect
        fig.add_shape(
            type="line",
            x0=original_effect, x1=original_effect,
            y0=0, y1=n + 1,
            line=dict(color="darkred", dash="dash", width=2)
        )
        
        # Add original CI as shaded region
        fig.add_shape(
            type="rect",
            x0=original_ci[0], x1=original_ci[1],
            y0=0, y1=n + 1,
            fillcolor="rgba(255, 0, 0, 0.1)",
            line=dict(width=0)
        )
        
        # Add each LOO result
        for i, study_id in enumerate(study_ids):
            loo = loo_results[study_id]
            y_pos = n - i
            
            color = "darkred" if loo["direction_changed"] else "steelblue"
            
            # Effect point
            fig.add_trace(go.Scatter(
                x=[loo["pooled_effect"]],
                y=[y_pos],
                mode="markers",
                marker=dict(size=8, color=color),
                showlegend=False,
                hovertemplate=(
                    f"<b>Excluding: {study_id}</b><br>"
                    f"Effect: {loo['pooled_effect']:.3f}<br>"
                    f"Change: {loo['effect_change']:+.3f}"
                    "<extra></extra>"
                )
            ))
            
            # CI line
            fig.add_trace(go.Scatter(
                x=[loo["ci_lower"], loo["ci_upper"]],
                y=[y_pos, y_pos],
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
                hoverinfo="skip"
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Pooled Effect (when study excluded)",
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(1, n + 1)),
                ticktext=study_ids[::-1],
                showgrid=False
            ),
            template=self.theme,
            height=max(400, n * 25 + 100)
        )
        
        return fig
    
    def create_cumulative_plot(
        self,
        cumulative_results: List[Dict[str, Any]],
        title: str = "Cumulative Meta-Analysis"
    ) -> go.Figure:
        """
        Create a cumulative meta-analysis plot.
        
        Args:
            cumulative_results: Results from cumulative analysis
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        if not cumulative_results:
            return go.Figure()
        
        n = len(cumulative_results)
        
        effects = [r["pooled_effect"] for r in cumulative_results]
        ci_lowers = [r["ci_lower"] for r in cumulative_results]
        ci_uppers = [r["ci_upper"] for r in cumulative_results]
        labels = [f"{r['n_studies']} studies" for r in cumulative_results]
        
        fig = go.Figure()
        
        # Add null line
        fig.add_shape(
            type="line",
            x0=0, x1=0,
            y0=0, y1=n + 1,
            line=dict(color="gray", dash="dash")
        )
        
        # Add effects with CIs
        for i, (effect, lower, upper, label) in enumerate(zip(effects, ci_lowers, ci_uppers, labels)):
            y_pos = n - i
            
            fig.add_trace(go.Scatter(
                x=[effect],
                y=[y_pos],
                mode="markers",
                marker=dict(size=8, color="steelblue"),
                showlegend=False,
                hovertemplate=f"<b>{label}</b><br>Effect: {effect:.3f}<extra></extra>"
            ))
            
            fig.add_trace(go.Scatter(
                x=[lower, upper],
                y=[y_pos, y_pos],
                mode="lines",
                line=dict(color="steelblue", width=2),
                showlegend=False,
                hoverinfo="skip"
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Pooled Effect",
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(1, n + 1)),
                ticktext=labels[::-1]
            ),
            template=self.theme,
            height=max(400, n * 25 + 100)
        )
        
        return fig
        return fig
    
    def create_bubble_plot(
        self,
        result: MetaAnalysisResult,
        title: str = "Bubble Plot: Effect Size vs Precision"
    ) -> go.Figure:
        """
        Create a bubble plot of Effect Size vs Inverse Standard Error.
        
        Args:
            result: MetaAnalysisResult
            title: Title for the plot
            
        Returns:
            Plotly Figure
        """
        studies = result.studies
        if not studies:
            return go.Figure()
            
        effects = [s.effect_size for s in studies]
        precisions = [1.0 / s.standard_error for s in studies]
        weights = [result.study_weights.get(s.study_id, 1.0) for s in studies]
        names = [s.study_name for s in studies]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=precisions,
            y=effects,
            mode='markers',
            marker=dict(
                size=[np.sqrt(w)*3 + 5 for w in weights],
                color='rgba(93, 164, 214, 0.6)',
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=names,
            hovertemplate="<b>%{text}</b><br>Precision: %{x:.2f}<br>Effect: %{y:.3f}<extra></extra>"
        ))
        
        # Add regression line (weighted fit)
        # Using simple linear regression
        if len(studies) > 2:
            coef = np.polyfit(precisions, effects, 1, w=weights)
            poly1d_fn = np.poly1d(coef)
            x_line = np.linspace(min(precisions), max(precisions), 100)
            
            fig.add_trace(go.Scatter(
                x=x_line,
                y=poly1d_fn(x_line),
                mode='lines',
                line=dict(color='firebrick', dash='dash'),
                name='Meta-Regression'
            ))
            
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Precision (1/SE)",
            yaxis_title="Effect Size",
            template=self.theme,
            height=500,
            showlegend=False
        )
        return fig

    def create_influence_plot(
        self,
        sensitivity_result: Any, # SensitivityResult type hint avoided to prevent circular import issues if checking types strict
        title: str = "Influence Diagnostics: Standardized Residuals vs Leverage"
    ) -> go.Figure:
        """
        Create an influence plot (Baujat-style or similar).
        
        Plots Standardized Residuals vs Leverage (Weight).
        
        Args:
            sensitivity_result: Result object from SensitivityAnalyzer
            title: Title for the plot
            
        Returns:
            Plotly Figure
        """
        if not sensitivity_result.influence:
            return go.Figure()
            
        study_ids = list(sensitivity_result.influence.keys())
        residuals = [sensitivity_result.influence[s]['std_residual'] for s in study_ids]
        leverages = [sensitivity_result.influence[s]['leverage'] for s in study_ids]
        cooks = [sensitivity_result.influence[s]['cooks_distance'] for s in study_ids]
        
        # Scale bubble size by Cook's Distance
        sizes = [min(50, max(10, c * 100 + 10)) for c in cooks]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=leverages,
            y=residuals,
            mode='markers+text',
            text=study_ids,
            textposition="top center",
            marker=dict(
                size=sizes,
                color=residuals,
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title="Std Residual"),
                line=dict(width=1, color='DarkSlateGrey')
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Leverage: %{x:.3f}<br>"
                "Std Residual: %{y:.3f}<br>"
                "Cook's D: %{marker.size:.1f}"
                "<extra></extra>"
            )
        ))
        
        # Add reference lines
        fig.add_shape(type="line", x0=0, x1=max(leverages)*1.1, y0=0, y1=0, line=dict(color="gray", dash="dash"))
        fig.add_shape(type="line", x0=0, x1=max(leverages)*1.1, y0=1.96, y1=1.96, line=dict(color="red", dash="dot"))
        fig.add_shape(type="line", x0=0, x1=max(leverages)*1.1, y0=-1.96, y1=-1.96, line=dict(color="red", dash="dot"))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Leverage (Study Weight)",
            yaxis_title="Standardized Residual",
            template=self.theme,
            height=500
        )
        return fig
