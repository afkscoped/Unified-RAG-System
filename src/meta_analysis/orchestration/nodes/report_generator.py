"""
Report Generator Node

Generates comprehensive meta-analysis reports.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger


def generate_report_node(
    meta_result: Dict[str, Any],
    bias_result: Optional[Dict] = None,
    sensitivity_result: Optional[Dict] = None,
    recommendations: Optional[List[Dict]] = None,
    format: str = "markdown"
) -> Dict[str, Any]:
    """
    Generate meta-analysis report.
    
    Args:
        meta_result: Meta-analysis results
        bias_result: Publication bias results
        sensitivity_result: Sensitivity analysis results
        recommendations: List of recommendations
        format: Output format ("markdown", "html", "pdf")
        
    Returns:
        Report generation results
    """
    result = {
        "report": "",
        "format": format,
        "sections": [],
        "export_ready": False
    }
    
    try:
        if format == "markdown":
            report = _generate_markdown_report(
                meta_result, bias_result, sensitivity_result, recommendations
            )
        else:
            report = _generate_markdown_report(
                meta_result, bias_result, sensitivity_result, recommendations
            )
        
        result["report"] = report
        result["export_ready"] = True
        result["sections"] = [
            "title", "summary", "methods", "results", 
            "heterogeneity", "publication_bias", "sensitivity",
            "recommendations", "limitations"
        ]
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Report generation failed: {e}")
    
    return result


def _generate_markdown_report(
    meta_result: Dict,
    bias_result: Optional[Dict],
    sensitivity_result: Optional[Dict],
    recommendations: Optional[List[Dict]]
) -> str:
    """Generate markdown format report."""
    
    lines = []
    
    # Title
    lines.append("# Meta-Analysis Report")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    
    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    
    if meta_result:
        effect = meta_result.get("pooled_effect", 0)
        ci = meta_result.get("confidence_interval", {})
        p_value = meta_result.get("p_value", 1)
        n_studies = meta_result.get("n_studies", 0)
        i2 = meta_result.get("heterogeneity", {}).get("I2", 0)
        
        sig_text = "statistically significant" if p_value < 0.05 else "not statistically significant"
        direction = "positive" if effect > 0 else "negative"
        
        lines.append(
            f"This meta-analysis of **{n_studies} studies** found a {direction} pooled effect "
            f"of **{effect:.3f}** (95% CI: {ci.get('lower', 0):.3f} to {ci.get('upper', 0):.3f}), "
            f"which is {sig_text} (p = {p_value:.4f}). "
            f"Heterogeneity was {'high' if i2 > 50 else 'moderate' if i2 > 25 else 'low'} (I² = {i2:.1f}%)."
        )
    lines.append("")
    
    # Methods
    lines.append("## Methods")
    lines.append("")
    if meta_result:
        model = meta_result.get("model_type", "random effects")
        lines.append(f"- **Model:** {model.title()} model")
        lines.append(f"- **Effect Size:** Log odds ratio")
        lines.append(f"- **Confidence Level:** 95%")
    lines.append("")
    
    # Results
    lines.append("## Results")
    lines.append("")
    
    if meta_result:
        lines.append("### Pooled Effect")
        lines.append("")
        lines.append("| Statistic | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Pooled Effect | {meta_result.get('pooled_effect', 0):.4f} |")
        lines.append(f"| Standard Error | {meta_result.get('standard_error', 0):.4f} |")
        lines.append(f"| 95% CI Lower | {meta_result.get('confidence_interval', {}).get('lower', 0):.4f} |")
        lines.append(f"| 95% CI Upper | {meta_result.get('confidence_interval', {}).get('upper', 0):.4f} |")
        lines.append(f"| Z-value | {meta_result.get('z_value', 0):.2f} |")
        lines.append(f"| P-value | {meta_result.get('p_value', 1):.4f} |")
        lines.append("")
        
        # Heterogeneity
        lines.append("### Heterogeneity")
        lines.append("")
        het = meta_result.get("heterogeneity", {})
        lines.append("| Statistic | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Q | {het.get('Q', 0):.2f} |")
        lines.append(f"| Q p-value | {het.get('Q_pvalue', 1):.4f} |")
        lines.append(f"| I² | {het.get('I2', 0):.1f}% |")
        lines.append(f"| τ² | {het.get('tau2', 0):.4f} |")
        lines.append("")
    
    # Publication Bias
    if bias_result:
        lines.append("## Publication Bias Assessment")
        lines.append("")
        
        eggers = bias_result.get("eggers_test", {})
        if eggers:
            lines.append(f"- **Egger's test:** p = {eggers.get('p_value', 1):.3f}")
        
        if bias_result.get("bias_detected"):
            lines.append(f"- **Bias severity:** {bias_result.get('bias_severity', 'unknown')}")
        
        trim_fill = bias_result.get("trim_and_fill", {})
        if trim_fill:
            k0 = trim_fill.get("k0", 0)
            if k0 > 0:
                lines.append(f"- **Trim-and-fill:** {k0} imputed studies")
                lines.append(f"- **Adjusted effect:** {trim_fill.get('adjusted_effect', 0):.4f}")
        
        lines.append("")
        lines.append(f"*{bias_result.get('interpretation', '')}*")
        lines.append("")
    
    # Sensitivity Analysis
    if sensitivity_result:
        lines.append("## Sensitivity Analysis")
        lines.append("")
        
        if sensitivity_result.get("robust"):
            lines.append("✓ Results are robust to removal of individual studies.")
        else:
            lines.append("⚠️ Results may be sensitive to individual studies.")
        
        influential = sensitivity_result.get("influential_studies", [])
        if influential:
            lines.append(f"\n**Influential studies:** {', '.join(influential)}")
        
        lines.append("")
    
    # Recommendations
    if recommendations:
        lines.append("## Recommendations")
        lines.append("")
        
        for i, rec in enumerate(recommendations[:5], 1):
            priority = rec.get("priority", "medium").upper()
            title = rec.get("title", "Recommendation")
            message = rec.get("message", "")
            
            lines.append(f"### {i}. {title}")
            lines.append(f"**Priority:** {priority}")
            lines.append("")
            lines.append(message)
            lines.append("")
    
    # Limitations
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Results should be interpreted in the context of original study designs")
    lines.append("- Heterogeneity may limit generalizability of pooled estimates")
    lines.append("- Publication bias assessment has limited power with few studies")
    lines.append("")
    
    return "\n".join(lines)


def export_to_pdf(markdown_content: str, output_path: str) -> bool:
    """
    Export markdown report to PDF.
    
    Note: Requires additional libraries (markdown, weasyprint)
    
    Args:
        markdown_content: Markdown string
        output_path: Path to save PDF
        
    Returns:
        Success status
    """
    try:
        # This is a placeholder - PDF export would require additional dependencies
        logger.warning("PDF export requires additional dependencies (markdown, weasyprint)")
        return False
    except Exception as e:
        logger.error(f"PDF export failed: {e}")
        return False
