"""
Analysis Prompts

LLM prompts for each stage of the meta-analysis pipeline.
"""

# Query Parser Prompt
QUERY_PARSER_PROMPT = """You are a query parser for A/B test meta-analysis.
Parse the user's query and extract:
1. Analysis type requested (meta-analysis, publication bias check, sensitivity analysis)
2. Filters (date range, metric type, minimum sample size)
3. Model preference (fixed effects, random effects, auto)
4. Specific questions or concerns

Output as JSON with keys: analysis_types, filters, model, questions

User Query: {query}
"""

# Interpretation Prompt
INTERPRETATION_PROMPT = """You are explaining meta-analysis results to a non-statistician.

**Results:**
- Pooled effect: {pooled_effect}
- 95% CI: ({ci_lower}, {ci_upper})
- P-value: {p_value}
- I² (heterogeneity): {i2}%
- Number of studies: {n_studies}
- Model: {model_type}

Please explain:
1. What the pooled effect means in practical terms
2. Whether the result is statistically significant
3. Whether studies are consistent with each other
4. Key limitations to consider

Use clear, accessible language without jargon.
"""

# Recommendation Prompt
RECOMMENDATION_PROMPT = """Based on the meta-analysis results, provide actionable recommendations.

**Context:**
- Pooled effect: {pooled_effect}
- Significance: {significance}
- Heterogeneity (I²): {i2}%
- Publication bias: {bias_status}
- Sensitivity: {sensitivity_status}

Provide 3-5 prioritized recommendations covering:
1. Whether to trust the pooled estimate
2. Next steps for analysis or research
3. Practical implications for decision-making

Format each recommendation with:
- Priority (High/Medium/Low)
- Recommendation
- Rationale
"""

# Report Generation Prompt
REPORT_GENERATION_PROMPT = """Generate a comprehensive meta-analysis report.

**Data:**
{analysis_data}

Create a report with sections:
1. Executive Summary (2-3 sentences)
2. Methods (model used, heterogeneity handling)
3. Results (pooled effect, CI, significance)
4. Quality Assessment (bias, sensitivity)
5. Conclusions and Recommendations

Use markdown formatting. Be precise but accessible.
"""

# Heterogeneity Explanation Prompt
HETEROGENEITY_EXPLANATION_PROMPT = """Explain the heterogeneity findings in this meta-analysis.

**Statistics:**
- I² = {i2}%
- τ² = {tau2}
- Q = {q} (p = {q_pvalue})

Explain:
1. What these statistics mean
2. Whether heterogeneity is a concern
3. Implications for interpreting the pooled effect
4. Suggestions for handling heterogeneity
"""

# Publication Bias Explanation Prompt
PUBLICATION_BIAS_PROMPT = """Explain the publication bias assessment results.

**Findings:**
- Egger's test p-value: {eggers_p}
- Trim-and-fill: {k0} imputed studies
- Original effect: {original_effect}
- Adjusted effect: {adjusted_effect}
- Failsafe N: {failsafe_n}

Explain:
1. Whether publication bias is likely
2. Impact on the conclusions
3. Whether the adjusted estimate should be used
4. Caveats about the assessment
"""

# Simpson's Paradox Explanation Prompt
SIMPSONS_PARADOX_PROMPT = """Explain the Simpson's Paradox analysis results.

**Findings:**
- Paradox detected: {detected}
- Overall effect: {overall_effect}
- Subgroup effects: {subgroup_effects}

If paradox detected, explain:
1. What this means for the meta-analysis
2. Which subgroup results to trust
3. Possible confounders
4. Recommendations for reporting

If not detected, confirm that subgroup and overall effects are consistent.
"""


def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with provided values.
    
    Args:
        template: Prompt template string
        **kwargs: Values to substitute
        
    Returns:
        Formatted prompt
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        # Return template with missing keys noted
        return template.replace("{" + str(e).strip("'") + "}", "[MISSING]")
