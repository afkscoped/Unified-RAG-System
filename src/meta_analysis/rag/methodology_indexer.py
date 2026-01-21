"""
Methodology Indexer

Indexes academic papers and methodology documents for RAG retrieval.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger


# Built-in methodology references (no external files needed)
METHODOLOGY_REFERENCES = {
    "dersimonian_laird_1986": {
        "title": "Meta-analysis in clinical trials",
        "authors": "DerSimonian R, Laird N",
        "year": 1986,
        "journal": "Controlled Clinical Trials",
        "key_concepts": ["random effects", "tau-squared", "between-study variance"],
        "summary": """
        The DerSimonian-Laird method is the most widely used approach for random-effects 
        meta-analysis. It estimates the between-study variance (τ²) using a method of moments 
        approach based on Cochran's Q statistic. The random-effects model assumes that the 
        true effect sizes vary across studies and come from a distribution of effects.
        
        The pooled estimate is calculated as a weighted average where weights account for 
        both within-study variance and between-study variance: w_i = 1/(σ_i² + τ²).
        
        When to use: When studies are heterogeneous (I² > 25%) or when you want to 
        generalize beyond the specific studies included.
        """
    },
    "higgins_thompson_2002": {
        "title": "Quantifying heterogeneity in a meta-analysis",
        "authors": "Higgins JPT, Thompson SG",
        "year": 2002,
        "journal": "Statistics in Medicine",
        "key_concepts": ["I-squared", "heterogeneity", "H statistic"],
        "summary": """
        This paper introduced the I² statistic as a measure of heterogeneity in meta-analysis.
        I² describes the percentage of total variation across studies that is due to 
        heterogeneity rather than sampling error.
        
        Interpretation guidelines:
        - I² = 0-25%: Low heterogeneity
        - I² = 25-50%: Moderate heterogeneity  
        - I² = 50-75%: Substantial heterogeneity
        - I² = 75-100%: Considerable heterogeneity
        
        Unlike Q, I² does not depend on the number of studies and can be compared 
        across meta-analyses. However, I² can be imprecise with few studies.
        """
    },
    "egger_1997": {
        "title": "Bias in meta-analysis detected by a simple, graphical test",
        "authors": "Egger M, Smith GD, Schneider M, Minder C",
        "year": 1997,
        "journal": "BMJ",
        "key_concepts": ["publication bias", "funnel plot", "asymmetry test"],
        "summary": """
        Egger's test is a regression-based method for detecting funnel plot asymmetry,
        which may indicate publication bias. The test regresses the standardized effect
        (effect/SE) on precision (1/SE).
        
        A significant intercept suggests that smaller studies show systematically 
        different effects than larger studies - often because small negative studies 
        are not published (publication bias).
        
        Limitations: The test has low power with fewer than 10 studies and can give
        false positives when there is true heterogeneity. Should be interpreted 
        alongside visual inspection of funnel plots.
        """
    },
    "duval_tweedie_2000": {
        "title": "Trim and fill: A simple funnel-plot-based method",
        "authors": "Duval S, Tweedie R",
        "year": 2000,
        "journal": "Biometrics",
        "key_concepts": ["trim and fill", "bias adjustment", "imputation"],
        "summary": """
        The trim-and-fill method estimates the number of 'missing' studies due to 
        publication bias and provides a bias-adjusted pooled effect estimate.
        
        The algorithm:
        1. Estimate the number of asymmetric (extreme) studies
        2. Trim these studies and re-estimate the pooled effect
        3. Fill in the 'missing' mirror-image studies
        4. Re-calculate the pooled effect with all studies
        
        The adjusted estimate is typically smaller in magnitude than the original
        if publication bias has inflated the observed effect.
        
        Caution: This method assumes asymmetry is due to publication bias, but 
        other factors can cause asymmetry.
        """
    },
    "cochran_1954": {
        "title": "The combination of estimates from different experiments",
        "authors": "Cochran WG",
        "year": 1954,
        "journal": "Biometrics",
        "key_concepts": ["Q statistic", "homogeneity test", "fixed effects"],
        "summary": """
        Cochran's Q statistic tests whether observed variation in effect sizes 
        exceeds what would be expected from sampling error alone.
        
        Q = Σw_i(θ_i - θ̂)² where w_i are study weights and θ̂ is the pooled estimate.
        
        Under homogeneity, Q follows a chi-squared distribution with k-1 degrees 
        of freedom (k = number of studies).
        
        A significant Q indicates heterogeneity, but:
        - The test has low power with few studies
        - The test may be overpowered with many studies
        Therefore, I² is often preferred as a descriptive measure.
        """
    },
    "simpsons_paradox": {
        "title": "Simpson's Paradox in Meta-Analysis",
        "authors": "Various",
        "year": "N/A",
        "journal": "N/A",
        "key_concepts": ["confounding", "ecological fallacy", "subgroup analysis"],
        "summary": """
        Simpson's Paradox occurs when an association observed in aggregate data 
        reverses direction or disappears when analyzed within subgroups.
        
        In meta-analysis, this can happen when:
        1. A confounding variable is associated with both the exposure and outcome
        2. The confounder distribution varies across studies
        3. The aggregate analysis does not account for this variation
        
        Detection: Compare overall pooled effect with subgroup-specific effects.
        If directions consistently differ, the stratification variable is likely
        a confounder.
        
        Resolution: Report stratified results, use meta-regression, or adjust
        for the confounder at the study level.
        """
    }
}


class MethodologyIndexer:
    """
    Indexes and retrieves methodology papers for RAG.
    
    Uses built-in references and can optionally index external documents.
    """
    
    def __init__(self, docs_dir: Optional[str] = None):
        """
        Initialize indexer.
        
        Args:
            docs_dir: Optional directory with additional methodology documents
        """
        self.docs_dir = Path(docs_dir) if docs_dir else None
        self.references = METHODOLOGY_REFERENCES.copy()
        self._indexed_docs = {}
    
    def get_reference(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a specific methodology reference."""
        return self.references.get(key)
    
    def search_references(
        self, 
        query: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search references by query string.
        
        Args:
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of matching references
        """
        query_lower = query.lower()
        results = []
        
        for key, ref in self.references.items():
            score = 0
            
            # Check title
            if query_lower in ref["title"].lower():
                score += 3
            
            # Check key concepts
            for concept in ref.get("key_concepts", []):
                if query_lower in concept.lower():
                    score += 2
            
            # Check summary
            if query_lower in ref.get("summary", "").lower():
                score += 1
            
            if score > 0:
                results.append({"ref_key": key, "score": score, **ref})
        
        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def get_summary_for_concept(self, concept: str) -> str:
        """
        Get relevant methodology summary for a statistical concept.
        
        Args:
            concept: Statistical concept (e.g., "I2", "random effects")
            
        Returns:
            Relevant methodology text
        """
        concept_lower = concept.lower().replace("²", "2").replace("-", " ")
        
        # Map concepts to references
        concept_map = {
            "i2": "higgins_thompson_2002",
            "i squared": "higgins_thompson_2002",
            "heterogeneity": "higgins_thompson_2002",
            "tau2": "dersimonian_laird_1986",
            "tau squared": "dersimonian_laird_1986",
            "random effects": "dersimonian_laird_1986",
            "dersimonian": "dersimonian_laird_1986",
            "publication bias": "egger_1997",
            "egger": "egger_1997",
            "funnel plot": "egger_1997",
            "trim and fill": "duval_tweedie_2000",
            "trim fill": "duval_tweedie_2000",
            "q statistic": "cochran_1954",
            "cochran": "cochran_1954",
            "simpson": "simpsons_paradox",
            "paradox": "simpsons_paradox"
        }
        
        for key, ref_id in concept_map.items():
            if key in concept_lower:
                ref = self.references.get(ref_id)
                if ref:
                    return ref.get("summary", "")
        
        return ""
    
    def list_all_references(self) -> List[Dict[str, Any]]:
        """List all available references."""
        return [
            {"key": key, "title": ref["title"], "year": ref["year"]}
            for key, ref in self.references.items()
        ]
