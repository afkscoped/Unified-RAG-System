"""
[ELITE ARCHITECTURE] predictive_metrics.py
Document and Query Profiling for Research Management.
"""

from typing import Dict, Any
import re
import math
from loguru import logger

class DocumentProfiler:
    """
    Innovation: Complexity Metrics.
    Calculates readability and structure scores to predict ingestion 
    difficulty and summarization fidelity.
    """
    
    @staticmethod
    def calculate_metrics(text: str) -> Dict[str, Any]:
        """
        Computes readability and lexical scores.
        """
        words = re.findall(r'\w+', text)
        sentences = [s for s in text.split('.') if len(s.strip()) > 0]
        
        if not words or not sentences:
            return {
                "reading_grade_level": 0,
                "lexical_diversity": 0,
                "complexity_label": "Unknown"
            }
            
        avg_word_per_sent = len(words) / len(sentences)
        syllables = sum([DocumentProfiler._count_syllables(w) for w in words])
        avg_syl_per_word = syllables / len(words)
        
        # Flesch-Kincaid Grade Level (Approximate)
        fk_grade = 0.39 * avg_word_per_sent + 11.8 * avg_syl_per_word - 15.59
        
        # Lexical Richness (Type-Token Ratio)
        unique_words = len(set(w.lower() for w in words))
        ttr = unique_words / len(words)
        
        return {
            "reading_grade_level": round(fk_grade, 1),
            "lexical_diversity": round(ttr, 3),
            "avg_sentence_length": round(avg_word_per_sent, 1),
            "complexity_label": "High" if fk_grade > 14 else ("Advanced" if fk_grade > 10 else "Baseline")
        }

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Heuristic syllable counter."""
        word = word.lower()
        if len(word) <= 3: return 1
        count = len(re.findall(r'[aeiouy]+', word))
        if word.endswith('e'): count -= 1
        return max(1, count)

class QueryEstimator:
    """Estimates the confidence of retrieval hits."""
    
    @staticmethod
    def estimate_difficulty(query: str, intent_label: str) -> str:
        words = query.split()
        if len(words) < 4: return "Incomplete"
        if intent_label == "cross_document": return "Complex"
        return "Standard"

if __name__ == "__main__":
    test_text = "The aerodynamic properties of carbon-fiber composites are essential for high-velocity aerospace applications."
    print(DocumentProfiler.calculate_metrics(test_text))
