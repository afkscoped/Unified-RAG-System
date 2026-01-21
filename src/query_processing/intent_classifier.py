"""
[ELITE ARCHITECTURE] intent_classifier.py
Heuristic and LLM-based Query Intent Classification.
"""

from enum import Enum
from loguru import logger
from langchain_core.prompts import PromptTemplate

class QueryIntent(Enum):
    FACT_RETRIEVAL = "fact_retrieval"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    CROSS_DOCUMENT = "cross_document"
    UNKNOWN = "unknown"

class IntentClassifier:
    """
    Innovation: Strategic Routing.
    Detects user intent to optimize retrieval settings (e.g. k=20 for summary vs k=5 for fact).
    """
    
    def __init__(self, llm_router):
        self.llm = llm_router
        self.intent_prompt = PromptTemplate.from_template("""
        Classify the user's query into one of these categories:
        - fact_retrieval: Specific details, numbers, or dates.
        - summarization: General overview or high-level conclusions.
        - comparison: Contrasting two or more entities/results.
        - cross_document: Requires synthesis across multiple reports.
        
        QUERY: {question}
        
        CATEGORY (Output ONLY the category name):
        """)

    def classify(self, query: str) -> QueryIntent:
        """Determines the tactical approach for the query."""
        try:
            logger.debug(f"Classifying intent for: {query[:40]}...")
            res = self.llm.generate(
                prompt=self.intent_prompt.format(question=query),
                system_prompt="You are a query classifier. Output ONLY the category name."
            ).strip().lower()
            
            for intent in QueryIntent:
                if intent.value in res:
                    return intent
            return QueryIntent.FACT_RETRIEVAL
        except Exception:
            return QueryIntent.FACT_RETRIEVAL

if __name__ == "__main__":
    print("Intent Classifier Standby.")
