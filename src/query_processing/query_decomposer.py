"""
[ELITE ARCHITECTURE] query_decomposer.py
Decomposes complex queries into atomic research tasks.
"""

from typing import List
from loguru import logger
from langchain_core.prompts import PromptTemplate

class QueryDecomposer:
    """
    Innovation: Multi-Hop Reasoning.
    Breaks a single complex prompt into a 'Research Plan' of sub-queries.
    Enables the system to answer questions that require separate retrieval steps.
    """
    
    def __init__(self, llm_router):
        self.llm = llm_router
        self.decomposer_prompt = PromptTemplate.from_template("""
        If the following question is complex or multi-part, break it down into 2-3 logical sub-questions. 
        If the question is simple, return the original question.
        
        QUERY: {question}
        
        RESEARCH PLAN (Output a numbered list of sub-questions):
        """)

    def decompose(self, query: str) -> List[str]:
        """Generates a list of atomic queries."""
        try:
            logger.debug(f"Decomposing research query: {query[:40]}...")
            res = self.llm.generate(
                prompt=self.decomposer_prompt.format(question=query),
                system_prompt="You are a research analyst. Break down complex queries into simple sub-questions."
            )
            
            # Simple list parsing
            sub_queries = [q.strip() for q in res.split("\n") if q.strip() and (q[0].isdigit() or "?" in q)]
            
            # Remove numbering
            clean_queries = []
            for q in sub_queries:
                clean_q = "".join(q.split(". ")[1:]) if ". " in q else q
                if clean_q: clean_queries.append(clean_q)
                
            return clean_queries if clean_queries else [query]
        except Exception:
            return [query]

if __name__ == "__main__":
    print("Query Decomposer Ready.")
