"""
[ELITE ARCHITECTURE] hyde_generator.py
Hypothetical Document Embeddings for Query Expansion.
"""

from loguru import logger
from langchain_core.prompts import PromptTemplate

class HyDEGenerator:
    """
    Innovation: HyDE (Gao et al., 2022).
    Generates a 'fake' but semantically rich answer to the query.
    Searching for the hypothetical answer often finds better context than the question alone.
    """
    
    def __init__(self, llm_router):
        self.llm = llm_router
        self.hyde_prompt = PromptTemplate.from_template("""
        Please write a short (2-3 sentences) factual, scientific explanation that answers the following question.
        Focus on technical terms and core concepts.
        
        Question: {question}
        
        Answer (Hypothetical):
        """)

    def generate_pseudo_doc(self, query: str) -> str:
        """Transforms a user query into a hypothetical expert response for vector search."""
        try:
            logger.debug(f"Initiating HyDE expansion for: {query[:50]}...")
            pseudo_doc = self.llm.generate(
                prompt=self.hyde_prompt.format(question=query),
                system_prompt="You are a research assistant writing hypothetical baseline text."
            )
            # Combine original query with expansion
            return f"{query} \n {pseudo_doc}"
        except Exception as e:
            logger.error(f"HyDE generation failure: {e}")
            return query
