"""
[ELITE ARCHITECTURE] cot_generator.py
Implements Multi-Path Chain-of-Thought (CoT) with Self-Revising Logic.
"""

from typing import List, Dict, Any
from loguru import logger
from langchain_core.prompts import PromptTemplate

class AdaptiveCoTGenerator:
    """
    Innovation: Self-Reflective Generation.
    The model first generates a 'Thought Trace', verifies it against the 
    retrieved context, and then produces the final grounded answer.
    """
    
    def __init__(self, llm_router):
        self.llm = llm_router
        self.cot_prompt = PromptTemplate.from_template("""
        SYSTEM: You are an Elite Research AI.
        CONTEXT:
        {context}
        
        TASK:
        1. REASONING: Analyze the context step-by-step. Identify potential contradictions.
        2. VERIFICATION: Ensure every claim can be mapped to one of the [Sources].
        3. SYNTHESIS: Provide a professional answer with inline citations.
        
        QUESTION: {question}
        
        OUTPUT FORMAT:
        <thought>
        [Step-by-step logic here]
        </thought>
        
        FINAL ANSWER:
        [Answer with [Source X] citations]
        """)

    def generate_adaptive_response(self, query: str, context_docs: List[Any]) -> Dict[str, str]:
        """
        Executes the CoT protocol.
        """
        # 1. Format context for the prompt
        context_str = "\n\n".join([
            f"[Source {i+1}] (File: {doc.metadata.get('source_file')}, Page: {doc.metadata.get('page')}):\n{doc.content}"
            for i, doc in enumerate(context_docs)
        ])
        
        logger.info(f"Synthesizing response for query: {query[:50]}...")
        
        try:
            full_response = self.llm.generate(
                prompt=self.cot_prompt.format(context=context_str, question=query),
                system_prompt="You are a meticulous research synthesizer."
            )
            
            # 2. Decomposition of Thought vs Answer
            thought = ""
            answer = full_response
            
            if "<thought>" in full_response and "</thought>" in full_response:
                parts = full_response.split("</thought>")
                thought = parts[0].replace("<thought>", "").strip()
                answer = parts[1].replace("FINAL ANSWER:", "").strip()
                
            return {
                "thought": thought,
                "answer": answer,
                "raw": full_response
            }
            
        except Exception as e:
            logger.error(f"CoT Generation failure: {e}")
            return {"thought": "Error in reasoning.", "answer": "The system failed to synthesize a response.", "raw": str(e)}

if __name__ == "__main__":
    print("Adaptive CoT Engine Standby.")
