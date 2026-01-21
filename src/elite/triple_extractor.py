"""
Triple Extractor: Extracts Subject-Verb-Object triples from text.
Designed to run in background during indexing.
"""

import threading
import json
import os
from typing import List, Dict
from loguru import logger

class TripleExtractor:
    """
    Background worker that extracts semantic triples (S-V-O) from ingested text.
    Uses the LLM to process text chunks and append results to a JSON line file.
    """
    
    def __init__(self, llm_router, output_path: str = "data/graphs/triples.jsonl"):
        self.llm = llm_router
        self.output_path = output_path
        
        # Ensure dir exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    def extract_async(self, text_chunks: List[str], metadata: Dict):
        """
        Starts the extraction in a daemon thread to prevent UI freezing.
        """
        thread = threading.Thread(
            target=self._process_chunks,
            args=(text_chunks, metadata)
        )
        thread.daemon = True
        thread.start()
        logger.info(f"Started background Triple Extraction for {len(text_chunks)} chunks.")
        
    def _process_chunks(self, chunks: List[str], metadata: Dict):
        """
        Worker function.
        """
        try:
            results = []
            filename = metadata.get('source_file', 'unknown')
            
            for i, chunk in enumerate(chunks):
                # We limit context size for speed
                process_text = chunk[:1000]
                
                prompt = f"""
                Extract core Subject-Verb-Object triples from the text below.
                Format: JSON List of lists: [["Subject", "Verb", "Object"], ...]
                Text: {process_text}
                
                Output ONLY Valid JSON.
                """
                
                try:
                    response = self.llm.generate(prompt, system_prompt="You are a precise Knowledge Graph extractor. Output JSON only.")
                    
                    # Clean response
                    response = response.strip()
                    if "```json" in response:
                        response = response.split("```json")[1].split("```")[0]
                    elif "{" not in response and "[" not in response:
                        continue # Skip bad output
                        
                    triples = json.loads(response)
                    
                    if isinstance(triples, list):
                         results.append({
                             "source": filename,
                             "chunk_id": i,
                             "triples": triples
                         })
                         
                except Exception as e:
                    logger.warning(f"Triple extraction chunk failure: {e}")
                    continue
            
            # Append to file
            if results:
                with open(self.output_path, "a", encoding="utf-8") as f:
                    for entry in results:
                        f.write(json.dumps(entry) + "\n")
                logger.success(f"Background Triple Extraction complete for {filename}: {len(results)} chunks processed.")
                
        except Exception as e:
            logger.error(f"Triple Extractor critical failure: {e}")
