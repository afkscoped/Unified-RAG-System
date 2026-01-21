"""
[ELITE ARCHITECTURE] layout_parser.py
Professional layout-aware document processor.
Optimized for academic papers and technical reports.
"""

import pdfplumber
from typing import List, Dict, Any
from loguru import logger

class LayoutParser:
    """
    Decomposes documents into structured components (Text, Tables, Headers).
    Innovation: Maintains spatial relationship for table extraction.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.enable_tables = config['preprocessing'].get('layout_aware', True)

    def parse_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts structured blocks from PDF using spatial heuristics.
        """
        elements = []
        logger.info(f"Layout Parsing Sequence: {file_path}")
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # 1. Table Extraction (Priority)
                    if self.enable_tables:
                        tables = page.extract_tables()
                        for table in tables:
                            if table:
                                # Convert table to Markdown format for LLM readability
                                table_str = self._table_to_markdown(table)
                                elements.append({
                                    "type": "table",
                                    "content": table_str,
                                    "metadata": {"page": page_num + 1}
                                })
                    
                    # 2. Text extraction with layout preservation
                    text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if text:
                        elements.append({
                            "type": "text",
                            "content": text,
                            "metadata": {"page": page_num + 1}
                        })
                        
            logger.debug(f"Parsed {len(elements)} structural elements from {file_path}")
            return elements
        except Exception as e:
            logger.error(f"Layout parsing failure: {e}")
            return []

    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """Converts raw table list of lists to GFM Markdown."""
        if not table: return ""
        # Filter out empty rows
        table = [row for row in table if any(cell for cell in row)]
        if not table: return ""
        
        headers = table[0]
        rows = table[1:]
        
        md = "| " + " | ".join([str(c).strip() if c else "" for c in headers]) + " |\n"
        md += "| " + " | ".join(["---" for _ in headers]) + " |\n"
        
        for row in rows:
            md += "| " + " | ".join([str(c).strip() if c else "" for c in row]) + " |\n"
            
        return md

if __name__ == "__main__":
    # Test stub
    parser = LayoutParser({"preprocessing": {"layout_aware": True}})
    print("Parser Standby.")
