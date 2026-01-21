"""
[ELITE ARCHITECTURE] graph_engine.py
Builds and visualizes a dynamic Knowledge Graph from research corpora.
"""

import networkx as nx
from typing import List, Dict, Any
from loguru import logger
import spacy

class KnowledgeGraphEngine:
    """
    Innovation: Relationship Mapping.
    Automatically extracts entities and semantic links between document chunks 
    to visualize the 'Intellectual Landscape' of the project.
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except:
            logger.warning(f"Spacy model {spacy_model} not found. Knowledge Graph will be limited.")
            self.nlp = None
            
        self.graph = nx.MultiDiGraph()

    def extract_and_add(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        """
        Extracts entities and links them to the document node.
        """
        if not self.nlp: return
        
        # Add Document Node
        self.graph.add_node(doc_id, type="document", name=metadata.get("source_file", doc_id))
        
        doc = self.nlp(content[:10000]) # Cap for performance
        
        # Extract Entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]:
                entity_name = ent.text.strip().lower()
                
                # Add Entity Node
                self.graph.add_node(entity_name, type="entity", label=ent.label_)
                
                # Add Relationship
                self.graph.add_edge(doc_id, entity_name, relationship="mentions")

    def get_graph_data(self) -> Dict[str, List]:
        """
        Formats the graph for visualization (e.g., Cytoscape or Plotly).
        """
        nodes = []
        for n, d in self.graph.nodes(data=True):
            nodes.append({"id": n, "label": d.get("name", n), "type": d.get("type", "entity")})
            
        edges = []
        for u, v, d in self.graph.edges(data=True):
            edges.append({"source": u, "target": v, "label": d.get("relationship", "link")})
            
        return {"nodes": nodes, "edges": edges}

    def clear(self):
        self.graph.clear()

if __name__ == "__main__":
    kg = KnowledgeGraphEngine()
    print("Knowledge Graph Engine Standby.")
