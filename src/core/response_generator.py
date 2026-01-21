"""
Persona-Driven Response Generator
Formats responses according to persona templates
"""

from typing import List, Dict, Optional


class ResponseGenerator:
    """Generate persona-specific responses from search results"""
    
    def generate_response(
        self,
        query: str,
        search_results: List,
        persona
    ) -> str:
        """Generate formatted response based on persona template"""
        from src.core.persona_engine import PersonaType
        
        facts = self._extract_facts(search_results)
        
        if not facts:
            return f"No relevant information found for: {query}"
        
        if persona.type == PersonaType.SCIENTIST:
            return self._generate_scientific_response(query, facts, persona)
        elif persona.type == PersonaType.CREATIVE:
            return self._generate_creative_response(query, facts, persona)
        elif persona.type == PersonaType.ANALYST:
            return self._generate_analyst_response(query, facts, persona)
        elif persona.type == PersonaType.EDUCATOR:
            return self._generate_educator_response(query, facts, persona)
        elif persona.type == PersonaType.CRITIC:
            return self._generate_critical_response(query, facts, persona)
        elif persona.type == PersonaType.SYNTHESIZER:
            return self._generate_synthesis_response(query, facts, persona)
        
        return self._generate_default_response(query, facts)
    
    def _extract_facts(self, results: List) -> List[Dict]:
        """Extract key facts from search results"""
        facts = []
        for idx, result in enumerate(results[:10], 1):
            if hasattr(result, 'content'):
                facts.append({
                    'content': result.content,
                    'source_id': idx,
                    'score': result.score if hasattr(result, 'score') else 0,
                    'metadata': result.metadata if hasattr(result, 'metadata') else {}
                })
            elif isinstance(result, dict):
                facts.append({
                    'content': result.get('content', ''),
                    'source_id': idx,
                    'score': result.get('score', 0),
                    'metadata': result.get('metadata', {})
                })
        return facts
    
    def _generate_scientific_response(self, query: str, facts: List[Dict], persona) -> str:
        """Generate evidence-based scientific response"""
        response = f"## {query}\n\n"
        
        # Direct answer first
        response += "### Direct Answer\n\n"
        response += f"Based on the available evidence: {facts[0]['content'][:300]}... [1]\n\n"
        
        # Supporting evidence
        response += "### Supporting Evidence\n\n"
        for i, fact in enumerate(facts[1:4], 2):
            response += f"**[{fact['source_id']}]** {fact['content'][:200]}...\n\n"
        
        # Limitations
        response += "### Limitations\n\n"
        response += "- Evidence based on available documents\n"
        response += "- Conclusions should be verified with primary sources\n"
        response += "- Data may not represent complete picture\n\n"
        
        # Citations
        response += "### References\n\n"
        for fact in facts[:5]:
            source = fact['metadata'].get('source', f"Source {fact['source_id']}")
            response += f"[{fact['source_id']}] {source} (Score: {fact['score']:.3f})\n"
        
        return response
    
    def _generate_creative_response(self, query: str, facts: List[Dict], persona) -> str:
        """Generate narrative-driven creative response"""
        response = f"# ✨ {query}\n\n"
        
        # Engaging hook
        response += "*Imagine if this concept were a landscape...*\n\n"
        
        # Narrative exploration
        response += "## The Journey\n\n"
        response += f"{facts[0]['content'][:400]}...\n\n"
        
        # Connections
        response += "## Connecting the Dots\n\n"
        if len(facts) > 1:
            response += f"🔗 {facts[1]['content'][:200]}...\n\n"
        
        # Analogies
        response += "## Think of it This Way\n\n"
        response += "This is like a river of information - different streams (sources) "
        response += "converge into a deeper understanding, each bringing unique perspectives.\n\n"
        
        # Insight
        if len(facts) > 2:
            response += "## Unexpected Connections\n\n"
            response += f"💡 {facts[2]['content'][:200]}...\n\n"
        
        return response
    
    def _generate_analyst_response(self, query: str, facts: List[Dict], persona) -> str:
        """Generate executive summary style response"""
        response = f"# 📊 {query}\n\n"
        
        # Executive Summary
        response += "## Executive Summary\n\n"
        response += f"{facts[0]['content'][:300]}...\n\n"
        
        # Key Findings
        response += "## Key Findings\n\n"
        for i, fact in enumerate(facts[:4], 1):
            response += f"- **Finding {i}:** {fact['content'][:150]}...\n"
        
        response += "\n"
        
        # Recommendations
        response += "## Recommendations\n\n"
        response += "1. **Immediate:** Review primary sources for detailed context\n"
        response += "2. **Short-term:** Cross-reference findings with external data\n"
        response += "3. **Strategic:** Monitor for updates and changes in this area\n\n"
        
        # Confidence Metrics
        response += "## Confidence Metrics\n\n"
        avg_score = sum(f['score'] for f in facts[:5]) / min(5, len(facts))
        response += f"- **Source Quality:** {avg_score:.1%}\n"
        response += f"- **Evidence Strength:** {len(facts)} supporting documents\n"
        response += f"- **Data Freshness:** Based on indexed content\n"
        
        return response
    
    def _generate_educator_response(self, query: str, facts: List[Dict], persona) -> str:
        """Generate layered educational response"""
        response = f"# 📚 Understanding: {query}\n\n"
        
        # Simple explanation
        response += "## 🎯 Simple Explanation\n\n"
        response += "Let's start with the basics:\n\n"
        response += f"> {facts[0]['content'][:250]}...\n\n"
        
        # Building complexity
        response += "## 📖 Going Deeper\n\n"
        response += "Now that we have the foundation, here's more detail:\n\n"
        if len(facts) > 1:
            response += f"{facts[1]['content'][:350]}...\n\n"
        
        # Examples
        response += "## 💡 Concrete Examples\n\n"
        for i, fact in enumerate(facts[2:4], 1):
            response += f"**Example {i}:** {fact['content'][:150]}...\n\n"
        
        # Reflection questions
        response += "## 🤔 Think About It\n\n"
        topic = query.lower().replace("what is", "").replace("how does", "").strip()
        response += f"- How does {topic} relate to what you already know?\n"
        response += "- What questions do you still have?\n"
        response += "- Where could you apply this understanding?\n"
        response += "- What would you teach someone else first?\n"
        
        return response
    
    def _generate_critical_response(self, query: str, facts: List[Dict], persona) -> str:
        """Generate critical analysis response"""
        response = f"# 🔍 Critical Analysis: {query}\n\n"
        
        # Main claim
        response += "## The Claim\n\n"
        response += f"{facts[0]['content'][:300]}...\n\n"
        
        # Strengths
        response += "## Strengths of the Evidence\n\n"
        response += f"✅ {facts[0]['content'][:150]}...\n\n"
        
        # Weaknesses
        response += "## Potential Weaknesses\n\n"
        response += "⚠️ Limited to available indexed sources\n"
        response += "⚠️ May not represent complete picture\n"
        response += "⚠️ Temporal scope may be limited\n\n"
        
        # Alternative perspectives
        if len(facts) > 1:
            response += "## Alternative Perspectives\n\n"
            response += f"However, consider: {facts[1]['content'][:200]}...\n\n"
        
        # What's missing
        response += "## What's Missing?\n\n"
        response += "- Primary research validation\n"
        response += "- Longitudinal data and trends\n"
        response += "- Broader contextual factors\n"
        response += "- Potential conflicting sources\n"
        
        return response
    
    def _generate_synthesis_response(self, query: str, facts: List[Dict], persona) -> str:
        """Generate integrative synthesis response"""
        response = f"# 🌐 Synthesis: {query}\n\n"
        
        # Big picture
        response += "## The Big Picture\n\n"
        response += f"Looking across multiple sources, this emerges as a multi-faceted concept:\n\n"
        response += f"{facts[0]['content'][:250]}...\n\n"
        
        # Multiple perspectives
        response += "## Integrating Perspectives\n\n"
        for i, fact in enumerate(facts[:3], 1):
            response += f"### Perspective {i}\n\n"
            response += f"{fact['content'][:200]}...\n\n"
        
        # Conceptual framework
        response += "## Conceptual Framework\n\n"
        response += "```\n"
        topic = query.split()[-1] if query else "Concept"
        response += f"    {topic.capitalize()}\n"
        response += "    ├── Core Concepts\n"
        response += "    ├── Applications\n"
        response += "    ├── Implications\n"
        response += "    └── Future Directions\n"
        response += "```\n\n"
        
        # Key connections
        response += "## Key Connections\n\n"
        response += "This concept connects to broader themes of understanding, "
        response += "application, and evolution. The sources reveal both "
        response += "convergence and divergence in perspectives.\n"
        
        return response
    
    def _generate_default_response(self, query: str, facts: List[Dict]) -> str:
        """Fallback response generation"""
        response = f"## {query}\n\n"
        response += "### Relevant Information\n\n"
        for i, fact in enumerate(facts[:5], 1):
            response += f"**[{i}]** {fact['content'][:250]}...\n\n"
        return response
