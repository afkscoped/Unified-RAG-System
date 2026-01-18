"""
Narrative Entity Extractor

Extracts characters, locations, events, and relationships from story text
using SpaCy NER and dependency parsing.

Enhanced with:
- Pattern matching for narrative relationships
- Co-occurrence-based relationship inference
- Entity deduplication and merging
- Confidence scores
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from loguru import logger

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("SpaCy not available. Entity extraction will be limited.")


class NarrativeEntityExtractor:
    """
    Extracts story entities and relationships from narrative text.
    
    Uses SpaCy for NER and dependency parsing to identify:
    - Characters (PERSON entities)
    - Locations (GPE, LOC, FAC entities)
    - Events (verb-centered constructs)
    - Relationships (subject-verb-object patterns)
    
    Enhanced features:
    - Pattern-based relationship detection
    - Co-occurrence inference
    - Entity deduplication
    - Confidence scoring
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the entity extractor.
        
        Args:
            model_name: SpaCy model name to load
        """
        self.nlp = None
        self.model_name = model_name
        
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
                logger.info(f"SpaCy model '{model_name}' loaded successfully")
            except OSError:
                logger.warning(
                    f"SpaCy model '{model_name}' not found. "
                    f"Run: python -m spacy download {model_name}"
                )
        
        # Verb patterns for relationship classification
        self.conflict_verbs = {
            "fought", "attacked", "betrayed", "opposed", "killed", "defeated",
            "hated", "despised", "confronted", "challenged", "destroyed", "hurt",
            "threatened", "stabbed", "shot", "ambushed", "cursed"
        }
        self.alliance_verbs = {
            "helped", "joined", "allied", "befriended", "saved", "protected",
            "loved", "trusted", "supported", "assisted", "rescued", "embraced",
            "kissed", "hugged", "comforted", "accompanied"
        }
        self.family_verbs = {
            "married", "wed", "adopted", "fathered", "mothered", "parented"
        }
        self.fear_verbs = {
            "feared", "dreaded", "fled", "escaped", "hid", "cowered", "trembled"
        }
        self.action_verbs = {
            "said", "shouted", "whispered", "thought", "felt", "looked",
            "walked", "ran", "traveled", "arrived", "left", "discovered"
        }
        
        # Pattern templates for relationship detection
        self.relationship_patterns = [
            (r"(\w+)\s+and\s+(\w+)\s+were\s+(friends|allies|partners)", "ALLIES_WITH", 0.9),
            (r"(\w+)\s+and\s+(\w+)\s+were\s+(enemies|rivals)", "CONFLICTS_WITH", 0.9),
            (r"(\w+)\s+was\s+(\w+)'s\s+(brother|sister|father|mother|son|daughter)", "FAMILY", 0.95),
            (r"(\w+)\s+loved\s+(\w+)", "LOVES", 0.85),
            (r"(\w+)\s+hated\s+(\w+)", "CONFLICTS_WITH", 0.85),
            (r"(\w+)\s+trusted\s+(\w+)", "ALLIES_WITH", 0.7),
            (r"(\w+)\s+feared\s+(\w+)", "FEARS", 0.8),
            (r"(\w+)\s+worked\s+with\s+(\w+)", "ALLIES_WITH", 0.6),
            (r"(\w+)\s+fought\s+against\s+(\w+)", "CONFLICTS_WITH", 0.9),
            (r"(\w+)\s+went\s+to\s+(\w+)", "LOCATED_IN", 0.7),
            (r"(\w+)\s+arrived\s+at\s+(\w+)", "LOCATED_IN", 0.8),
            (r"(\w+)\s+lived\s+in\s+(\w+)", "LOCATED_IN", 0.9),
        ]
        
        # Entity name variants for deduplication
        self.entity_variants: Dict[str, Set[str]] = defaultdict(set)
        
    def extract_entities(
        self, 
        text: str, 
        chapter: int = 1
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract entities and relationships from text.
        
        Args:
            text: Story text to process
            chapter: Current chapter number
            
        Returns:
            Tuple of (entities, relationships)
        """
        entities = []
        relationships = []
        
        if not self.nlp:
            # Fallback: basic pattern matching
            return self._extract_basic(text, chapter)
        
        doc = self.nlp(text)
        
        # Track seen entities to avoid duplicates
        seen_entities = {}
        
        # Extract named entities with confidence
        for ent in doc.ents:
            entity_type = self._map_spacy_label(ent.label_)
            if not entity_type:
                continue
            
            # Deduplicate via canonical name
            canonical_name = self._get_canonical_name(ent.text)
            entity_id = self._make_entity_id(canonical_name, ent.label_)
            
            if entity_id in seen_entities:
                # Add mention variant
                seen_entities[entity_id]["mentions"].append(ent.text)
                continue
            
            # Calculate confidence based on entity type and context
            confidence = self._calculate_entity_confidence(ent, doc)
            
            entity_data = {
                "id": entity_id,
                "type": entity_type,
                "name": canonical_name,
                "mentions": [ent.text],
                "chapter": chapter,
                "confidence": confidence,
                "attributes": {}
            }
            seen_entities[entity_id] = entity_data
            entities.append(entity_data)
        
        # Extract relationships via multiple methods
        
        # 1. Dependency parsing
        for sent in doc.sents:
            sent_relationships = self._extract_sentence_relationships(sent, chapter)
            relationships.extend(sent_relationships)
        
        # 2. Pattern matching
        pattern_relationships = self._extract_pattern_relationships(text, chapter)
        relationships.extend(pattern_relationships)
        
        # 3. Co-occurrence inference
        cooccurrence_relationships = self._extract_cooccurrence_relationships(doc, chapter)
        relationships.extend(cooccurrence_relationships)
        
        # Deduplicate relationships
        relationships = self._deduplicate_relationships(relationships)
        
        logger.debug(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
        return entities, relationships
    
    def _extract_basic(
        self, 
        text: str, 
        chapter: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Basic entity extraction without SpaCy.
        Uses simple pattern matching for capitalized words.
        """
        entities = []
        relationships = []
        
        # Find capitalized words that might be names
        name_pattern = r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b'
        potential_names = set(re.findall(name_pattern, text))
        
        # Filter out common words
        common_words = {
            "The", "A", "An", "He", "She", "It", "They", "We", "I",
            "This", "That", "These", "Those", "What", "Where", "When",
            "Then", "Now", "Here", "There", "But", "And", "Or", "So"
        }
        
        for name in potential_names:
            if name not in common_words:
                entity_id = f"char_{name.lower().replace(' ', '_')}"
                entities.append({
                    "id": entity_id,
                    "type": "CHARACTER",
                    "name": name,
                    "mentions": [name],
                    "chapter": chapter,
                    "confidence": 0.5,
                    "attributes": {}
                })
        
        # Extract pattern-based relationships even without SpaCy
        relationships = self._extract_pattern_relationships(text, chapter)
        
        return entities, relationships
    
    def _get_canonical_name(self, name: str) -> str:
        """Get canonical (deduplicated) name for entity."""
        name_lower = name.lower()
        base_name = name.split()[0].lower() if name else ""
        
        # Check if this is a variant of a known name
        for canonical, variants in self.entity_variants.items():
            if name_lower in variants or base_name == canonical.split()[0].lower():
                return canonical
        
        # Add as new canonical name
        self.entity_variants[name].add(name_lower)
        return name
    
    def merge_entities(self, name1: str, name2: str) -> None:
        """Manually merge two entity names as the same entity."""
        canonical = name1 if len(name1) >= len(name2) else name2
        other = name2 if canonical == name1 else name1
        
        self.entity_variants[canonical].add(other.lower())
        self.entity_variants[canonical].add(name1.lower())
        self.entity_variants[canonical].add(name2.lower())
        
        # Remove other from being a canonical name
        if other in self.entity_variants:
            variants = self.entity_variants.pop(other)
            self.entity_variants[canonical].update(variants)
    
    def _calculate_entity_confidence(self, ent, doc) -> float:
        """Calculate confidence score for extracted entity."""
        confidence = 0.7  # Base confidence
        
        # Boost for PERSON entities
        if ent.label_ == "PERSON":
            confidence += 0.1
        
        # Boost for entities mentioned multiple times
        mentions = sum(1 for e in doc.ents if e.text.lower() == ent.text.lower())
        if mentions > 1:
            confidence += min(0.15, mentions * 0.03)
        
        # Boost for entities at sentence start (often important)
        if ent.start == ent.sent.start:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _make_entity_id(self, text: str, label: str) -> str:
        """Generate consistent entity ID."""
        prefix = {
            "PERSON": "char",
            "GPE": "loc",
            "LOC": "loc",
            "FAC": "loc",
            "ORG": "org",
            "EVENT": "event"
        }.get(label, "entity")
        
        clean_text = re.sub(r'[^a-z0-9]', '_', text.lower())
        return f"{prefix}_{clean_text}"
    
    def _map_spacy_label(self, label: str) -> Optional[str]:
        """Map SpaCy NER label to our entity types."""
        mapping = {
            "PERSON": "CHARACTER",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "FAC": "LOCATION",
            "ORG": "FACTION",
            "EVENT": "EVENT"
        }
        return mapping.get(label)
    
    def _extract_sentence_relationships(
        self, 
        sent, 
        chapter: int
    ) -> List[Dict]:
        """
        Extract relationships from a single sentence using dependency parsing.
        """
        relationships = []
        
        # Find subject-verb-object patterns
        for token in sent:
            if token.pos_ == "VERB":
                # Find subject
                subject = None
                subject_token = None
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subject = child.text
                        subject_token = child
                        break
                
                if not subject:
                    continue
                
                # Find object(s)
                objects = []
                for child in token.children:
                    if child.dep_ in ("dobj", "pobj", "attr", "dative"):
                        objects.append((child.text, child))
                    # Also check prepositional phrases
                    if child.dep_ == "prep":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                objects.append((grandchild.text, grandchild))
                
                # Classify relationship type
                rel_type = self._classify_relation(token.lemma_)
                
                # Calculate relationship strength
                strength = self._calculate_relationship_strength(token, subject_token)
                
                # Create relationships
                for obj_text, obj_token in objects:
                    # Only create if both look like named entities
                    if self._looks_like_entity(subject) and self._looks_like_entity(obj_text):
                        relationships.append({
                            "source": subject,
                            "source_id": f"char_{subject.lower().replace(' ', '_')}",
                            "target": obj_text,
                            "target_id": f"char_{obj_text.lower().replace(' ', '_')}",
                            "relation_type": rel_type,
                            "strength": strength,
                            "confidence": 0.7,
                            "chapter": chapter,
                            "context": sent.text[:100],
                            "verb": token.lemma_
                        })
        
        return relationships
    
    def _extract_pattern_relationships(self, text: str, chapter: int) -> List[Dict]:
        """Extract relationships using regex patterns."""
        relationships = []
        
        for pattern, rel_type, confidence in self.relationship_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    source = groups[0]
                    target = groups[1]
                    
                    if self._looks_like_entity(source) and self._looks_like_entity(target):
                        relationships.append({
                            "source": source,
                            "source_id": f"char_{source.lower().replace(' ', '_')}",
                            "target": target,
                            "target_id": f"char_{target.lower().replace(' ', '_')}",
                            "relation_type": rel_type,
                            "strength": 0.8,
                            "confidence": confidence,
                            "chapter": chapter,
                            "context": match.group(0),
                            "extraction_method": "pattern"
                        })
        
        return relationships
    
    def _extract_cooccurrence_relationships(self, doc, chapter: int) -> List[Dict]:
        """Infer relationships from entity co-occurrence in sentences."""
        relationships = []
        
        for sent in doc.sents:
            # Find all character entities in sentence
            sent_entities = [ent for ent in doc.ents 
                           if ent.start >= sent.start and ent.end <= sent.end
                           and ent.label_ == "PERSON"]
            
            # Create weak relationships between co-occurring characters
            for i, ent1 in enumerate(sent_entities):
                for ent2 in sent_entities[i+1:]:
                    # Skip if same entity
                    if ent1.text.lower() == ent2.text.lower():
                        continue
                    
                    relationships.append({
                        "source": ent1.text,
                        "source_id": f"char_{ent1.text.lower().replace(' ', '_')}",
                        "target": ent2.text,
                        "target_id": f"char_{ent2.text.lower().replace(' ', '_')}",
                        "relation_type": "INTERACTS_WITH",
                        "strength": 0.3,
                        "confidence": 0.5,
                        "chapter": chapter,
                        "context": sent.text[:100],
                        "extraction_method": "cooccurrence"
                    })
        
        return relationships
    
    def _deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Remove duplicate relationships, keeping highest confidence."""
        seen = {}
        for rel in relationships:
            key = (rel["source_id"], rel["target_id"], rel["relation_type"])
            if key not in seen or rel.get("confidence", 0) > seen[key].get("confidence", 0):
                seen[key] = rel
        return list(seen.values())
    
    def _looks_like_entity(self, text: str) -> bool:
        """Check if text looks like a named entity."""
        if not text or len(text) < 2:
            return False
        # First letter capitalized, not all caps, not common word
        common = {"the", "a", "an", "he", "she", "it", "they", "we", "i", "you",
                 "his", "her", "their", "this", "that", "what", "who", "which"}
        return (text[0].isupper() and 
                not text.isupper() and 
                text.lower() not in common)
    
    def _calculate_relationship_strength(self, verb_token, subject_token) -> float:
        """Calculate relationship strength based on verb and context."""
        strength = 0.5  # Base
        
        # Boost for strong verbs
        verb = verb_token.lemma_.lower()
        if verb in self.conflict_verbs or verb in self.alliance_verbs:
            strength += 0.2
        if verb in self.family_verbs:
            strength += 0.3
        
        # Boost for adverb modifiers
        for child in verb_token.children:
            if child.dep_ == "advmod":
                if child.text.lower() in {"deeply", "truly", "always", "forever"}:
                    strength += 0.1
        
        return min(1.0, strength)
    
    def _classify_relation(self, verb: str) -> str:
        """Map verbs to relationship types."""
        verb_lower = verb.lower()
        
        if verb_lower in self.conflict_verbs:
            return "CONFLICTS_WITH"
        elif verb_lower in self.alliance_verbs:
            return "ALLIES_WITH"
        elif verb_lower in self.family_verbs:
            return "FAMILY"
        elif verb_lower in self.fear_verbs:
            return "FEARS"
        else:
            return "INTERACTS_WITH"
    
    def extract_character_traits(
        self, 
        text: str, 
        character_name: str
    ) -> Dict[str, List[str]]:
        """
        Extract personality traits and attributes for a character.
        """
        traits = []
        emotions = []
        
        if not self.nlp:
            return {"personality_traits": [], "emotional_states": []}
        
        doc = self.nlp(text)
        
        # Find sentences mentioning the character
        for sent in doc.sents:
            if character_name.lower() not in sent.text.lower():
                continue
            
            # Find adjectives near character mention
            for token in sent:
                if token.pos_ == "ADJ":
                    if self._is_trait_for_character(token, character_name, sent):
                        traits.append(token.text.lower())
                
                # Extract emotional states from verbs
                if token.pos_ == "VERB" and token.lemma_ in {
                    "feel", "felt", "seem", "look", "appear"
                }:
                    for child in token.children:
                        if child.pos_ == "ADJ":
                            emotions.append(child.text.lower())
        
        return {
            "personality_traits": list(set(traits)),
            "emotional_states": list(set(emotions))
        }
    
    def _is_trait_for_character(self, adj_token, character_name: str, sent) -> bool:
        """Check if an adjective describes the target character."""
        char_indices = []
        for token in sent:
            if character_name.lower() in token.text.lower():
                char_indices.append(token.i)
        
        if not char_indices:
            return False
        
        # Check if adjective is within 5 tokens of character mention
        for idx in char_indices:
            if abs(adj_token.i - idx) <= 5:
                return True
        
        return False
    
    def extract_locations(self, text: str, chapter: int = 1) -> List[Dict]:
        """Extract location mentions from text."""
        locations = []
        
        if not self.nlp:
            return locations
        
        doc = self.nlp(text)
        seen = set()
        
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC", "FAC") and ent.text not in seen:
                seen.add(ent.text)
                locations.append({
                    "id": f"loc_{ent.text.lower().replace(' ', '_')}",
                    "type": "LOCATION",
                    "name": ent.text,
                    "chapter": chapter,
                    "confidence": 0.8
                })
        
        return locations
    
    def extract_events(self, text: str, chapter: int = 1) -> List[Dict]:
        """
        Extract significant events from text.
        Events are identified by action verbs with subjects and objects.
        """
        events = []
        
        if not self.nlp:
            return events
        
        doc = self.nlp(text)
        
        for sent in doc.sents:
            # Look for sentences with actionable content
            has_action = False
            action_verb = None
            
            for token in sent:
                if token.pos_ == "VERB" and token.lemma_ in {
                    "discover", "find", "create", "destroy", "reveal",
                    "attack", "defeat", "win", "lose", "escape", "capture",
                    "meet", "arrive", "leave", "die", "born", "marry"
                }:
                    has_action = True
                    action_verb = token.lemma_
                    break
            
            if has_action:
                event_id = f"event_{chapter}_{len(events)}"
                events.append({
                    "id": event_id,
                    "type": "EVENT",
                    "name": f"{action_verb.capitalize()} event",
                    "description": sent.text.strip(),
                    "chapter": chapter,
                    "action": action_verb,
                    "confidence": 0.7
                })
        
        return events
    
    def get_entity_summary(self) -> Dict:
        """Get summary of known entity variants."""
        return {
            "total_canonical_names": len(self.entity_variants),
            "variants": {k: list(v) for k, v in self.entity_variants.items()}
        }

