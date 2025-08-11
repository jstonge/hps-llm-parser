"""
📚 INCREMENTAL HPS ANALYZER 📚

An analyzer that builds cumulative knowledge across text chunks:
- Maintains persistent entity database
- Updates entities with new information 
- Uses prior knowledge to inform subsequent analysis
- Tracks entity mentions and relationships across the full text

    "Building knowledge incrementally, like a detective solving a case!"

         .-------------------.
       /   First mention:     \
      |   "Walter J. Ong"     |
      |                       |
       \   Later: "Ong said"  /
        \   -> Same person!  /
         '-------------------'
"""

from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from rich import print
from rich.panel import Panel
from rich.console import Console

# Import our existing classes
from hps_analyzer import (
    HPSAnalyzer, Author, Work, Topic, ChapterAnalysis,
    ScholarRole, WorkType, AcademicField
)

@dataclass
class EntityMention:
    """A single mention of an entity in text"""
    chunk_id: str
    mention_text: str
    context: str  # Surrounding text for context
    page_number: Optional[int] = None
    confidence: float = 0.8

@dataclass 
class KnowledgeEntity:
    """Enhanced entity with cumulative knowledge"""
    # Core identity
    canonical_name: str
    entity_type: str  # "person", "work", "topic"
    
    # All known variations of the name
    name_variants: Set[str] = field(default_factory=set)
    
    # Cumulative attributes (grows over time)
    attributes: Dict[str, str] = field(default_factory=dict)
    
    # All mentions across chunks
    mentions: List[EntityMention] = field(default_factory=list)
    
    # Relationships to other entities
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    
    # Confidence scores for different attributes
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    def add_mention(self, mention: EntityMention):
        """Add a new mention of this entity"""
        self.mentions.append(mention)
    
    def update_attribute(self, key: str, value: str, confidence: float = 0.8):
        """Update an attribute if new info is more reliable"""
        current_confidence = self.confidence_scores.get(key, 0.0)
        
        # Update if new info is more confident or if we don't have this info yet
        if confidence > current_confidence or key not in self.attributes:
            self.attributes[key] = value
            self.confidence_scores[key] = confidence
    
    def add_name_variant(self, name: str):
        """Add a new way this entity can be referred to"""
        self.name_variants.add(name.strip())
    
    def get_summary(self) -> str:
        """Get a summary of what we know about this entity"""
        summary = f"{self.canonical_name} ({self.entity_type})"
        
        if self.attributes:
            key_attrs = []
            if "role" in self.attributes:
                key_attrs.append(f"Role: {self.attributes['role']}")
            if "time_period" in self.attributes:
                key_attrs.append(f"Time: {self.attributes['time_period']}")
            if "place" in self.attributes:
                key_attrs.append(f"Place: {self.attributes['place']}")
                
            if key_attrs:
                summary += " - " + ", ".join(key_attrs)
        
        summary += f" [{len(self.mentions)} mentions]"
        return summary

class KnowledgeBase:
    """Persistent knowledge base that grows with each chunk"""
    
    def __init__(self):
        self.entities: Dict[str, KnowledgeEntity] = {}
        self.chunk_count = 0
        self.name_to_canonical: Dict[str, str] = {}  # Maps any name variant to canonical name
    
    def find_entity_by_name(self, name: str) -> Optional[KnowledgeEntity]:
        """Find entity by any known name variant"""
        name_clean = name.strip().lower()
        
        # Try exact match first
        canonical = self.name_to_canonical.get(name_clean)
        if canonical:
            return self.entities.get(canonical)
        
        # Try fuzzy matching for common patterns
        for variant, canonical in self.name_to_canonical.items():
            if self._names_match(name_clean, variant):
                return self.entities.get(canonical)
        
        return None
    
    def _names_match(self, name1: str, name2: str) -> bool:
        """Check if two names likely refer to the same person"""
        # Handle common cases like "Walter J. Ong" vs "Ong" vs "Walter Ong"
        
        # Split into parts
        parts1 = name1.replace('.', '').split()
        parts2 = name2.replace('.', '').split()
        
        # If one is a subset of the other (e.g., "Ong" subset of "Walter J Ong")
        if all(part in parts2 for part in parts1) or all(part in parts1 for part in parts2):
            return True
        
        # Check last name matching (common for academic references)
        if len(parts1) > 1 and len(parts2) > 1:
            if parts1[-1] == parts2[-1]:  # Same last name
                return True
        
        return False
    
    def add_or_update_entity(
        self, 
        name: str, 
        entity_type: str, 
        attributes: Dict[str, str], 
        chunk_id: str,
        context: str = ""
    ) -> KnowledgeEntity:
        """Add new entity or update existing one"""
        
        # Try to find existing entity
        existing = self.find_entity_by_name(name)
        
        if existing:
            # Update existing entity
            entity = existing
            entity.add_name_variant(name)
            
            # Update attributes with new information
            for key, value in attributes.items():
                if value and value != "not specified":
                    # Higher confidence for more recent mentions
                    confidence = 0.8 + (0.1 * self.chunk_count / 10)  # Slight boost for later chunks
                    entity.update_attribute(key, value, confidence)
                    
        else:
            # Create new entity
            canonical_name = self._get_canonical_name(name, entity_type)
            entity = KnowledgeEntity(
                canonical_name=canonical_name,
                entity_type=entity_type,
                attributes=attributes
            )
            entity.add_name_variant(name)
            self.entities[canonical_name] = entity
        
        # Add mention
        mention = EntityMention(
            chunk_id=chunk_id,
            mention_text=name,
            context=context[:200]  # Keep first 200 chars of context
        )
        entity.add_mention(mention)
        
        # Update name mapping
        name_clean = name.strip().lower()
        self.name_to_canonical[name_clean] = entity.canonical_name
        
        return entity
    
    def _get_canonical_name(self, name: str, entity_type: str) -> str:
        """Generate canonical name for entity"""
        # For now, just use the first full name we encounter
        return name.strip()
    
    def get_known_entities_summary(self) -> str:
        """Get a summary of all known entities for context"""
        if not self.entities:
            return "No entities known yet."
        
        summaries = []
        for entity in list(self.entities.values())[:10]:  # Limit to top 10 for context
            summaries.append(entity.get_summary())
        
        result = f"Previously identified entities ({len(self.entities)} total):\n"
        result += "\n".join([f"• {s}" for s in summaries])
        
        if len(self.entities) > 10:
            result += f"\n... and {len(self.entities) - 10} more"
        
        return result
    
    def export_knowledge_base(self) -> Dict:
        """Export knowledge base to JSON-serializable format"""
        return {
            "entities": {
                canonical: {
                    "canonical_name": entity.canonical_name,
                    "entity_type": entity.entity_type,
                    "name_variants": list(entity.name_variants),
                    "attributes": entity.attributes,
                    "mention_count": len(entity.mentions),
                    "relationships": entity.relationships
                }
                for canonical, entity in self.entities.items()
            },
            "total_chunks_processed": self.chunk_count,
            "total_entities": len(self.entities)
        }

class IncrementalChunkAnalysis(BaseModel):
    """Analysis result for a single chunk that's aware of prior knowledge"""
    
    # Standard fields
    chunk_summary: str
    chunk_id: str
    
    # Entities found in this chunk
    new_authors: List[Author] = Field(default_factory=list)
    new_works: List[Work] = Field(default_factory=list) 
    new_topics: List[Topic] = Field(default_factory=list)
    
    # References to previously known entities
    referenced_entities: List[str] = Field(
        description="Names of previously known entities mentioned in this chunk",
        default_factory=list
    )
    
    # New relationships discovered
    new_relationships: List[str] = Field(
        description="New relationships between entities (e.g., 'Ong influenced by McLuhan')",
        default_factory=list
    )

class IncrementalHPSAnalyzer:
    """HPS Analyzer that builds knowledge incrementally"""
    
    def __init__(
        self,
        model,
        tokenizer, 
        token_max: int = 900,
        chunk_size: int = 500,
        knowledge_base: Optional[KnowledgeBase] = None
    ):
        # Use the base analyzer for LLM operations
        self.base_analyzer = HPSAnalyzer(model, tokenizer, token_max=token_max, chunk_size=chunk_size)
        
        # Knowledge base for persistent storage
        self.knowledge_base = knowledge_base or KnowledgeBase()
        
        # Enhanced prompt that includes prior knowledge
        self.incremental_prompt_template = """You are an expert in History and Philosophy of Science (HPS) analyzing a text incrementally.

PRIOR KNOWLEDGE FROM PREVIOUS CHUNKS:
{prior_knowledge}

CURRENT CHUNK TO ANALYZE:
{chunk_text}

Your task:
1. Identify any NEW entities (people, works, topics) not already in the prior knowledge
2. Identify any mentions of PREVIOUSLY KNOWN entities from the prior knowledge 
3. Look for new information about previously known entities
4. Identify relationships between entities

Focus on incremental knowledge building - what's NEW in this chunk vs what we already knew?

Analyze this chunk:"""

    def analyze_text_incrementally(
        self, 
        text: str, 
        format_output: bool = True
    ) -> List[IncrementalChunkAnalysis]:
        """Analyze text incrementally, building knowledge as we go"""
        
        console = Console()
        
        if format_output:
            console.print(Panel(
                "[bold blue]Starting Incremental HPS Analysis[/]\n"
                f"Text length: {len(text)} characters\n"
                f"Starting knowledge: {len(self.knowledge_base.entities)} entities",
                border_style="blue"
            ))
        
        # Chunk the text
        chunks = self.base_analyzer._chunk_text(text)
        results = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i+1:03d}"
            self.knowledge_base.chunk_count += 1
            
            if format_output:
                console.print(f"\n[cyan]Processing {chunk_id} ({i+1}/{len(chunks)})...[/]")
            
            # Analyze this chunk with context of prior knowledge
            chunk_result = self._analyze_chunk_incrementally(chunk, chunk_id)
            
            if chunk_result:
                results.append(chunk_result)
                
                # Update knowledge base with new findings
                self._update_knowledge_base(chunk_result, chunk_id, chunk)
                
                if format_output:
                    self._print_chunk_summary(chunk_result, console)
            
        if format_output:
            self._print_final_knowledge_summary(console)
        
        return results
    
    def _analyze_chunk_incrementally(
        self, 
        chunk: str, 
        chunk_id: str
    ) -> Optional[IncrementalChunkAnalysis]:
        """Analyze a single chunk with awareness of prior knowledge"""
        
        # Get context of what we already know
        prior_knowledge = self.knowledge_base.get_known_entities_summary()
        
        # Create context-aware prompt
        full_prompt = self.incremental_prompt_template.format(
            prior_knowledge=prior_knowledge,
            chunk_text=chunk
        )
        
        # Use the base analyzer's generation capability
        messages = [{"role": "user", "content": full_prompt}]
        prompt = self.base_analyzer.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        try:
            # Generate using our incremental schema
            import outlines.generate as generate
            incremental_generator = generate.json(
                self.base_analyzer.model,
                IncrementalChunkAnalysis
            )
            
            result = incremental_generator(prompt, max_tokens=self.base_analyzer.token_max)
            
            if isinstance(result, str):
                import json
                result_dict = json.loads(result)
                return IncrementalChunkAnalysis(**result_dict)
            elif isinstance(result, IncrementalChunkAnalysis):
                return result
            else:
                return IncrementalChunkAnalysis(
                    chunk_summary="Analysis failed",
                    chunk_id=chunk_id
                )
                
        except Exception as e:
            print(f"[red]Error analyzing {chunk_id}: {e}[/]")
            return None
    
    def _update_knowledge_base(
        self, 
        chunk_result: IncrementalChunkAnalysis, 
        chunk_id: str, 
        chunk_text: str
    ):
        """Update knowledge base with findings from this chunk"""
        
        # Add new authors
        for author in chunk_result.new_authors:
            self.knowledge_base.add_or_update_entity(
                name=author.name,
                entity_type="person",
                attributes={
                    "role": author.role.value if author.role else "unknown",
                    "time_period": author.time_period,
                    "place": author.place,
                    "description": author.brief_description
                },
                chunk_id=chunk_id,
                context=chunk_text[:300]
            )
        
        # Add new works
        for work in chunk_result.new_works:
            self.knowledge_base.add_or_update_entity(
                name=work.title,
                entity_type="work", 
                attributes={
                    "author": work.author,
                    "type": work.work_type.value if work.work_type else "unknown",
                    "publication_info": work.publication_info,
                    "significance": work.significance
                },
                chunk_id=chunk_id,
                context=chunk_text[:300]
            )
        
        # Add new topics
        for topic in chunk_result.new_topics:
            self.knowledge_base.add_or_update_entity(
                name=topic.name,
                entity_type="topic",
                attributes={
                    "field": topic.field.value if topic.field else "unknown",
                    "description": topic.description,
                    "historical_context": topic.historical_context,
                    "related_concepts": ", ".join(topic.related_concepts)
                },
                chunk_id=chunk_id,
                context=chunk_text[:300]
            )
        
        # Update mentions of previously known entities
        for entity_name in chunk_result.referenced_entities:
            entity = self.knowledge_base.find_entity_by_name(entity_name)
            if entity:
                mention = EntityMention(
                    chunk_id=chunk_id,
                    mention_text=entity_name,
                    context=chunk_text[:200]
                )
                entity.add_mention(mention)
    
    def _print_chunk_summary(self, result: IncrementalChunkAnalysis, console: Console):
        """Print summary of what was found in this chunk"""
        
        new_count = len(result.new_authors) + len(result.new_works) + len(result.new_topics)
        ref_count = len(result.referenced_entities)
        
        summary_text = f"[green]New entities: {new_count}[/] | [blue]References to known: {ref_count}[/]"
        
        if result.new_relationships:
            summary_text += f" | [purple]New relationships: {len(result.new_relationships)}[/]"
        
        console.print(f"  {summary_text}")
        
        # Show a few examples
        if result.new_authors:
            console.print(f"  [dim]New people: {', '.join([a.name for a in result.new_authors[:3]])}[/]")
        
        if result.referenced_entities:
            console.print(f"  [dim]Mentioned again: {', '.join(result.referenced_entities[:3])}[/]")
    
    def _print_final_knowledge_summary(self, console: Console):
        """Print final summary of accumulated knowledge"""
        
        kb = self.knowledge_base
        
        # Count by type
        people = sum(1 for e in kb.entities.values() if e.entity_type == "person")
        works = sum(1 for e in kb.entities.values() if e.entity_type == "work") 
        topics = sum(1 for e in kb.entities.values() if e.entity_type == "topic")
        
        summary = Panel(
            f"[bold green]Final Knowledge Base Summary[/]\n\n"
            f"[blue]Total Entities:[/] {len(kb.entities)}\n"
            f"  • People: {people}\n"
            f"  • Works: {works}\n" 
            f"  • Topics: {topics}\n\n"
            f"[yellow]Chunks Processed:[/] {kb.chunk_count}\n\n"
            f"[purple]Most Mentioned Entities:[/]\n" +
            "\n".join([
                f"  • {entity.canonical_name} ({len(entity.mentions)} mentions)"
                for entity in sorted(kb.entities.values(), key=lambda e: len(e.mentions), reverse=True)[:5]
            ]),
            border_style="green",
            title="[bold]Incremental Analysis Complete[/]"
        )
        
        console.print(summary)
    
    def save_knowledge_base(self, filepath: str):
        """Save knowledge base to JSON file"""
        data = self.knowledge_base.export_knowledge_base()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[green]Knowledge base saved to {filepath}[/]")
    
    def load_knowledge_base(self, filepath: str):
        """Load knowledge base from JSON file"""
        # Implementation would reconstruct KnowledgeBase from saved data
        pass

# Example usage
def demo_incremental_analysis():
    """Demo the incremental analysis system"""
    
    sample_texts = [
        """Walter J. Ong was a Jesuit priest and scholar who studied the transition from oral to literate cultures. 
        Born in 1912, he became famous for his work "Orality and Literacy" published in 1982.""",
        
        """Ong argued that writing fundamentally changed human consciousness. He distinguished between primary oral cultures 
        and literate cultures, showing how each shapes thought differently.""",
        
        """The work of Ong influenced many scholars including Eric Havelock, who also studied ancient Greek literacy. 
        Havelock's research on Homer complemented Ong's theoretical framework."""
    ]
    
    print("[green]Demo: Incremental HPS Analysis[/]")
    print("[yellow]This shows how knowledge builds across chunks:[/]")
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n[bold blue]--- Processing Text Segment {i} ---[/]")
        print(f"[dim]{text[:100]}...[/]")
        print("[cyan]In real usage, this would analyze with your model[/]")
        
        # Show what would be extracted
        if i == 1:
            print("[green]NEW: Walter J. Ong (person) - Jesuit priest, born 1912[/]")
            print("[green]NEW: 'Orality and Literacy' (work) - published 1982[/]")
        elif i == 2: 
            print("[blue]REFERENCE: Ong (already known)[/]")
            print("[green]NEW: primary oral cultures (topic)[/]")
        elif i == 3:
            print("[blue]REFERENCE: Ong (already known)[/]") 
            print("[green]NEW: Eric Havelock (person) - scholar[/]")
            print("[purple]RELATIONSHIP: Ong influenced Havelock[/]")

if __name__ == "__main__":
    demo_incremental_analysis()