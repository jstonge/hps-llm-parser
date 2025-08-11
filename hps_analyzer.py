"""
📚 HPS ANALYZER (History and Philosophy of Science Text Analysis) 📚

An analyzer for extracting structured information from HPS texts:
- Authors (scholars, philosophers, scientists)
- Works (books, papers, theories)
- Topics (concepts, research areas, methodologies)

    "Every text tells a story of ideas, people, and discoveries!"

         .-------------------.
       /   Who influenced    \
      |     whom and when?    |
      |                       |
       \   What concepts     /
        \   emerged here?   /
         '-------------------'

    Current Status:
    Curiosity       ▓▓▓▓▓▓▓▓▓▓ 100%
    Context         ▓▓▓▓▓▓▓░░░  70%
    Understanding   ▓▓▓▓▓▓░░░░  60%

    Analysis Focus:
    - Intellectual genealogies
    - Conceptual evolution
    - Historical contextualization
    - Cross-referencing networks
"""

# Imports
from enum import Enum
from typing import List, Optional, Literal
import outlines
from pydantic import BaseModel, Field
from datetime import datetime
import os
from pathlib import Path

# For pretty printing
from rich import print
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.text import Text

# Scholar roles in HPS contexts
class ScholarRole(str, Enum):
    PHILOSOPHER = "PHILOSOPHER"
    SCIENTIST = "SCIENTIST" 
    HISTORIAN = "HISTORIAN"
    MATHEMATICIAN = "MATHEMATICIAN"
    THEORIST = "THEORIST"
    EMPIRICIST = "EMPIRICIST"
    CRITIC = "CRITIC"
    TRANSLATOR = "TRANSLATOR"
    COMMENTATOR = "COMMENTATOR"
    UNKNOWN = "UNKNOWN"

# Types of works referenced
class WorkType(str, Enum):
    BOOK = "BOOK"
    PAPER = "PAPER"
    ESSAY = "ESSAY"
    TREATISE = "TREATISE"
    DIALOGUE = "DIALOGUE"
    CORRESPONDENCE = "CORRESPONDENCE"
    THEORY = "THEORY"
    EXPERIMENT = "EXPERIMENT"
    UNKNOWN = "UNKNOWN"

# Academic fields/disciplines
class AcademicField(str, Enum):
    PHILOSOPHY = "PHILOSOPHY"
    PHYSICS = "PHYSICS"
    MATHEMATICS = "MATHEMATICS"
    BIOLOGY = "BIOLOGY"
    CHEMISTRY = "CHEMISTRY"
    ASTRONOMY = "ASTRONOMY"
    LOGIC = "LOGIC"
    EPISTEMOLOGY = "EPISTEMOLOGY"
    METAPHYSICS = "METAPHYSICS"
    HISTORY_OF_SCIENCE = "HISTORY_OF_SCIENCE"
    INTERDISCIPLINARY = "INTERDISCIPLINARY"
    UNKNOWN = "UNKNOWN"

class Author(BaseModel):
    """A person mentioned in the text (scholar, philosopher, scientist, etc.)"""
    name: str = Field(description="Full name of the person")
    time_period: str = Field(
        description="Historical period or specific dates (e.g., '5th century BC', '1643-1727', 'Medieval period')",
        default="not specified"
    )
    place: str = Field(
        description="Geographic location or cultural context (e.g., 'Ancient Greece', 'Cambridge', 'Vienna Circle')",
        default="not specified"
    )
    role: ScholarRole = Field(
        description="Primary intellectual role or occupation",
        default=ScholarRole.UNKNOWN
    )
    brief_description: str = Field(
        description="Brief description of their contribution or significance",
        default="not specified"
    )

class Work(BaseModel):
    """A work (book, paper, theory, etc.) mentioned in the text"""
    title: str = Field(description="Title of the work")
    author: str = Field(
        description="Author of the work (if mentioned)",
        default="not specified"
    )
    work_type: WorkType = Field(
        description="Type of work",
        default=WorkType.UNKNOWN
    )
    publication_info: str = Field(
        description="Publication year, publisher, or other relevant info",
        default="not specified"
    )
    significance: str = Field(
        description="Why this work is mentioned or its importance in context",
        default="not specified"
    )

class Topic(BaseModel):
    """A concept, theory, or research area discussed in the text"""
    name: str = Field(description="Name of the topic/concept")
    field: AcademicField = Field(
        description="Primary academic field",
        default=AcademicField.UNKNOWN
    )
    description: str = Field(
        description="Brief explanation of the concept",
        default="not specified"
    )
    related_concepts: List[str] = Field(
        description="Related topics or concepts mentioned nearby",
        default_factory=list
    )
    historical_context: str = Field(
        description="When and how this concept emerged or evolved",
        default="not specified"
    )

class ChapterAnalysis(BaseModel):
    """Complete analysis of an HPS text chapter"""
    # High-level overview
    summary: str = Field(description="Brief summary of the chapter's main themes")
    main_thesis: str = Field(
        description="The primary argument or thesis of the chapter",
        default="not specified"
    )
    
    # Extracted entities
    authors: List[Author] = Field(
        description="All people mentioned in the text",
        default_factory=list
    )
    works: List[Work] = Field(
        description="All works (books, papers, theories) referenced",
        default_factory=list
    )
    topics: List[Topic] = Field(
        description="Key concepts and research areas discussed",
        default_factory=list
    )
    
    # Temporal and geographic analysis
    time_periods_covered: List[str] = Field(
        description="Historical periods discussed (e.g., 'Ancient Greece', '17th century', 'Modern era')",
        default_factory=list
    )
    geographical_locations: List[str] = Field(
        description="Places mentioned (cities, countries, institutions)",
        default_factory=list
    )
    
    # Intellectual relationships
    key_debates: List[str] = Field(
        description="Major intellectual debates or controversies discussed",
        default_factory=list
    )
    influence_networks: List[str] = Field(
        description="'Who influenced whom' relationships (e.g., 'Kant influenced by Hume')",
        default_factory=list
    )
    
    # Metadata
    complexity_score: float = Field(
        ge=0.0,
        le=10.0, 
        description="How conceptually complex the chapter is on a scale of 0-10 (0=very simple, 10=extremely complex)",
        default=5.0
    )
    primary_fields: List[AcademicField] = Field(
        description="Main academic disciplines covered",
        default_factory=list
    )

def format_hps_analysis(analysis: ChapterAnalysis):
    """Format an HPS analysis into rich console output"""
    console = Console()
    
    # Header
    header = Panel(
        f"[bold blue]HPS Chapter Analysis Report[/]\n[cyan]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]",
        border_style="blue"
    )
    
    # Summary section
    summary_text = f"[bold white]Main Thesis:[/]\n[cyan]{analysis.main_thesis}[/]\n\n"
    summary_text += f"[bold white]Summary:[/]\n[cyan]{analysis.summary}[/]\n\n"
    summary_text += f"[bold white]Complexity Score:[/] [yellow]{analysis.complexity_score}/10[/]\n"
    summary_text += f"[bold white]Primary Fields:[/] [green]{', '.join([f.value for f in analysis.primary_fields])}[/]"
    
    summary = Panel(summary_text, border_style="cyan", title="[bold]Overview[/]")
    
    # Authors table
    authors_table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    authors_table.add_column("Name", style="magenta", width=20)
    authors_table.add_column("Role", style="cyan", width=15)
    authors_table.add_column("Period", style="yellow", width=15)
    authors_table.add_column("Place", style="green", width=20)
    authors_table.add_column("Description", style="white", width=40)
    
    for author in analysis.authors:
        authors_table.add_row(
            author.name,
            author.role.value,
            author.time_period,
            author.place,
            author.brief_description[:80] + "..." if len(author.brief_description) > 80 else author.brief_description
        )
    
    # Works table
    works_table = Table(show_header=True, header_style="bold blue", show_lines=True)
    works_table.add_column("Title", style="blue", width=30)
    works_table.add_column("Author", style="magenta", width=20)
    works_table.add_column("Type", style="cyan", width=12)
    works_table.add_column("Info", style="yellow", width=20)
    works_table.add_column("Significance", style="white", width=40)
    
    for work in analysis.works:
        works_table.add_row(
            work.title[:30] + "..." if len(work.title) > 30 else work.title,
            work.author,
            work.work_type.value,
            work.publication_info,
            work.significance[:40] + "..." if len(work.significance) > 40 else work.significance
        )
    
    # Topics table
    topics_table = Table(show_header=True, header_style="bold green", show_lines=True)
    topics_table.add_column("Concept", style="green", width=25)
    topics_table.add_column("Field", style="cyan", width=15)
    topics_table.add_column("Description", style="white", width=50)
    topics_table.add_column("Related", style="yellow", width=30)
    
    for topic in analysis.topics:
        topics_table.add_row(
            topic.name,
            topic.field.value,
            topic.description[:50] + "..." if len(topic.description) > 50 else topic.description,
            ", ".join(topic.related_concepts[:3])  # Show first 3 related concepts
        )
    
    # Context panels
    temporal_text = "Time Periods: " + ", ".join(analysis.time_periods_covered) if analysis.time_periods_covered else "No specific time periods identified"
    geo_text = "Locations: " + ", ".join(analysis.geographical_locations) if analysis.geographical_locations else "No specific locations identified"
    
    context_panel = Panel(
        f"[bold yellow]Temporal Context:[/]\n{temporal_text}\n\n[bold yellow]Geographic Context:[/]\n{geo_text}",
        border_style="yellow",
        title="[bold]Historical Context[/]"
    )
    
    # Print everything
    console.print(header)
    console.print(summary)
    console.print(f"\n[bold magenta]👥 Authors & Scholars ({len(analysis.authors)} found):[/]")
    console.print(authors_table)
    console.print(f"\n[bold blue]📚 Works & Publications ({len(analysis.works)} found):[/]")
    console.print(works_table)
    console.print(f"\n[bold green]🧠 Concepts & Topics ({len(analysis.topics)} found):[/]")
    console.print(topics_table)
    console.print(context_panel)
    
    if analysis.key_debates:
        debates_text = "\n".join([f"• {debate}" for debate in analysis.key_debates])
        debates_panel = Panel(debates_text, border_style="red", title="[bold red]Key Debates[/]")
        console.print(debates_panel)
    
    if analysis.influence_networks:
        influence_text = "\n".join([f"• {influence}" for influence in analysis.influence_networks])
        influence_panel = Panel(influence_text, border_style="purple", title="[bold purple]Intellectual Influences[/]")
        console.print(influence_panel)


class HPSAnalyzer:
    """Main class for analyzing HPS texts"""
    
    def __init__(
        self,
        model,
        tokenizer,
        prompt_template_path: Optional[str] = None,
        token_max: int = 2000,
        chunk_size: int = 500  # words per chunk for long texts
    ):
        if token_max <= 0:
            raise ValueError("token_max must be positive")
        
        self.model = model
        self.tokenizer = tokenizer
        self.token_max = token_max
        self.chunk_size = chunk_size
        
        # Default prompt template
        if prompt_template_path and os.path.exists(prompt_template_path):
            with open(prompt_template_path, "r") as file:
                self.prompt_template = file.read()
        else:
            self.prompt_template = self._get_default_prompt()
        
        # Initialize generator
        self.analyzer = outlines.generate.json(
            self.model,
            ChapterAnalysis,
            sampler=outlines.samplers.greedy(),
        )
    
    def _get_default_prompt(self) -> str:
        return """You are an expert in History and Philosophy of Science (HPS). 
Analyze the following text and extract structured information about:

1. **Authors/Scholars**: People mentioned (philosophers, scientists, historians)
2. **Works**: Books, papers, theories, or other intellectual productions referenced  
3. **Topics**: Concepts, research areas, methodologies, or theoretical frameworks discussed
4. **Historical Context**: Time periods, locations, and intellectual relationships
5. **Key Arguments**: Main thesis and significant debates

For complexity_score, use a scale of 0-10 where:
- 0-2: Very simple, introductory material
- 3-5: Moderate complexity, standard academic content
- 6-8: High complexity, advanced concepts
- 9-10: Extremely complex, cutting-edge research

Text to analyze:
{text}

Provide a comprehensive analysis following the required JSON schema. Be thorough but precise."""

    def _to_prompt(self, text: str) -> str:
        """Convert text to analysis prompt"""
        messages = [
            {
                "role": "user", 
                "content": self.prompt_template.format(text=text)
            }
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split long text into manageable chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(" ".join(chunk_words))
        
        return chunks
    
    def analyze_chapter(
        self,
        text: str,
        format_output: bool = True,
        use_chunking: bool = True
    ) -> ChapterAnalysis:
        """
        Analyze a single chapter or text section
        
        Args:
            text: The text to analyze
            format_output: Whether to print formatted output
            use_chunking: Whether to chunk long texts
            
        Returns:
            ChapterAnalysis object
        """
        
        # Check if text is too long and should be chunked
        word_count = len(text.split())
        
        if use_chunking and word_count > self.chunk_size:
            print(f"[yellow]Text is long ({word_count} words). Processing in chunks...[/]")
            chunks = self._chunk_text(text)
            
            # Analyze each chunk and combine results
            all_analyses = []
            for i, chunk in enumerate(chunks):
                print(f"[cyan]Processing chunk {i+1}/{len(chunks)}...[/]")
                prompt = self._to_prompt(chunk)
                analysis = self.analyzer(prompt, max_tokens=self.token_max)
                all_analyses.append(analysis)
            
            # Combine analyses (simplified - just merge lists)
            combined = self._combine_analyses(all_analyses)
            
        else:
            # Process as single text
            prompt = self._to_prompt(text)
            combined = self.analyzer(prompt, max_tokens=self.token_max)
        
        if format_output:
            format_hps_analysis(combined)
        
        return combined
    
    def _combine_analyses(self, analyses: List[ChapterAnalysis]) -> ChapterAnalysis:
        """Combine multiple chunk analyses into one"""
        if not analyses:
            return ChapterAnalysis(summary="No analysis performed")
        
        if len(analyses) == 1:
            return analyses[0]
        
        # Combine all the lists
        all_authors = []
        all_works = []
        all_topics = []
        all_time_periods = []
        all_locations = []
        all_debates = []
        all_influences = []
        all_fields = []
        
        for analysis in analyses:
            all_authors.extend(analysis.authors)
            all_works.extend(analysis.works)
            all_topics.extend(analysis.topics)
            all_time_periods.extend(analysis.time_periods_covered)
            all_locations.extend(analysis.geographical_locations)
            all_debates.extend(analysis.key_debates)
            all_influences.extend(analysis.influence_networks)
            all_fields.extend(analysis.primary_fields)
        
        # Remove duplicates (simplified - by name/title)
        unique_authors = {author.name: author for author in all_authors}.values()
        unique_works = {work.title: work for work in all_works}.values()
        unique_topics = {topic.name: topic for topic in all_topics}.values()
        
        # Create combined summary
        summaries = [a.summary for a in analyses if a.summary != "not specified"]
        combined_summary = " | ".join(summaries) if summaries else "Combined analysis of multiple text sections"
        
        return ChapterAnalysis(
            summary=combined_summary,
            main_thesis=" | ".join([a.main_thesis for a in analyses if a.main_thesis != "not specified"]),
            authors=list(unique_authors),
            works=list(unique_works),
            topics=list(unique_topics),
            time_periods_covered=list(set(all_time_periods)),
            geographical_locations=list(set(all_locations)),
            key_debates=list(set(all_debates)),
            influence_networks=list(set(all_influences)),
            complexity_score=min(10.0, sum(a.complexity_score for a in analyses) / len(analyses)),
            primary_fields=list(set(all_fields))
        )
    
    def analyze_file(self, file_path: str) -> ChapterAnalysis:
        """Analyze a text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self.analyze_chapter(text)

# Example usage function
def demo_analysis():
    """Demo function showing how to use the HPS analyzer"""
    
    # Sample HPS text
    sample_text = """
    Whether Ong would have seen cyberspace as fundamentally oral or literary, he would surely have recognized it as transformative: not just a revitalization of older forms, not just an amplification, but something wholly new. He might have sensed a coming discontinuity akin to the emergence of literacy itself. Few understood better than Ong just how profound a discontinuity that had been.
    
    When he began his studies, "oral literature" was a common phrase. It is an oxymoron laced with anachronism; the words imply an all-too-unconscious approach to the past by way of the present. Oral literature was generally treated as a variant of writing; this, Ong said, was "rather like thinking of horses as automobiles without wheels."
    
    "Language in fact bears the same relationship to the concept of mind that legislation bears to the concept of parliament," says Jonathan Miller: "it is a competence forever bodying itself in a series of concrete performances." Much the same might be said of writing—it is concrete performance—but when the word is instantiated in paper or stone, it takes on a separate existence as artifice. It is a product of tools, and it is a tool. And like many technologies that followed, it thereby inspired immediate detractors.
    
    One unlikely Luddite was also one of the first long-term beneficiaries. Plato (channeling the nonwriter Socrates) warned that this technology meant impoverishment.
    """
    
    print("[green]This is a demo of the HPS Analyzer![/]")
    print("[yellow]In practice, you would initialize with your actual model:[/]")
    print("[dim]# analyzer = HPSAnalyzer(model, tokenizer)[/]")
    print("[dim]# result = analyzer.analyze_chapter(your_text)[/]")
    
    # This would be the actual usage:
    # analyzer = HPSAnalyzer(your_model, your_tokenizer)
    # result = analyzer.analyze_chapter(sample_text)

if __name__ == "__main__":
    demo_analysis()