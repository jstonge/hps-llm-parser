# HPS LLM Parser =Ú

**Intelligent extraction and analysis of scholarly information from History and Philosophy of Science texts using Large Language Models**

Transform dense academic texts into structured, interconnected knowledge graphs that track authors, works, concepts, and their relationships across entire documents.

---

## <¯ What This Does

The HPS LLM Parser analyzes academic texts (books, papers, chapters) and extracts:

- **=e Authors & Scholars**: People mentioned with their roles, time periods, and contributions
- **=Ö Works & Publications**: Books, papers, theories with authorship and significance  
- **>à Concepts & Topics**: Ideas, methodologies, research areas with relationships
- **= Intellectual Networks**: Who influenced whom, citation relationships
- **=Í Historical Context**: Time periods, locations, academic institutions

## ( Key Features

### = **Incremental Knowledge Building**
- **Persistent Memory**: Remembers entities across text chunks
- **Smart Name Matching**: "Walter J. Ong" ’ "Ong" ’ "W. Ong" (same person)
- **Progressive Enhancement**: Each mention adds information to existing entities
- **Confidence Scoring**: Weights information by reliability and recency

### <¯ **Structured Output Guaranteed** 
- **JSON Schema Enforcement**: Uses Outlines/Guidance for reliable data extraction
- **Pydantic Validation**: Type-safe, validated output structures
- **No Parsing Errors**: Structured generation prevents malformed JSON

### =Ê **Rich Analysis & Visualization**
- **Beautiful Console Output**: Color-coded tables and panels via Rich
- **Entity Relationship Maps**: Track intellectual genealogies  
- **Mention Tracking**: See where each entity appears across the text
- **Exportable Knowledge Base**: Save results as JSON for further analysis

---

## =€ Quick Start

### Installation
```bash
pip install outlines transformers torch pydantic rich
# OR: pip install guidance  # Alternative to outlines
```

### Basic Usage

#### Standard Analysis
```python
from hps_analyzer import HPSAnalyzer
import outlines
from transformers import AutoTokenizer

# Load your model
model = outlines.models.transformers("path/to/llama-model", device="cuda")
tokenizer = AutoTokenizer.from_pretrained("path/to/llama-model")

# Analyze text
analyzer = HPSAnalyzer(model, tokenizer)
result = analyzer.analyze_chapter(your_text)

# Rich formatted output is displayed automatically
# Access structured data: result.authors, result.works, result.topics
```

#### Incremental Analysis (Recommended)
```python
from incremental_hps_analyzer import IncrementalHPSAnalyzer

# Initialize with persistent knowledge base
analyzer = IncrementalHPSAnalyzer(model, tokenizer)

# Analyze long text - builds knowledge incrementally
results = analyzer.analyze_text_incrementally(long_text)

# Save knowledge base for future use
analyzer.save_knowledge_base("knowledge.json")
```

---

## <× Architecture

### Core Components

#### 1. **Base HPS Analyzer** (`hps_analyzer.py`)
- Single-pass analysis of text chunks
- Pydantic schemas for Authors, Works, Topics
- Rich formatting and visualization
- Handles chunking for long texts

#### 2. **Incremental Analyzer** (`incremental_hps_analyzer.py`) 
- **Knowledge Base**: Persistent entity storage across chunks
- **Smart Entity Matching**: Links name variations to canonical entities
- **Progressive Information Building**: Updates entities with new mentions
- **Context-Aware Prompting**: Uses prior knowledge to inform analysis

#### 3. **Structured Generation** 
- **Outlines Integration**: Guarantees JSON schema compliance
- **Guidance Support**: Alternative structured generation backend
- **Error Recovery**: Automatic retries with reduced token limits
- **Validation**: Pydantic ensures data integrity

### Data Flow

```
=Ä Input Text
    “
=* Text Chunking (400-600 words)
    “
>à LLM Analysis + Prior Knowledge Context
    “
 Schema Validation (Pydantic)
    “
=Ä Knowledge Base Update
    “
=Ê Rich Formatted Output
    “
=¾ Export (JSON)
```

---

## =Ë Output Schema

### Author Entity
```python
{
  "name": "Walter J. Ong",
  "time_period": "1912-2003", 
  "place": "United States",
  "role": "PHILOSOPHER",
  "brief_description": "Jesuit priest who studied orality and literacy"
}
```

### Work Entity
```python
{
  "title": "Orality and Literacy",
  "author": "Walter J. Ong",
  "work_type": "BOOK",
  "publication_info": "1982",
  "significance": "Seminal work on transition from oral to written culture"
}
```

### Topic Entity
```python
{
  "name": "primary orality",
  "field": "PHILOSOPHY", 
  "description": "Cultures that have never known writing",
  "related_concepts": ["secondary orality", "literacy", "consciousness"],
  "historical_context": "Concept developed in 20th century media studies"
}
```

### Knowledge Base Entity (Incremental)
```python
{
  "canonical_name": "Walter J. Ong",
  "entity_type": "person",
  "name_variants": ["Walter J. Ong", "Ong", "W. Ong", "Walter Ong"],
  "attributes": {
    "role": "philosopher", 
    "birth_year": "1912",
    "major_work": "Orality and Literacy"
  },
  "mentions": [
    {"chunk_id": "chunk_001", "mention_text": "Walter J. Ong", "context": "..."},
    {"chunk_id": "chunk_015", "mention_text": "Ong", "context": "..."}
  ],
  "relationships": ["influenced_by: McLuhan", "contemporary_of: Havelock"]
}
```

---

## <¯ Use Cases

### Academic Research
- **Literature Reviews**: Map intellectual landscapes across multiple works
- **Citation Analysis**: Track influence networks and idea genealogies
- **Conceptual Evolution**: Follow how ideas develop across time periods
- **Scholar Mapping**: Build comprehensive profiles of academic figures

### Digital Humanities
- **Corpus Analysis**: Process large collections of academic texts
- **Knowledge Graph Construction**: Create queryable databases of scholarly knowledge
- **Historical Scholarship**: Trace development of ideas across centuries
- **Interdisciplinary Studies**: Map connections between fields

### Educational Applications
- **Study Guides**: Generate structured summaries of complex texts
- **Concept Maps**: Visual representations of idea relationships
- **Historical Timelines**: Chronological organization of scholars and works
- **Research Training**: Teach systematic analysis of academic literature

---

## ™ Configuration

### Model Settings (LLaMA-3.2-3B Optimized)
```python
analyzer = IncrementalHPSAnalyzer(
    model=model,
    tokenizer=tokenizer,
    token_max=900,        # Conservative for reliable JSON generation
    chunk_size=500        # ~750 tokens, captures complete thoughts
)
```

### Chunking Strategy
- **Optimal Size**: 400-600 words per chunk
- **Context Preservation**: Maintains paragraph boundaries when possible  
- **Overlap**: Optional overlap between chunks for continuity
- **Length Handling**: Automatically chunks 10-20 page texts

### Performance Tuning
- **Batch Processing**: Process multiple chunks in parallel
- **Memory Management**: Incremental processing for large corpora
- **Error Recovery**: Automatic retries with reduced token limits
- **Caching**: Save intermediate results for interrupted processing

---

## =' Advanced Features

### Name Matching Algorithm
```python
# Handles variations automatically
"Walter J. Ong" == "Ong" == "W. Ong" == "Walter Ong"
```

**Matching Rules:**
- Last name matching for academic references
- Initial expansion (W. ’ Walter)
- Subset matching (full name contains shorter form)
- Fuzzy matching for OCR errors

### Confidence Scoring
- **Recency Bias**: Later mentions get slight confidence boost
- **Context Quality**: Full descriptions weighted higher than passing references
- **Source Reliability**: Can weight different text sources differently
- **Consensus Building**: Multiple mentions increase confidence

### Relationship Extraction
- **Influence Networks**: "X influenced by Y"
- **Contemporary Relationships**: "X contemporary of Y"
- **Institutional Connections**: "X studied under Y"
- **Collaboration**: "X co-authored with Y"

---

## =Ê Example Output

### Console Output
```
=Ú HPS Chapter Analysis Report
2024-01-15 14:30:25

  Overview                                  
 Main Thesis: Analysis of information       
 theory's historical development            
 Complexity Score: 7/10                      
 Primary Fields: PHILOSOPHY, HISTORY        
                                            

=e Authors & Scholars (4 found):
              ,             ,             ,               
 Name          Role         Period       Description   
              <             <             <               $
 Walter J.Ong  PHILOSOPHER  1912-2003    Jesuit scholar
 Jonathan M.   THEORIST     20th cent.   Mind-language 
 Plato         PHILOSOPHER  Ancient      Warned about 
 Socrates      PHILOSOPHER  Ancient      Oral tradition
              4             4             4               
```

### Knowledge Base Growth
```
Processing chunk_001 (1/15)...
   New entities: 3 | =5 References to known: 0

Processing chunk_005 (5/15)...  
   New entities: 1 | =5 References to known: 2 | = New relationships: 1

Final Knowledge Base Summary
  Incremental Analysis Complete  
 Total Entities: 23              
 " People: 12                    
 " Works: 6                       
 " Topics: 5                     
                                 
 Most Mentioned Entities:        
 " Walter J. Ong (8 mentions)   
 " Plato (4 mentions)           
 " Orality and Literacy (3)     
                                 
```

---

## > Contributing

We welcome contributions! Areas where help is needed:

- **Model Support**: Additional LLM backends (GPT, Claude, Gemini)
- **Schema Extensions**: Domain-specific entity types
- **Visualization**: Web interfaces, network graphs
- **Language Support**: Multi-language academic texts
- **Performance**: Optimization for large-scale processing

---

## =Ä License

MIT License - see LICENSE file for details.

---

## =O Acknowledgments

- **Outlines**: For structured generation capabilities
- **Guidance**: Alternative structured generation framework  
- **Rich**: For beautiful console output
- **Pydantic**: For robust data validation
- **LLaMA**: For the underlying language model capabilities

Built for the digital humanities and computational social science communities.

---

*"Every text tells a story of ideas, people, and discoveries. Let's make those stories machine-readable."* =Ú(