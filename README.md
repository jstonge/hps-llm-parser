# HPS LLM Parser

Intelligent extraction and analysis of scholarly information from History and Philosophy of Science texts using Large Language Models.

Transform dense academic texts into structured, interconnected knowledge graphs that track authors, works, concepts, and their relationships across entire documents.

## What This Does

The HPS LLM Parser analyzes academic texts and extracts:

- **Authors & Scholars**: People mentioned with their roles, time periods, and contributions
- **Works & Publications**: Books, papers, theories with authorship and significance  
- **Concepts & Topics**: Ideas, methodologies, research areas with relationships
- **Intellectual Networks**: Who influenced whom, citation relationships
- **Historical Context**: Time periods, locations, academic institutions

## Key Features

### Incremental Knowledge Building
- **Persistent Memory**: Remembers entities across text chunks
- **Smart Name Matching**: "Walter J. Ong" -> "Ong" -> "W. Ong" (same person)
- **Progressive Enhancement**: Each mention adds information to existing entities
- **Confidence Scoring**: Weights information by reliability and recency

### Structured Output Guaranteed
- **JSON Schema Enforcement**: Uses Outlines/Guidance for reliable data extraction
- **Pydantic Validation**: Type-safe, validated output structures
- **No Parsing Errors**: Structured generation prevents malformed JSON

### Rich Analysis & Visualization
- **Beautiful Console Output**: Color-coded tables and panels via Rich
- **Entity Relationship Maps**: Track intellectual genealogies  
- **Mention Tracking**: See where each entity appears across the text
- **Exportable Knowledge Base**: Save results as JSON for further analysis

## Quick Start

### Installation
```bash
pip install outlines transformers torch pydantic rich
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

## How It Works

The system processes text in chunks, maintaining a persistent knowledge base:

1. **Text Chunking**: Splits long texts into 400-600 word segments
2. **LLM Analysis**: Each chunk analyzed with context of prior knowledge
3. **Entity Matching**: Links name variations to existing entities
4. **Knowledge Update**: Adds new information to entity profiles
5. **Rich Output**: Displays results in formatted tables and panels

### Example: Incremental Learning

```
Chunk 1: "Walter J. Ong was born in 1912..."
-> Creates: Walter J. Ong (person, born 1912)

Chunk 5: "Ong argued that writing..."  
-> Recognizes: This is the same Walter J. Ong
-> Updates: Adds "argued about writing" to his profile

Chunk 10: "As Ong noted earlier..."
-> Links: Another reference to same person
-> Builds: Complete profile across all mentions
```

## Output Schema

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
    {"chunk_id": "chunk_001", "mention_text": "Walter J. Ong"},
    {"chunk_id": "chunk_015", "mention_text": "Ong"}
  ]
}
```

## Configuration

### Optimized for LLaMA-3.2-3B
```python
analyzer = IncrementalHPSAnalyzer(
    model=model,
    tokenizer=tokenizer,
    token_max=900,        # Conservative for reliable JSON generation
    chunk_size=500        # Captures complete thoughts
)
```

## Advanced Features

### Name Matching Algorithm
- Handles academic reference variations automatically
- "Walter J. Ong" == "Ong" == "W. Ong" == "Walter Ong"
- Last name matching for academic citations
- Fuzzy matching for OCR errors

### Confidence Scoring
- Later mentions get slight confidence boost
- Full descriptions weighted higher than passing references
- Multiple mentions increase confidence

## Use Cases

- **Academic Research**: Map intellectual landscapes across multiple works
- **Digital Humanities**: Process large collections of academic texts
- **Literature Reviews**: Track influence networks and idea genealogies
- **Educational Applications**: Generate structured summaries of complex texts

## Files

- `hps_analyzer.py`: Core analysis functionality
- `incremental_hps_analyzer.py`: Incremental knowledge building system
- `test-hps-analyzer.ipynb`: Basic usage examples
- `test-incremental-analyzer.ipynb`: Incremental analysis examples
- `guidance-hps.ipynb`: Guidance-based implementation
- `outlines-hps.ipynb`: Outlines-based implementation

## License

MIT License

## Acknowledgments

Built for the digital humanities and computational social science communities.

*"Every text tells a story of ideas, people, and discoveries. Let's make those stories machine-readable."*