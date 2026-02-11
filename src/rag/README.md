# RAG (Retrieval-Augmented Generation) Module

This module implements a RAG system that retrieves relevant documents from a vector database and uses them to generate contextually accurate answers.

## Features

- **Vector Database**: Chroma for efficient semantic search
- **Hybrid Retrieval** (Optional): Combines Chroma + BM25 for better results
- **Multi-format Support**: Processes .txt, .md, .docx files
- **Interactive CLI**: Chat interface with conversation history
- **Evaluation**: Comprehensive metrics for retrieval and answer quality

---

## Quick Start

### 1. Ingest Documents

```bash
# Ingest documents from data_input/
python src/rag/ingest.py --input_dir data_input --db_path rag_db

# Custom paths
python src/rag/ingest.py --input_dir my_docs --db_path my_rag_db
```

### 2. Query RAG System

```bash
# Interactive CLI
python src/rag/query.py --db_path rag_db

# Single query
python src/rag/query.py --db_path rag_db --query "What is integrated pest management?"
```

### 3. Evaluate RAG System

```bash
# Evaluate on test data
python src/rag/evaluate_rag.py \
    --test_data data_output/training_test.jsonl \
    --db_path rag_db \
    --output data_output/rag_evaluation_report.json
```

---

## Module Structure

```
src/rag/
├── __init__.py           # Module initialization
├── ingest.py             # Document ingestion script
├── query.py              # Query/inference script (CLI)
├── store.py              # Vector database management
├── generator.py          # Response generation
├── pipeline.py           # RAG pipeline orchestration
├── evaluate_rag.py       # Evaluation script
└── README.md             # This file
```

---

## Components

### Document Ingestion (`ingest.py`)

Processes documents and creates vector database:

```python
from src.rag.store import DocumentStore

store = DocumentStore(db_path="rag_db")
store.ingest_directory("data_input")
```

**Features:**
- Recursive directory processing
- Multi-format support (.txt, .md, .docx)
- Chunking (512 tokens)
- Progress tracking
- Metadata preservation

### Query System (`query.py`)

Interactive or programmatic querying:

```python
from src.rag.query import RAGQuery

rag = RAGQuery(db_path="rag_db")
response, sources = rag.query("What is IPM?")
```

**Features:**
- Semantic search with Chroma
- Optional BM25 hybrid retrieval
- Conversation history
- Source citation
- Configurable top-k

### Evaluation (`evaluate_rag.py`)

Comprehensive RAG evaluation:

**Metrics:**
- **Retrieval**: Precision, recall, MRR, success rate
- **Answer Quality**: BLEU, ROUGE, embedding similarity
- **Performance**: Response time

```bash
python src/rag/evaluate_rag.py \
    --test_data data_output/training_test.jsonl \
    --db_path rag_db
```

---

## Configuration

### Default Parameters

```python
{
    "db_path": "rag_db",
    "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "embedding_model": "all-MiniLM-L6-v2",
    "chunk_size": 512,
    "top_k": 5,
    "use_bm25": False,  # Enable for hybrid retrieval
    "similarity_threshold": 0.3
}
```

### Customization

Override defaults via command-line arguments:

```bash
# Use larger retrieval set
python src/rag/query.py --db_path rag_db --top_k 10

# Enable hybrid retrieval (Chroma + BM25)
python src/rag/query.py --db_path rag_db --use_bm25

# Use different base model
python src/rag/query.py --base_model "microsoft/Phi-3-mini-4k-instruct"
```

---

## Hybrid Retrieval (Optional)

Combines dense (Chroma) and sparse (BM25) retrieval:

```bash
# Ingest with BM25 indexing
python src/rag/ingest.py --input_dir data_input --db_path rag_db --use_bm25

# Query with hybrid retrieval
python src/rag/query.py --db_path rag_db --use_bm25
```

**Benefits:**
- Better keyword matching (BM25)
- Better semantic matching (Chroma)
- More robust retrieval

---

## Performance

### Typical Performance

| Metric | Value |
|--------|-------|
| Ingestion | ~100 docs/minute |
| Query Response Time | 2-4 seconds |
| Retrieval Accuracy | 70-85% |
| Memory Usage | ~2GB (for 1000 docs) |

### Optimization Tips

1. **GPU Acceleration**: Install sentence-transformers with CUDA
2. **Batch Ingestion**: Process documents in batches
3. **Cache Embeddings**: Reuse embeddings when possible
4. **Adjust top-k**: Balance between accuracy and speed

---

## Examples

### Example 1: Basic Usage

```python
from src.rag.query import RAGQuery

# Initialize
rag = RAGQuery(db_path="rag_db")

# Query
response, sources = rag.query("What are beneficial insects?")

print("Answer:", response)
print("\nSources:")
for source in sources:
    print(f"  - {source['source']}: {source['content'][:100]}...")
```

### Example 2: Batch Evaluation

```python
from src.rag.evaluate_rag import RAGEvaluator

# Initialize evaluator
evaluator = RAGEvaluator(
    db_path="rag_db",
    use_embedding=True,
    top_k=5
)

# Run evaluation
report = evaluator.run_evaluation("data_output/training_test.jsonl")

# Print results
evaluator.print_report()
evaluator.save_report("rag_evaluation.json")
```

---

## Troubleshooting

### Common Issues

**1. "No embedding function provided"**
```bash
# Install sentence-transformers
pip install sentence-transformers
```

**2. "Database not found"**
```bash
# Create database first
python src/rag/ingest.py --input_dir data_input --db_path rag_db
```

**3. "Out of memory during ingestion"**
```bash
# Reduce batch size (edit ingest.py)
# Or process fewer documents at once
```

**4. "Slow query response"**
```bash
# Reduce top_k
python src/rag/query.py --top_k 3

# Or use GPU for embeddings
export CUDA_VISIBLE_DEVICES=0
```

---

## API Reference

### DocumentStore

```python
class DocumentStore:
    def __init__(self, db_path: str, embedding_model: str = "all-MiniLM-L6-v2")
    def ingest_directory(self, directory: str, use_bm25: bool = False)
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]
```

### RAGQuery

```python
class RAGQuery:
    def __init__(self, db_path: str, base_model: str, use_bm25: bool = False, top_k: int = 5)
    def query(self, question: str) -> Tuple[str, List[Dict]]
    def clear_history(self)
```

### RAGEvaluator

```python
class RAGEvaluator:
    def __init__(self, db_path: str, base_model: str, use_embedding: bool = True)
    def run_evaluation(self, test_data_path: str) -> RAGEvaluationReport
    def print_report(self)
    def save_report(self, output_path: str)
```

---

## Integration

### With Fine-tuned Model

See `src/Hybrid_RAG/` for combining RAG with fine-tuned models.

### With Streamlit GUI

```bash
cd GUI
streamlit run app.py
# Select "RAG" mode
```

### With Pipeline

```bash
# Full pipeline with RAG
./scripts/run_full_pipeline_with_rag.sh
```

---
