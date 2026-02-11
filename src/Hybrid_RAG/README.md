# Hybrid RAG Module

This module combines **fine-tuned LLM** with **RAG (Retrieval-Augmented Generation)** to leverage both domain-specific knowledge and up-to-date document retrieval.

## Concept

**Hybrid RAG = Fine-tuned Model + RAG**

1. **Fine-tuned LLM**: Answers with domain-specific knowledge (fast, specialized)
2. **RAG**: Retrieves relevant documents and generates answer (updatable, grounded)
3. **Synthesizer**: Intelligently merges both answers for optimal response

**Benefits:**
- Domain expertise from fine-tuned model
- Fresh information from RAG
- Grounded responses with source citations
- Best of both worlds

---

## Quick Start

### 1. Prerequisites

You need both:
- Fine-tuned model (from training pipeline)
- RAG database (from document ingestion)

```bash
# Step 1: Fine-tune model
./scripts/run_pipeline.sh

# Step 2: Create RAG database
python src/rag/ingest.py --input_dir data_input --db_path rag_db
```

### 2. Setup Hybrid RAG

```bash
# Create hybrid RAG configuration
python src/Hybrid_RAG/pipeline.py \
    --model_dir data_output/final_model \
    --rag_db rag_db \
    --hybrid_db hybrid_rag_db
```

### 3. Query Hybrid RAG

```bash
# Interactive CLI
python src/Hybrid_RAG/query.py \
    --model_dir data_output/final_model \
    --db_path hybrid_rag_db

# Single query
python src/Hybrid_RAG/query.py \
    --model_dir data_output/final_model \
    --db_path hybrid_rag_db \
    --query "What are biological control methods?"
```

### 4. Evaluate Hybrid RAG

```bash
python src/Hybrid_RAG/evaluate_hybrid_rag.py \
    --test_data data_output/training_test.jsonl \
    --model_dir data_output/final_model \
    --db_path hybrid_rag_db \
    --output data_output/hybrid_evaluation_report.json
```

---

## Module Structure

```
src/Hybrid_RAG/
├── __init__.py                  # Module initialization
├── pipeline.py                  # Hybrid RAG setup
├── query.py                     # Query/inference script
├── synthesizer.py               # Answer merging logic
├── restater.py                  # Query reformulation
├── small_llm.py                 # LLM utilities
├── evaluate_hybrid_rag.py       # Evaluation script
└── README.md                    # This file
```

---

## How It Works

### Query Flow

```
User Question
    ↓
[1] Fine-tuned Model → Answer A (domain knowledge)
    ↓
[2] RAG Retrieval → Documents
    ↓
[3] RAG Generation → Answer B (grounded in docs)
    ↓
[4] Synthesizer → Merged Answer (A + B)
    ↓
Final Response + Sources
```

### Synthesis Strategy

The synthesizer intelligently merges answers based on:

1. **Confidence Scoring**: Which answer is more confident?
2. **Relevance**: Which answer better addresses the question?
3. **Completeness**: Which provides more detail?
4. **Source Grounding**: Is RAG answer backed by sources?

---

## Components

### Pipeline (`pipeline.py`)

Sets up Hybrid RAG environment:

```python
from src.Hybrid_RAG.pipeline import HybridRAGPipeline

pipeline = HybridRAGPipeline(
    model_dir="data_output/final_model",
    rag_db="rag_db",
    hybrid_db="hybrid_rag_db"
)
pipeline.setup()
```

### Query System (`query.py`)

Handles user queries with both systems:

```python
from src.Hybrid_RAG.query import HybridRAGQuery

hybrid = HybridRAGQuery(
    model_dir="data_output/final_model",
    db_path="hybrid_rag_db"
)

response = hybrid.query("What is IPM?")
print(response["merged_answer"])
print("Sources:", response["sources"])
```

**Returns:**
```python
{
    "fine_tuned_answer": "...",      # From fine-tuned model
    "rag_answer": "...",              # From RAG
    "merged_answer": "...",           # Final synthesis
    "sources": [...],                 # RAG source documents
    "confidence_scores": {...}        # Model confidence
}
```

### Synthesizer (`synthesizer.py`)

Merges answers intelligently:

```python
from src.Hybrid_RAG.synthesizer import AnswerSynthesizer

synth = AnswerSynthesizer(model_name="meta-llama/Llama-3.2-1B-Instruct")

merged = synth.synthesize(
    question="What is biological control?",
    fine_tuned_answer="...",
    rag_answer="...",
    sources=[...]
)
```

**Synthesis Modes:**
- `concatenate`: Combine both answers
- `weighted`: Weighted by confidence
- `llm_merge`: Use LLM to merge (best quality)

### Evaluation (`evaluate_hybrid_rag.py`)

Comprehensive evaluation tracking both components:

```python
from src.Hybrid_RAG.evaluate_hybrid_rag import HybridRAGEvaluator

evaluator = HybridRAGEvaluator(
    model_dir="data_output/final_model",
    db_path="hybrid_rag_db",
    use_embedding=True
)

report = evaluator.run_evaluation("data_output/training_test.jsonl")
evaluator.print_report()
```

**Metrics:**
- Fine-tuned answer quality
- RAG answer quality
- Merged answer quality
- Response time (slower than either alone)
- Source quality

---

## Configuration

### Default Parameters

```python
{
    "model_dir": "data_output/final_model",
    "db_path": "hybrid_rag_db",
    "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "embedding_model": "all-MiniLM-L6-v2",
    "top_k": 5,
    "synthesis_mode": "llm_merge",
    "response_threshold": 8.0  # seconds
}
```

### Customization

```bash
# Use more retrieved documents
python src/Hybrid_RAG/query.py \
    --model_dir data_output/final_model \
    --db_path hybrid_rag_db \
    --top_k 10

# Change synthesis strategy
# Edit synthesizer.py, change SYNTHESIS_MODE
```

---

## Examples

### Example 1: Basic Query

```python
from src.Hybrid_RAG.query import HybridRAGQuery

# Initialize
hybrid = HybridRAGQuery(
    model_dir="data_output/final_model",
    db_path="hybrid_rag_db"
)

# Query
result = hybrid.query("What are natural enemies of aphids?")

print("Fine-tuned says:", result["fine_tuned_answer"])
print("RAG says:", result["rag_answer"])
print("\nFinal answer:", result["merged_answer"])
print("\nSources:")
for source in result["sources"]:
    print(f"  - {source['source']}")
```

### Example 2: Batch Evaluation

```python
from src.Hybrid_RAG.evaluate_hybrid_rag import HybridRAGEvaluator

# Initialize
evaluator = HybridRAGEvaluator(
    model_dir="data_output/final_model",
    db_path="hybrid_rag_db",
    use_embedding=True
)

# Run evaluation
report = evaluator.run_evaluation("data_output/training_test.jsonl")

# Results
print(f"Fine-tuned accuracy: {report.fine_tuned_metrics.overall_accuracy:.2%}")
print(f"RAG accuracy: {report.rag_metrics.overall_accuracy:.2%}")
print(f"Merged accuracy: {report.merged_metrics.overall_accuracy:.2%}")
print(f"Average response time: {report.performance.avg_response_time:.2f}s")
```

---

## Troubleshooting

### Common Issues

**1. "Model directory not found"**
```bash
# Train model first
./scripts/run_pipeline.sh
```

**2. "RAG database not found"**
```bash
# Create RAG database first
python src/rag/ingest.py --input_dir data_input --db_path rag_db
```

**3. "Out of memory"**
```bash
# Hybrid RAG needs ~4GB RAM minimum
# Close other applications or use smaller batch sizes
```

**4. "Slow response time"**
```bash
# Expected: 5-8 seconds (combines both systems)
# To speed up: Use GPU, reduce top_k, or use fine-tuned/RAG alone
```

**5. "Poor synthesis quality"**
```bash
# Try different synthesis modes in synthesizer.py
# Or use larger synthesis model (currently using Llama-3.2-1B)
```

---

## API Reference

### HybridRAGPipeline

```python
class HybridRAGPipeline:
    def __init__(self, model_dir: str, rag_db: str, hybrid_db: str)
    def setup(self) -> bool
    def validate(self) -> bool
```

### HybridRAGQuery

```python
class HybridRAGQuery:
    def __init__(self, model_dir: str, db_path: str, top_k: int = 5)
    def query(self, question: str) -> Dict[str, Any]
    def clear_history(self)
```

### AnswerSynthesizer

```python
class AnswerSynthesizer:
    def __init__(self, model_name: str)
    def synthesize(
        self, 
        question: str, 
        fine_tuned_answer: str, 
        rag_answer: str, 
        sources: List[Dict]
    ) -> str
```

### HybridRAGEvaluator

```python
class HybridRAGEvaluator:
    def __init__(self, model_dir: str, db_path: str, use_embedding: bool = True)
    def run_evaluation(self, test_data_path: str) -> HybridRAGEvaluationReport
    def print_report(self)
    def save_report(self, output_path: str)
```

---

## Integration

### With Streamlit GUI

```bash
cd GUI
streamlit run app.py
# Select "Hybrid RAG" mode in dropdown
```

### With Pipeline

```bash
# Full pipeline: Fine-tuning → RAG → Hybrid RAG
./scripts/run_full_pipeline_with_rag.sh
```

### With Comparison Tool

```bash
# Compare all three systems
./scripts/run_comparison.sh
```

---