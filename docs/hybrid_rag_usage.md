# Hybrid RAG Usage

The **Hybrid RAG** pipeline uses a small LLM to restate your question in multiple ways, runs both RAG (with merged retrieval over those queries) and your fine-tuned model on the same questions, then has the small LLM synthesize one final answer from all candidates.

## When to use it

- You have built a RAG index (`python -m src.rag.ingest ...`) and have a **fine-tuned model** (LoRA weights).
- You want one high-confidence answer that combines retrieval-based and model-based responses.

## Flow

1. **Restate:** A small LLM (e.g. TinyLlama) rewrites the question in 5 different ways (same meaning).
2. **RAG branch:** The original question + 5 restatements are used to query the vector store; results are merged with reciprocal rank fusion, then one RAG answer is generated from the merged context.
3. **Fine-tuned branch:** The same 6 questions are sent to your fine-tuned model (no conversation history); you get 6 FT answers.
4. **Synthesize:** The small LLM receives the question, the 1 RAG answer, and the 6 FT answers, and produces one final answer (combining or choosing the best information).

## CLI

```bash
python -m src.Hybrid_RAG.query --db-path rag_db --peft-model fine_tuned_weights --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --interactive
```

**Required:**

- `--peft-model` — Path to your fine-tuned model weights (LoRA adapter).
- `--base-model` — Base model name (must match the model you fine-tuned).

**Optional:**

- `--db-path` — Path to the Chroma RAG database (default: `rag_db`).
- `--small-llm` — Model name for the small LLM used for restating and synthesizing (default from `HYBRID_RAG_CONFIG` in config.py).
- `--interactive` — Read questions from stdin in a loop. Without this, pass a single question as a positional argument.

**Examples:**

```bash
# Single question
python -m src.Hybrid_RAG.query --db-path rag_db --peft-model fine_tuned_weights --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 "What is the main topic?"

# Interactive
python -m src.Hybrid_RAG.query --db-path rag_db --peft-model fine_tuned_weights --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --interactive
```

## Use in code

```python
from src.Hybrid_RAG import HybridRAGPipeline

pipeline = HybridRAGPipeline(
    db_path="rag_db",
    peft_model_path="fine_tuned_weights",
    base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    small_llm_name=None,  # optional; default from config
)
answer = pipeline.answer("Your question here")
```

## Configuration

In `config.py`, **HYBRID_RAG_CONFIG** controls:

- `small_llm_model` — Model used for restating and synthesizing.
- `num_restatements` — Number of restatements (default: 5).
- `per_query_top_k` — Chunks retrieved per query before merge (e.g. 5).
- `merged_retrieval_top_k` — Total chunks after RRF merge (e.g. 12).
- `max_new_tokens_restater` / `max_new_tokens_synthesizer` — Generation limits.
- `temperature` — Sampling for the small LLM.

Use `get_hybrid_rag_config(**overrides)` to read defaults and override as needed.
