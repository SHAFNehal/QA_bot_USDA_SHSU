# Evaluation Guide

This document describes how model evaluation works: metrics, the holdout test set, and how to run evaluation (fine-tuned model, RAG, Hybrid RAG).

## Evaluation Metrics

The pipeline computes the following metrics when comparing model answers to reference answers. All are implemented in `src/inference/metrics.py`.

| Metric | Description | Range |
|--------|-------------|--------|
| **BLEU** | N-gram overlap (precision-oriented) | 0–1 (higher is better) |
| **ROUGE-1 / ROUGE-2 / ROUGE-L** | N-gram and longest-common-subsequence F1 | 0–1 (higher is better) |
| **Embedding similarity** | Cosine similarity of sentence embeddings (e.g. all-MiniLM-L6-v2) | 0–1 (higher is better) |
| **Exact match** | Binary: normalized candidate equals reference | 0 or 1 |
| **Token F1** | Token-level precision/recall F1 | 0–1 (higher is better) |
| **LLM-as-judge** (optional) | Model-based score; enable with `--llm_judge` | Configurable |

If a dependency is missing (e.g. `rouge_score`, `nltk`, `sentence-transformers`), that metric is skipped. Install `requirements.txt` to ensure all metrics are available.

## Data-Driven Evaluation (Fine-Tuned Model)

All evaluation tests are **dynamically generated from the test data file**. No preset greeting/gratitude tests.

- **Holdout**: Full test set with BLEU, ROUGE, embedding sim, exact match, token F1
- **Single-turn questions**: Up to 10 single-turn Q&A from the data
- **Multi-turn conversations**: Up to 4 multi-turn conversations from the data

`--test_data` is **required**. All tests use this file.

## Running Fine-Tuned Model Evaluation

### Holdout only (test_data_only)

```bash
python src/inference/evaluate.py \
  --model_path fine_tuned_weights \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --test_data data_output/test_set_cleaned.jsonl \
  --test_data_only \
  --output data_output/evaluation_report.json
```

### Full suite (holdout + single-turn + multi-turn)

```bash
python src/inference/evaluate.py \
  --model_path fine_tuned_weights \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --test_data data_output/test_set_cleaned.jsonl \
  --output data_output/evaluation_report.json
```

### SLURM (eval_on_test_set.slurm)

```bash
MODEL_PATH=/path/to/peft BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.2 sbatch scripts/eval_on_test_set.slurm
```

### Optional flags

- `--no_embedding`: Disable embedding similarity metric
- `--llm_judge`: Enable LLM-as-judge (slower)
- `--max_new_tokens 512`: Max tokens for generation (default: 512)

## RAG Evaluation

Evaluates the RAG system (retrieval + generation) on answer quality and retrieval metrics.

```bash
python src/rag/evaluate_rag.py \
  --test_data data_output/test_set_cleaned.jsonl \
  --db_path rag_db \
  --output data_output/rag_evaluation_report.json
```

Options: `--use_bm25`, `--top_k`, `--no_embedding`

## Hybrid RAG Evaluation

Evaluates the Hybrid RAG system (fine-tuned + RAG + synthesis) on answer quality.

```bash
python src/Hybrid_RAG/evaluate_hybrid_rag.py \
  --test_data data_output/test_set_cleaned.jsonl \
  --model_path fine_tuned_weights \
  --db_path rag_db_hybrid \
  --output data_output/hybrid_rag_evaluation_report.json
```

Options: `--base_model`, `--no_embedding`

## Report Format

The evaluation report (JSON) includes:

- Aggregate metrics over the test set (BLEU, ROUGE, embedding_sim, exact_match, token_f1)
- Per-example results when applicable
- Category scores (holdout, rephrased_questions, multiturn)
- Success criteria check
