# Evaluation Guide

This document describes how model evaluation works: metrics, the holdout test set, and how to run evaluation (full pipeline, eval-only, or on documents).

## Evaluation Metrics

The pipeline computes the following metrics when comparing model answers to reference answers (e.g. on the holdout set or a custom test file). All are implemented in `src/inference/metrics.py`.

| Metric | Description | Range |
|--------|-------------|--------|
| **BLEU** | N-gram overlap (precision-oriented) | 0–1 (higher is better) |
| **ROUGE-1 / ROUGE-2 / ROUGE-L** | N-gram and longest-common-subsequence F1 | 0–1 (higher is better) |
| **Embedding similarity** | Cosine similarity of sentence embeddings (e.g. all-MiniLM-L6-v2) | 0–1 (higher is better) |
| **Exact match** | Binary: normalized candidate equals reference | 0 or 1 |
| **Token F1** | Token-level precision/recall F1 | 0–1 (higher is better) |
| **LLM-as-judge** (optional) | Model-based score; enable with `--use-llm-judge` | Configurable |

If a dependency is missing (e.g. `rouge_score`, `nltk`, `sentence-transformers`), that metric is skipped and not included in the report. Install `requirements.txt` to ensure all metrics are available.

## Train / Validation / Holdout Split

Before training, the merged dataset is split into:

- **80% train** — used for gradient updates
- **10% validation** — used for early stopping (validation loss)
- **10% holdout** — **not** used during training; saved to a JSONL file (e.g. `data_output/training_test.jsonl`) and used only for evaluation

The holdout set gives an unbiased estimate of performance on unseen data. The pipeline script saves it automatically and passes it to the evaluation step.

## When Evaluation Runs

1. **Full pipeline** (`./scripts/run_pipeline.sh`): After training, evaluation runs on (a) the holdout set (`training_test.jsonl`) and (b) preset tests (greetings, gratitude, paraphrases, follow-ups). The report is written to the output directory.

2. **Eval-only** (`./scripts/run_pipeline.sh --eval-only`): Uses the existing model and, if present, the holdout file in the output directory. Same metrics and report format.

3. **Evaluate on a specific test file**: Run the evaluation script directly with `--test_data path/to/test.jsonl` and optionally `--test_data_only` to skip preset tests.

4. **Evaluate on documents** (`./scripts/run_eval_on_documents.sh`): Generates QA from a document folder, then evaluates the model **only** on that generated QA (no preset tests). Uses the same metrics. Output: `generated_qa_eval.jsonl` and `evaluation_report.json` in the chosen output directory.

## Running Evaluation

### After training (holdout + preset tests)

```bash
python src/inference/evaluate.py \
  --model_path fine_tuned_weights \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --test_data data_output/training_test.jsonl \
  --output data_output/evaluation_report.json
```

- Omit `--test_data` to run only preset tests (no holdout).
- Add `--test_data_only` to run **only** on the file given by `--test_data` (no greeting/paraphrase/follow-up tests).

### Evaluate existing model on a document folder

```bash
./scripts/run_eval_on_documents.sh \
  --peft-model fine_tuned_weights \
  --input-dir ./my_docs \
  --output-dir ./eval_output
```

This generates QA from `my_docs`, runs the model on those questions, compares to the generated references with the metrics above, and writes `eval_output/evaluation_report.json` and `eval_output/generated_qa_eval.jsonl`.

### Optional: LLM-as-judge

If you want to use an LLM to score answers (e.g. for alignment or preference), pass `--use-llm-judge` and the appropriate model/API options as documented in the evaluation script help:

```bash
python src/inference/evaluate.py --help
```

## Report Format

The evaluation report (JSON) includes:

- Aggregate metrics over the test set (e.g. average BLEU, ROUGE, embedding similarity, exact match, token F1)
- Per-example results when applicable
- Summary of which tests were run (holdout, greetings, etc.)

Use the same report format for consistency whether you run on the holdout set, a custom JSONL, or document-generated QA.
