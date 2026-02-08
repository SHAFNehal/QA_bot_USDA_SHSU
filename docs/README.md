# Documentation Index

This folder contains detailed guides for specific parts of the QA Bot pipeline. The main [README.md](../README.md) in the project root covers installation, quick start, and full workflow.

## Guides

| Document | Description |
|----------|-------------|
| [qa_generation_guide.md](qa_generation_guide.md) | How to choose the number of question-answer pairs per chunk; quality vs. speed; config and CLI options. |
| [rag_usage.md](rag_usage.md) | RAG pipeline: ingest (build index from documents or JSONL), query (CLI and Python API), configuration. |
| [hybrid_rag_usage.md](hybrid_rag_usage.md) | Hybrid RAG: restate question → RAG + fine-tuned model → synthesize final answer; CLI and code. |
| [evaluation.md](evaluation.md) | Evaluation metrics (BLEU, ROUGE, embedding similarity, exact match, token F1), holdout set, and how to run evaluation (pipeline, eval-only, or on documents). |

## Other documentation

- **Chat GUI** — [GUI/README.md](../GUI/README.md): How to run the Streamlit chat app and edit `gui_config.env` (Fine-tuned / RAG / Hybrid RAG).
- **Main README** — [README.md](../README.md): Installation, full pipeline, step-by-step usage, configuration, troubleshooting, examples.
