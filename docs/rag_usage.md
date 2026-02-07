# RAG (Retrieval-Augmented Generation) Usage

The RAG pipeline lets you build a searchable index from your documents and answer questions using retrieved context plus a local LLM. All components are open source: **Chroma** (vector store), **sentence-transformers** (embeddings), **rank_bm25** (optional sparse retrieval), and **Hugging Face Transformers** (generation).

## Overview

1. **Ingest** — Chunk documents (from a directory or JSONL), embed with sentence-transformers, and store in Chroma (and optionally build a BM25 index for hybrid search).
2. **Query** — Run the pipeline: retrieve top-k chunks for a question, format as context, and generate an answer with the configured LLM.

The **db-path** is the directory where the Chroma database lives. Optionally, a BM25 index is stored in the same directory (e.g. `rag_db/bm25_index.pkl`) when you use hybrid retrieval.

## Ingest CLI

Build or update the index:

```bash
python -m src.rag.ingest --input-dir data_input --db-path rag_db
```

**Options:**

| Option | Description | Default |
|--------|-------------|--------|
| `--input-dir` | Directory of .txt, .md, .docx files | — |
| `--jsonl` | Path to JSONL (each line: object with `text` or `content` field) | — |
| `--db-path` | Output path for Chroma DB (and BM25 pickle if hybrid) | `rag_db` |
| `--chunk-size` | Token/chunk size for splitting | from config |
| `--overlap` | Overlap between chunks | from config |
| `--embedding-model` | sentence-transformers model name | from config |
| `--use-hybrid` | Build BM25 index for hybrid retrieval | off |
| `--replace` | Clear and rebuild index at db-path (don’t append) | off |

You can combine `--input-dir` and `--jsonl` in one run. Chunk size and overlap default to `RAG_CONFIG` in `config.py`.

**Examples:**

```bash
# Directory only
python -m src.rag.ingest --input-dir data_input --db-path rag_db

# JSONL only (e.g. existing QA or text corpus)
python -m src.rag.ingest --jsonl data_output/qa_cleaned.jsonl --db-path rag_db

# Both, with hybrid search and replace
python -m src.rag.ingest --input-dir data_input --jsonl extra.jsonl --db-path rag_db --use-hybrid --replace
```

## Query CLI

Ask questions against the index:

```bash
# Interactive (read questions from stdin)
python -m src.rag.query --db-path rag_db --interactive

# Single question
python -m src.rag.query --db-path rag_db "What is the main topic of the documents?"
```

**Options:**

| Option | Description | Default |
|--------|-------------|--------|
| `--db-path` | Path to Chroma DB | `rag_db` |
| `--model` | Generator model name (Hugging Face) | from config (tinyllama) |
| `--interactive` | Loop: read questions from stdin | off |
| `question` | One or more words as a single question | — |

If you don’t pass a question and don’t use `--interactive`, the script exits with a short usage message.

## Using the Pipeline in Code

Other code can import and use the pipeline without the CLIs:

```python
from src.rag import RAGPipeline

# Defaults: db_path, model from config (tinyllama), top_k and use_hybrid from RAG_CONFIG
pipeline = RAGPipeline(db_path="rag_db")

# Override model and retrieval
pipeline = RAGPipeline(
    db_path="rag_db",
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    top_k=5,
    use_hybrid=True,
)

answer = pipeline.answer("Your question here")
# Optional: override top_k for this call
answer = pipeline.answer("Another question", top_k=10)
```

The generator can be swapped later (e.g. to a fine-tuned or PEFT-loaded model) as long as it keeps the same interface (`generate(context, question)`).

You can also run the ingest and query CLIs from code by importing and calling `run_ingest` or `run_query` (they use `sys.argv`, so set args before calling or invoke via `python -m src.rag.ingest` / `python -m src.rag.query`):

```python
from src.rag import RAGPipeline, run_ingest, run_query
```

## Configuration

RAG settings live in `config.py` under **RAG_CONFIG**:

- `top_k` — number of chunks to retrieve per query  
- `chunk_size` / `overlap` — used when ingesting from directory or JSONL  
- `embedding_model` — sentence-transformers model (e.g. `sentence-transformers/all-MiniLM-L6-v2`)  
- `use_hybrid` — whether to use dense + BM25 and merge with reciprocal rank fusion  
- `collection_name` — Chroma collection name  

The generator uses the same model config as the rest of the project (e.g. `get_model_config("tinyllama")`); you can override `model_name` when constructing `RAGPipeline` or pass `--model` in the query CLI.

## Dependencies

Ensure these are installed (they are listed in `requirements.txt`):

- `chromadb` — persistent vector store  
- `rank_bm25` — optional BM25 for hybrid retrieval  
- `sentence-transformers` — embeddings  
- `transformers` / `torch` — generator (already used elsewhere in the project)  

Ingest uses the existing `file_processor` (e.g. `process_files`, `chunk_text`, `load_jsonl`) and config chunk/overlap where applicable.
