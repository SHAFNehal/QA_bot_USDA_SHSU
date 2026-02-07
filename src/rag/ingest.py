"""
RAG ingest CLI: build index from input_dir or JSONL, write to db_path.
Uses file_processor for chunking, sentence-transformers for embedding, RAGStore for persistence.
"""

import argparse
import os
from typing import List, Dict, Any

from config import get_rag_config, RAG_CONFIG
from src.utils.file_processor import process_files, load_jsonl, chunk_text, clean_text
from src.rag.store import RAGStore


def chunks_from_input_dir(
    input_dir: str,
    chunk_size: int,
    overlap: int,
) -> List[Dict[str, Any]]:
    """Load documents from directory and return RAG chunks (text + metadata)."""
    processed = process_files(input_dir, chunk_size=chunk_size, overlap=overlap)
    out = []
    for i, p in enumerate(processed):
        out.append({
            "text": p["content"],
            "metadata": {
                "id": f"chunk_{i}",
                "source_file": p["source_file"],
                "chunk_index": p["chunk_index"],
                "total_chunks": p["total_chunks"],
            },
        })
    return out


def chunks_from_jsonl(
    jsonl_path: str,
    chunk_size: int,
    overlap: int,
    text_key: str = "text",
) -> List[Dict[str, Any]]:
    """Load JSONL (each line: object with text field), chunk and return RAG chunks."""
    rows = load_jsonl(jsonl_path)
    out = []
    idx = 0
    for row in rows:
        text = row.get(text_key) or row.get("content") or row.get("answer", "")
        if isinstance(text, dict):
            continue
        text = str(text).strip()
        if not text:
            continue
        text = clean_text(text)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, ch in enumerate(chunks):
            out.append({
                "text": ch,
                "metadata": {
                    "id": f"chunk_{idx}",
                    "source_row": row.get("source_file", idx),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            })
            idx += 1
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build RAG index from documents (directory or JSONL) and write to db_path."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing .txt, .md, .docx files",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="Path to JSONL file (each line: object with 'text' or 'content' field)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="rag_db",
        help="Output path for Chroma DB (and optional BM25 pickle)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=f"Chunk size (default from config: {RAG_CONFIG.get('chunk_size', 1000)})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help=f"Chunk overlap (default from config: {RAG_CONFIG.get('overlap', 100)})",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Sentence-transformers model name (default from config)",
    )
    parser.add_argument(
        "--use-hybrid",
        action="store_true",
        help="Build BM25 index for hybrid retrieval",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing index at db_path instead of appending",
    )
    args = parser.parse_args()

    rag_cfg = get_rag_config()
    chunk_size = args.chunk_size or rag_cfg["chunk_size"]
    overlap = args.overlap or rag_cfg["overlap"]
    embedding_model = args.embedding_model or rag_cfg["embedding_model"]

    chunks: List[Dict[str, Any]] = []
    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            raise SystemExit(f"Input directory not found: {args.input_dir}")
        chunks.extend(chunks_from_input_dir(args.input_dir, chunk_size, overlap))
        print(f"Loaded {len(chunks)} chunks from directory {args.input_dir}")
    if args.jsonl:
        if not os.path.isfile(args.jsonl):
            raise SystemExit(f"JSONL file not found: {args.jsonl}")
        from_jsonl = chunks_from_jsonl(args.jsonl, chunk_size, overlap)
        chunks.extend(from_jsonl)
        print(f"Loaded {len(from_jsonl)} chunks from JSONL {args.jsonl}")

    if not chunks:
        raise SystemExit("No chunks to ingest. Provide --input-dir and/or --jsonl with valid data.")

    store = RAGStore(
        db_path=args.db_path,
        embedding_model_name=embedding_model,
        collection_name=rag_cfg.get("collection_name", "rag_docs"),
        use_hybrid=args.use_hybrid,
    )
    store.add_documents(chunks, replace=args.replace)
    print(f"Indexed {len(chunks)} chunks at {args.db_path}")


if __name__ == "__main__":
    main()
