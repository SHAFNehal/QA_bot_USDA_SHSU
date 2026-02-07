"""
RAG (Retrieval-Augmented Generation) module.

Standalone pipeline: persistent vector store (Chroma) + optional BM25 hybrid,
config-driven LLM generator. Use for document Q&A with retrieval.
"""

from src.rag.pipeline import RAGPipeline
from src.rag.store import RAGStore
from src.rag.ingest import main as run_ingest
from src.rag.query import main as run_query

__all__ = ["RAGPipeline", "RAGStore", "run_ingest", "run_query"]
