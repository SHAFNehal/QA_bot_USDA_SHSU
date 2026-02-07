"""
RAG pipeline: load store + generator, answer(question) = retrieve -> format context -> generate.
No imports from inference or training; uses config, model_utils, file_processor, rag.store, rag.generator.
"""

from typing import Optional

from config import get_rag_config, get_model_config
from src.rag.store import RAGStore
from src.rag.generator import RAGGenerator


def _format_context(chunks: list) -> str:
    """Turn list of chunks (dict with 'text' and optionally 'metadata') into a single context string."""
    parts = []
    for i, c in enumerate(chunks):
        text = c.get("text", "").strip()
        if text:
            parts.append(text)
    return "\n\n---\n\n".join(parts) if parts else ""


class RAGPipeline:
    """
    End-to-end RAG: load Chroma (and optional BM25) from db_path,
    load embedding model and LLM generator; answer(question) retrieves and generates.
    """

    def __init__(
        self,
        db_path: str,
        model_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        use_hybrid: Optional[bool] = None,
        top_k: Optional[int] = None,
        model_key: str = "tinyllama",
    ):
        rag_cfg = get_rag_config()
        self.db_path = db_path
        self.top_k = top_k if top_k is not None else rag_cfg["top_k"]
        self.use_hybrid = use_hybrid if use_hybrid is not None else rag_cfg.get("use_hybrid", False)
        emb = embedding_model or rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        coll = rag_cfg.get("collection_name", "rag_docs")

        self._store = RAGStore(
            db_path=db_path,
            embedding_model_name=emb,
            collection_name=coll,
            use_hybrid=self.use_hybrid,
        )
        model_cfg = get_model_config(model_key)
        name = model_name or model_cfg.get("name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self._generator = RAGGenerator(
            model_name=name,
            max_new_tokens=model_cfg.get("max_length", 512),
            temperature=model_cfg.get("temperature", 0.7),
        )

    def answer(self, question: str, top_k: Optional[int] = None) -> str:
        """
        Retrieve top_k chunks for question, format as context, generate and return answer.
        """
        k = top_k if top_k is not None else self.top_k
        chunks = self._store.search(question, top_k=k, use_hybrid=self.use_hybrid)
        context = _format_context(chunks)
        if not context.strip():
            return "No relevant context was found for this question."
        return self._generator.generate(context, question)
