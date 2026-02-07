"""
Hybrid RAG pipeline: restate (small LLM) -> RAG (multi-query merge) + FT -> synthesize (small LLM).
"""

from typing import List, Dict, Any, Optional

from config import get_hybrid_rag_config, get_rag_config
from src.rag.store import RAGStore
from src.rag.generator import RAGGenerator
from src.inference.inference import QAInference

from src.Hybrid_RAG.small_llm import SmallLLM
from src.Hybrid_RAG.restater import Restater
from src.Hybrid_RAG.synthesizer import Synthesizer


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """Turn list of chunks (text + metadata) into a single context string."""
    parts = [c.get("text", "").strip() for c in chunks if c.get("text", "").strip()]
    return "\n\n---\n\n".join(parts) if parts else ""


def _merge_retrievals_rrf(
    store: RAGStore,
    queries: List[str],
    per_query_top_k: int,
    merged_top_k: int,
) -> List[Dict[str, Any]]:
    """
    Run store.search for each query, merge by chunk key (id or text) with
    reciprocal rank fusion, return top merged_top_k chunks.
    """
    scores: Dict[Any, float] = {}
    key_to_chunk: Dict[Any, Dict[str, Any]] = {}

    for q in queries:
        results = store.search(q, top_k=per_query_top_k)
        for rank, chunk in enumerate(results):
            key = chunk.get("id") if chunk.get("id") is not None else chunk.get("text", "")
            key_to_chunk[key] = chunk
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + 1)

    ordered_keys = sorted(scores.keys(), key=lambda k: -scores[k])[:merged_top_k]
    return [key_to_chunk[k] for k in ordered_keys]


class HybridRAGPipeline:
    """
    Orchestrates: restate question -> multi-query RAG (merge + one generation) +
    fine-tuned model (6 answers) -> synthesize final answer with small LLM.
    """

    def __init__(
        self,
        db_path: str,
        peft_model_path: str,
        base_model_name: str,
        small_llm_name: Optional[str] = None,
        per_query_top_k: Optional[int] = None,
        merged_retrieval_top_k: Optional[int] = None,
        num_restatements: Optional[int] = None,
    ):
        h_cfg = get_hybrid_rag_config()
        r_cfg = get_rag_config()

        self.db_path = db_path
        self.per_query_top_k = per_query_top_k or h_cfg["per_query_top_k"]
        self.merged_retrieval_top_k = merged_retrieval_top_k or h_cfg["merged_retrieval_top_k"]
        num_rest = num_restatements or h_cfg["num_restatements"]

        small_llm_model = small_llm_name or h_cfg["small_llm_model"]
        self._small_llm = SmallLLM(
            model_name=small_llm_model,
            max_new_tokens=max(
                h_cfg.get("max_new_tokens_restater", 256),
                h_cfg.get("max_new_tokens_synthesizer", 512),
            ),
            temperature=h_cfg.get("temperature", 0.7),
        )
        self._restater = Restater(
            self._small_llm,
            num_restatements=num_rest,
            max_new_tokens=h_cfg.get("max_new_tokens_restater", 256),
        )
        self._synthesizer = Synthesizer(
            self._small_llm,
            max_new_tokens=h_cfg.get("max_new_tokens_synthesizer", 512),
        )

        self._store = RAGStore(
            db_path=db_path,
            embedding_model_name=r_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            collection_name=r_cfg.get("collection_name", "rag_docs"),
            use_hybrid=r_cfg.get("use_hybrid", False),
        )
        # RAG generator uses same small LLM model for consistency, or use RAG default
        self._rag_generator = RAGGenerator(
            model_name=small_llm_model,
            max_new_tokens=h_cfg.get("max_new_tokens_synthesizer", 512),
            temperature=h_cfg.get("temperature", 0.7),
        )

        self._ft = QAInference(
            base_model_name=base_model_name,
            peft_model_path=peft_model_path,
        )

    def answer(self, question: str) -> str:
        """
        1. Restate question 5 ways.
        2. RAG: merge retrieval over [question] + 5 restatements, one generation.
        3. FT: 6 answers (original + 5 restatements).
        4. Synthesize final answer from RAG + FT candidates.
        """
        restatements = self._restater.restate(question)
        queries = [question] + restatements[:5]

        merged_chunks = _merge_retrievals_rrf(
            self._store,
            queries,
            self.per_query_top_k,
            self.merged_retrieval_top_k,
        )
        context = _format_context(merged_chunks)
        if context.strip():
            rag_answer = self._rag_generator.generate(context, question)
        else:
            rag_answer = "No relevant context was found for this question."

        ft_answers = [
            self._ft.generate_answer(q, use_history=False)
            for q in queries
        ]

        return self._synthesizer.synthesize(question, rag_answer, ft_answers)
