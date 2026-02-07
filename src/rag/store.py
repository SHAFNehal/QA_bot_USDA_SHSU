"""
RAG store: Chroma-backed vector store with optional BM25 hybrid retrieval.
Open-source only: Chroma (Apache 2.0), rank_bm25 (Apache 2.0), sentence-transformers.
"""

import os
import pickle
from typing import List, Dict, Any, Optional

# Chroma
import chromadb
from chromadb.config import Settings as ChromaSettings

# Embeddings
from sentence_transformers import SentenceTransformer

# BM25 (optional)
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


def _tokenize_for_bm25(text: str) -> List[str]:
    """Simple tokenization for BM25 (lowercase, split on non-alnum)."""
    import re
    return re.findall(r"\w+", text.lower())


class RAGStore:
    """
    Persistent vector store (Chroma) with optional sparse BM25 index.
    add_documents(chunks) -> ingest; search(query, top_k, use_hybrid) -> list of chunks.
    """

    BM25_PICKLE = "bm25_index.pkl"

    def __init__(
        self,
        db_path: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "rag_docs",
        use_hybrid: bool = False,
    ):
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.use_hybrid = use_hybrid and HAS_BM25

        os.makedirs(db_path, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG document chunks"},
        )
        self._embedding_model: Optional[SentenceTransformer] = None
        self._bm25_corpus: List[str] = []
        self._bm25_doc_ids: List[str] = []
        self._bm25: Optional[Any] = None
        self._load_bm25_if_present()

    def _get_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def _load_bm25_if_present(self) -> None:
        pkl_path = os.path.join(self.db_path, self.BM25_PICKLE)
        if not self.use_hybrid or not os.path.isfile(pkl_path):
            return
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            self._bm25_corpus = data.get("corpus", [])
            self._bm25_doc_ids = data.get("doc_ids", [])
            tokenized = [ _tokenize_for_bm25(t) for t in self._bm25_corpus ]
            self._bm25 = BM25Okapi(tokenized) if HAS_BM25 and tokenized else None
        except Exception:
            self._bm25_corpus = []
            self._bm25_doc_ids = []
            self._bm25 = None

    def _save_bm25(self) -> None:
        if not self._bm25_corpus or not self._bm25_doc_ids:
            return
        pkl_path = os.path.join(self.db_path, self.BM25_PICKLE)
        with open(pkl_path, "wb") as f:
            pickle.dump({"corpus": self._bm25_corpus, "doc_ids": self._bm25_doc_ids}, f)

    def add_documents(self, chunks: List[Dict[str, Any]], replace: bool = False) -> None:
        """
        Ingest chunks into Chroma (and optionally build BM25 index).
        Each chunk: {"text": str, "metadata": dict}. metadata must be flat (str, int, float, bool).
        """
        if not chunks:
            return
        if replace:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document chunks"},
            )
            self._bm25_corpus = []
            self._bm25_doc_ids = []
            self._bm25 = None

        texts = [ c["text"] for c in chunks ]
        metadatas = [ c.get("metadata", {}) for c in chunks ]
        ids = [ c.get("metadata", {}).get("id", f"chunk_{i}") for i, c in enumerate(chunks) ]
        for i, id_ in enumerate(ids):
            if isinstance(id_, (int, float)):
                ids[i] = str(id_)
        if not all(isinstance(i, str) for i in ids):
            ids = [ f"chunk_{i}" for i in range(len(chunks)) ]

        model = self._get_embedding_model()
        embeddings = model.encode(texts, show_progress_bar=len(texts) > 50).tolist()

        self._collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        if self.use_hybrid and HAS_BM25:
            self._bm25_corpus.extend(texts)
            self._bm25_doc_ids.extend(ids)
            tokenized = [ _tokenize_for_bm25(t) for t in self._bm25_corpus ]
            self._bm25 = BM25Okapi(tokenized)
            self._save_bm25()

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return top_k chunks. use_hybrid defaults to self.use_hybrid.
        Each result: {"text": str, "metadata": dict}.
        """
        if use_hybrid is None:
            use_hybrid = self.use_hybrid

        if use_hybrid and HAS_BM25 and self._bm25 is not None and self._bm25_corpus:
            return self._search_hybrid(query, top_k)
        return self._search_dense(query, top_k)

    def _search_dense(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        model = self._get_embedding_model()
        q_emb = model.encode([query]).tolist()
        count = self._collection.count()
        if count == 0:
            return []
        res = self._collection.query(
            query_embeddings=q_emb,
            n_results=min(top_k, count),
            include=["documents", "metadatas", "ids"],
        )
        if not res["documents"] or not res["documents"][0]:
            return []
        out = []
        for doc, meta, id_ in zip(
            res["documents"][0],
            res["metadatas"][0] or [],
            res["ids"][0] or [],
        ):
            out.append({"text": doc, "metadata": meta or {}, "id": id_})
        return out

    def _search_hybrid(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        # Build id -> doc from collection (needed for BM25-only ids)
        all_data = self._collection.get(include=["documents", "metadatas"])
        id_to_doc = {}
        for i, id_ in enumerate(all_data["ids"]):
            id_to_doc[id_] = {
                "text": all_data["documents"][i],
                "metadata": (all_data["metadatas"] or [{}])[i] or {},
            }

        k_expand = min(top_k * 2, len(self._bm25_corpus))
        dense_results = self._search_dense(query, k_expand)
        dense_scores: Dict[str, float] = {}
        for r, item in enumerate(dense_results):
            dense_scores[item["id"]] = 1.0 / (r + 1)

        tokenized_q = _tokenize_for_bm25(query)
        bm25_scores = self._bm25.get_scores(tokenized_q)
        order = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])
        bm25_scores_by_id: Dict[str, float] = {}
        for rank, i in enumerate(order[:k_expand]):
            doc_id = self._bm25_doc_ids[i]
            bm25_scores_by_id[doc_id] = 1.0 / (rank + 1)

        all_ids = set(dense_scores) | set(bm25_scores_by_id)
        merged = [(dense_scores.get(i, 0) + bm25_scores_by_id.get(i, 0), i) for i in all_ids]
        merged.sort(key=lambda x: -x[0])

        out = []
        for _, doc_id in merged[:top_k]:
            if doc_id in id_to_doc:
                out.append(id_to_doc[doc_id])
        return out
