"""
Standard evaluation metrics for QA/generation: BLEU, ROUGE, embedding similarity,
exact match, token F1, and optional LLM-as-judge.

All metrics compare a model response (candidate) to a reference answer.
Lazy imports: if a dependency is missing, that metric is skipped and not included.
"""

from typing import Dict, List, Optional, Any


def _tokenize(text: str) -> list:
    """Simple whitespace tokenization for BLEU/F1."""
    return text.strip().lower().split()


def compute_bleu(reference: str, candidate: str) -> Optional[float]:
    """BLEU score (0-1, higher is better). Uses nltk; returns None if nltk not available."""
    if not reference or not candidate:
        return 0.0
    try:
        from nltk.translate.bleu_score import sentence_bleu
        ref_tokens = _tokenize(reference)
        cand_tokens = _tokenize(candidate)
        if not ref_tokens:
            return 0.0
        # sentence_bleu expects reference as list of lists (multiple refs) or list of tokens
        ref = [ref_tokens]
        return sentence_bleu(ref, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    except ImportError:
        return None
    except Exception:
        return None


def compute_rouge(reference: str, candidate: str) -> Optional[Dict[str, float]]:
    """ROUGE-1, ROUGE-2, ROUGE-L F1 (0-1). Returns dict or None if rouge_score not available."""
    if not reference or not candidate:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        scores = scorer.score(reference, candidate)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }
    except ImportError:
        return None
    except Exception:
        return None


def compute_embedding_similarity(reference: str, candidate: str) -> Optional[float]:
    """Cosine similarity (0-1) between sentence embeddings. Optional: requires sentence-transformers."""
    if not reference or not candidate:
        return 0.0
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("all-MiniLM-L6-v2")
        ref_emb = model.encode(reference, convert_to_numpy=True)
        cand_emb = model.encode(candidate, convert_to_numpy=True)
        cos = np.dot(ref_emb, cand_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(cand_emb) + 1e-9)
        return float(np.clip((cos + 1) / 2, 0, 1))  # map [-1,1] to [0,1]
    except ImportError:
        return None
    except Exception:
        return None


def compute_exact_match(reference: str, candidate: str) -> float:
    """Binary exact match after normalizing (strip, lower, collapse whitespace). 0 or 1."""
    if not reference and not candidate:
        return 1.0
    import re
    ref_norm = re.sub(r"\s+", " ", reference.strip().lower())
    cand_norm = re.sub(r"\s+", " ", candidate.strip().lower())
    return 1.0 if ref_norm == cand_norm else 0.0


def compute_token_f1(reference: str, candidate: str) -> float:
    """Token-level F1: overlap of tokens (SQuAD-style). Returns 0-1."""
    ref_tokens = set(_tokenize(reference))
    cand_tokens = _tokenize(candidate)
    if not ref_tokens and not cand_tokens:
        return 1.0
    if not ref_tokens or not cand_tokens:
        return 0.0
    common = sum(1 for t in cand_tokens if t in ref_tokens)
    precision = common / len(cand_tokens) if cand_tokens else 0.0
    recall = common / len(ref_tokens) if ref_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_llm_judge(
    question: str,
    reference: str,
    response: str,
    inference_fn,  # callable that takes (prompt) and returns generated text
) -> Optional[float]:
    """
    Optional LLM-as-judge: ask the model to rate the response 0-1.
    inference_fn(question) should return the model's reply string.
    Returns None if parsing fails or not used.
    """
    if not inference_fn:
        return None
    try:
        prompt = (
            f"Question: {question}\n"
            f"Reference answer: {reference}\n"
            f"Model response: {response}\n"
            "Rate how well the model response matches the reference (0.0 = bad, 1.0 = perfect). Reply with only a number between 0 and 1."
        )
        out = inference_fn(prompt)
        if out is None:
            return None
        import re
        numbers = re.findall(r"0?\.\d+|1\.0|1")
        if numbers:
            score = float(numbers[0])
            return max(0.0, min(1.0, score))
        return None
    except Exception:
        return None


def compute_all_metrics(
    reference: str,
    candidate: str,
    question: Optional[str] = None,
    inference_fn: Optional[Any] = None,
    use_embedding: bool = True,
    use_llm_judge: bool = False,
) -> Dict[str, float]:
    """
    Compute all available metrics. Returns a dict with keys like bleu, rouge1, rouge2, rougeL,
    embedding_sim, exact_match, token_f1, and optionally llm_judge.
    Missing metrics (e.g. dependency not installed) are omitted.
    """
    out: Dict[str, float] = {}

    bleu = compute_bleu(reference, candidate)
    if bleu is not None:
        out["bleu"] = round(bleu, 4)

    rouge = compute_rouge(reference, candidate)
    if rouge is not None:
        out["rouge1"] = round(rouge["rouge1"], 4)
        out["rouge2"] = round(rouge["rouge2"], 4)
        out["rougeL"] = round(rouge["rougeL"], 4)

    if use_embedding:
        emb = compute_embedding_similarity(reference, candidate)
        if emb is not None:
            out["embedding_sim"] = round(emb, 4)

    out["exact_match"] = round(compute_exact_match(reference, candidate), 4)
    out["token_f1"] = round(compute_token_f1(reference, candidate), 4)

    if use_llm_judge and question and inference_fn:
        judge = compute_llm_judge(question, reference, candidate, inference_fn)
        if judge is not None:
            out["llm_judge"] = round(judge, 4)

    return out


def aggregate_metrics(results_with_metrics: list) -> Dict[str, float]:
    """
    Given a list of dicts each with an 'extra_metrics' key (dict of metric name -> value),
    return aggregate (mean) for each metric.
    """
    if not results_with_metrics:
        return {}
    agg: Dict[str, List[float]] = {}
    for r in results_with_metrics:
        em = getattr(r, "extra_metrics", None) or (r.get("extra_metrics") if isinstance(r, dict) else None)
        if not em:
            continue
        for k, v in em.items():
            if isinstance(v, (int, float)):
                agg.setdefault(k, []).append(float(v))
    return {k: round(sum(v) / len(v), 4) for k, v in agg.items() if v}
