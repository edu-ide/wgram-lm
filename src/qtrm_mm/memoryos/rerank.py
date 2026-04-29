from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable


DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_RERANK_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
SUPPORTED_RERANK_BACKENDS = {"none", "lexical", "cross_encoder"}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "is",
    "of",
    "the",
    "to",
    "what",
    "which",
    "with",
}


def _terms(text: str) -> set[str]:
    normalized = "".join(ch if ch.isalnum() else " " for ch in text.casefold())
    return {token for token in normalized.split() if len(token) > 1 and token not in _STOPWORDS}


def lexical_rerank_score(query: str, document: str) -> float:
    query_terms = _terms(query)
    doc_terms = _terms(document)
    if not query_terms:
        return 0.0
    overlap = query_terms & doc_terms
    phrase_bonus = sum(1 for term in query_terms if term in document.casefold())
    return float(len(overlap) + 0.1 * phrase_bonus)


@lru_cache(maxsize=2)
def _cached_cross_encoder(model_id: str, device: str | None, max_length: int | None):
    from sentence_transformers import CrossEncoder

    kwargs: dict[str, Any] = {}
    if device:
        kwargs["device"] = device
    if max_length:
        kwargs["max_length"] = max_length
    return CrossEncoder(model_id, **kwargs)


def load_cross_encoder(model_id: str = DEFAULT_RERANKER_MODEL, *, device: str | None = None, max_length: int | None = None):
    return _cached_cross_encoder(model_id, device, max_length)


def _annotate(score: float, rec: dict[str, Any], *, backend: str, retrieval_score: float) -> tuple[float, dict[str, Any]]:
    enriched = dict(rec)
    enriched["retrieval_score"] = float(retrieval_score)
    enriched["rerank_score"] = float(score)
    enriched["rerank_backend"] = backend
    return float(score), enriched


def rerank_results(
    query: str,
    results: Iterable[tuple[float, dict[str, Any]]],
    *,
    top_k: int,
    backend: str = "none",
    model: Any | None = None,
    model_id: str = DEFAULT_RERANKER_MODEL,
    device: str | None = None,
    max_length: int | None = None,
) -> list[tuple[float, dict[str, Any]]]:
    candidates = list(results)
    if backend not in SUPPORTED_RERANK_BACKENDS:
        raise ValueError(f"unknown rerank backend: {backend}")
    if top_k <= 0:
        return []

    if backend == "none":
        return [
            _annotate(float(score), rec, backend=backend, retrieval_score=float(score))
            for score, rec in candidates[:top_k]
        ]

    if backend == "lexical":
        scored = [
            _annotate(
                lexical_rerank_score(query, f"{rec.get('source', '')} {rec.get('text', '')}"),
                rec,
                backend=backend,
                retrieval_score=float(score),
            )
            for score, rec in candidates
        ]
        scored.sort(key=lambda item: (item[0], item[1]["retrieval_score"]), reverse=True)
        return scored[:top_k]

    if model is None:
        model = load_cross_encoder(model_id, device=device, max_length=max_length)

    documents = [str(rec.get("text", "")) for _, rec in candidates]
    if hasattr(model, "rank"):
        rankings = model.rank(query, documents, top_k=top_k)
        reranked: list[tuple[float, dict[str, Any]]] = []
        for item in rankings:
            idx = int(item.get("corpus_id"))
            score = float(item.get("score"))
            retrieval_score, rec = candidates[idx]
            reranked.append(
                _annotate(score, rec, backend=backend, retrieval_score=float(retrieval_score))
            )
        return reranked

    scores = model.predict([(query, doc) for doc in documents])
    scored = [
        _annotate(float(score), rec, backend=backend, retrieval_score=float(retrieval_score))
        for score, (retrieval_score, rec) in zip(scores, candidates)
    ]
    scored.sort(key=lambda item: (item[0], item[1]["retrieval_score"]), reverse=True)
    return scored[:top_k]
