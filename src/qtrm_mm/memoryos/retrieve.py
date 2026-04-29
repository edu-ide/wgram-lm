from __future__ import annotations
import argparse
import json
from functools import lru_cache
from pathlib import Path
import numpy as np

from .vector_backends import search_vector_index
from .rerank import DEFAULT_RERANKER_MODEL, rerank_results

DEFAULT_TEXT_EMBED_MODEL = "microsoft/harrier-oss-v1-270m"


@lru_cache(maxsize=4)
def load_query_embedder(model_id: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, model_kwargs={"dtype": "auto"})


def encode_query(model, query: str, *, model_id: str):
    if "harrier-oss" in model_id:
        try:
            return model.encode(
                [query],
                prompt_name="web_search_query",
                normalize_embeddings=True,
            )
        except TypeError:
            pass
    return model.encode(
        [f"Instruct: Retrieve relevant passages that answer the query\nQuery: {query}"],
        normalize_embeddings=True,
    )


def retrieve(
    index_dir: str,
    query: str,
    top_k: int = 5,
    model_id: str | None = None,
    *,
    backend: str | None = None,
    hnsw_ef_search: int | None = None,
    rerank_backend: str = "none",
    reranker_model_id: str = DEFAULT_RERANKER_MODEL,
    rerank_top_k: int | None = None,
    reranker_device: str | None = None,
):
    index_dir = Path(index_dir)
    records = [json.loads(line) for line in (index_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    meta = json.loads((index_dir / "meta.json").read_text()) if (index_dir / "meta.json").exists() else {}
    model_id = model_id or meta.get("model_id", DEFAULT_TEXT_EMBED_MODEL)
    model = load_query_embedder(model_id)
    q = encode_query(model, query, model_id=model_id)
    q = np.asarray(q, dtype="float32")
    scores, ids = search_vector_index(
        index_dir,
        q,
        top_k=top_k,
        backend=backend,
        hnsw_ef_search=hnsw_ef_search,
    )
    results = [
        (float(scores[0][i]), records[int(ids[0][i])])
        for i in range(len(ids[0]))
        if int(ids[0][i]) >= 0
    ]
    if rerank_backend != "none":
        return rerank_results(
            query,
            results,
            top_k=rerank_top_k or top_k,
            backend=rerank_backend,
            model_id=reranker_model_id,
            device=reranker_device,
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index_dir")
    ap.add_argument("query")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--hnsw-ef-search", type=int, default=None)
    ap.add_argument("--rerank-backend", default="none", choices=["none", "lexical", "cross_encoder"])
    ap.add_argument("--reranker-model-id", default=DEFAULT_RERANKER_MODEL)
    ap.add_argument("--rerank-top-k", type=int, default=None)
    ap.add_argument("--reranker-device", default=None)
    args = ap.parse_args()
    for score, rec in retrieve(
        args.index_dir,
        args.query,
        args.top_k,
        backend=args.backend,
        hnsw_ef_search=args.hnsw_ef_search,
        rerank_backend=args.rerank_backend,
        reranker_model_id=args.reranker_model_id,
        rerank_top_k=args.rerank_top_k,
        reranker_device=args.reranker_device,
    ):
        retrieval_score = rec.get("retrieval_score")
        score_text = f"SCORE={score:.4f}"
        if retrieval_score is not None:
            score_text += f" RETRIEVAL={float(retrieval_score):.4f}"
        print(f"{score_text} SOURCE={rec['source']} CHUNK={rec['chunk_id']}")
        print(rec["text"][:500].replace("\n", " "))
        print("---")


if __name__ == "__main__":
    main()
