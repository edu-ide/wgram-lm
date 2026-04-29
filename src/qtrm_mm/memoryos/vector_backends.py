from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DEFAULT_VECTOR_BACKEND = "faiss_flat"
SUPPORTED_VECTOR_BACKENDS = {"numpy_flat", "faiss_flat", "faiss_hnsw"}


def _as_float32_matrix(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype="float32")
    if arr.ndim != 2:
        raise ValueError("vectors must be a 2D float32-compatible array")
    return arr


def _write_vector_meta(out_dir: Path, *, backend: str, dim: int, count: int, **extra) -> None:
    meta = {
        "backend": backend,
        "dim": dim,
        "count": count,
        **extra,
    }
    (out_dir / "vector_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_vector_meta(index_dir: str | Path) -> dict:
    path = Path(index_dir) / "vector_meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_vector_index(
    vectors: np.ndarray,
    out_dir: str | Path,
    *,
    backend: str = DEFAULT_VECTOR_BACKEND,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
) -> str:
    if backend not in SUPPORTED_VECTOR_BACKENDS:
        raise ValueError(f"unknown vector backend: {backend}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = _as_float32_matrix(vectors)

    if backend == "numpy_flat":
        np.save(out / "embeddings.npy", emb)
        _write_vector_meta(out, backend=backend, dim=emb.shape[1], count=emb.shape[0])
        return backend

    try:
        import faiss
    except Exception:
        if backend == "faiss_hnsw":
            raise
        np.save(out / "embeddings.npy", emb)
        _write_vector_meta(
            out,
            backend="numpy_flat",
            requested_backend=backend,
            dim=emb.shape[1],
            count=emb.shape[0],
        )
        return "numpy_flat"

    if backend == "faiss_flat":
        index = faiss.IndexFlatIP(emb.shape[1])
    elif backend == "faiss_hnsw":
        index = faiss.IndexHNSWFlat(emb.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = hnsw_ef_construction
    else:
        raise ValueError(f"unknown vector backend: {backend}")

    index.add(emb)
    faiss.write_index(index, str(out / "index.faiss"))
    _write_vector_meta(
        out,
        backend=backend,
        dim=emb.shape[1],
        count=emb.shape[0],
        hnsw_m=hnsw_m if backend == "faiss_hnsw" else None,
        hnsw_ef_construction=hnsw_ef_construction if backend == "faiss_hnsw" else None,
    )
    return backend


def search_vector_index(
    index_dir: str | Path,
    query_vectors: np.ndarray,
    *,
    top_k: int,
    backend: str | None = None,
    hnsw_ef_search: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    index_dir = Path(index_dir)
    q = _as_float32_matrix(query_vectors)
    vector_meta = read_vector_meta(index_dir)
    backend = backend or vector_meta.get("backend") or (
        "faiss_flat" if (index_dir / "index.faiss").exists() else "numpy_flat"
    )

    if backend == "numpy_flat":
        emb = np.load(index_dir / "embeddings.npy")
        scores = np.asarray(emb @ q.T, dtype="float32").T
        ids = np.argsort(-scores, axis=1)[:, :top_k]
        sorted_scores = np.take_along_axis(scores, ids, axis=1)
        return sorted_scores, ids.astype("int64")

    if backend in {"faiss_flat", "faiss_hnsw"}:
        import faiss

        index = faiss.read_index(str(index_dir / "index.faiss"))
        if backend == "faiss_hnsw" and hnsw_ef_search is not None and hasattr(index, "hnsw"):
            index.hnsw.efSearch = hnsw_ef_search
        scores, ids = index.search(q, top_k)
        return np.asarray(scores), np.asarray(ids)

    raise ValueError(f"unknown vector backend: {backend}")
