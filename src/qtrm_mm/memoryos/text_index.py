from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from .chunk import iter_text_files, chunk_text
from .vector_backends import DEFAULT_VECTOR_BACKEND, build_vector_index

DEFAULT_TEXT_EMBED_MODEL = "microsoft/harrier-oss-v1-270m"


def load_embedder(model_id: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id, model_kwargs={"dtype": "auto"})


def build_index(
    input_dir: str,
    out_dir: str,
    model_id: str = DEFAULT_TEXT_EMBED_MODEL,
    *,
    backend: str = DEFAULT_VECTOR_BACKEND,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for p in iter_text_files(input_dir):
        text = p.read_text(errors="ignore")
        for i, ch in enumerate(chunk_text(text)):
            records.append({"source": str(p), "chunk_id": i, "text": ch})
    if not records:
        raise RuntimeError(f"No text files found in {input_dir}")
    model = load_embedder(model_id)
    docs = [r["text"] for r in records]
    emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=True, batch_size=8)
    emb = np.asarray(emb, dtype="float32")
    actual_backend = build_vector_index(
        emb,
        out,
        backend=backend,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
    )
    (out / "records.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    (out / "meta.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "num_records": len(records),
                "backend": actual_backend,
                "requested_backend": backend,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"built text index: {out}, records={len(records)}, backend={actual_backend}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--model-id", default=DEFAULT_TEXT_EMBED_MODEL)
    ap.add_argument("--backend", default=DEFAULT_VECTOR_BACKEND, choices=["faiss_flat", "faiss_hnsw", "numpy_flat"])
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--hnsw-ef-construction", type=int, default=200)
    args = ap.parse_args()
    build_index(
        args.input_dir,
        args.out_dir,
        args.model_id,
        backend=args.backend,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
    )


if __name__ == "__main__":
    main()
