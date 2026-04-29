#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qtrm_mm.eval.memory_retrieval import case_index_records, load_cases
from qtrm_mm.memoryos.text_index import DEFAULT_TEXT_EMBED_MODEL, load_embedder
from qtrm_mm.memoryos.vector_backends import DEFAULT_VECTOR_BACKEND, build_vector_index


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Build a MemoryOS vector index from memory retrieval probe JSONL cases."
    )
    ap.add_argument("--cases", default="data/eval/memory_retrieval_distractor_probe.jsonl")
    ap.add_argument("--out-dir", default="runs/eval/memory_retrieval_memoryos_index")
    ap.add_argument(
        "--model-id",
        default=DEFAULT_TEXT_EMBED_MODEL,
        help="SentenceTransformer embedding model for the probe index.",
    )
    ap.add_argument("--target-only", action="store_true", help="Index only target evidence, omitting distractors.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--backend", default=DEFAULT_VECTOR_BACKEND, choices=["faiss_flat", "faiss_hnsw", "numpy_flat"])
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--hnsw-ef-construction", type=int, default=200)
    return ap


def build_probe_index(
    *,
    cases_path: str,
    out_dir: str,
    model_id: str,
    include_distractors: bool,
    batch_size: int,
    backend: str,
    hnsw_m: int,
    hnsw_ef_construction: int,
) -> int:
    cases = load_cases(cases_path)
    records = case_index_records(cases, include_distractors=include_distractors)
    if not records:
        raise RuntimeError(f"No probe records found in {cases_path}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = load_embedder(model_id)
    docs = [str(record.get("text", "")) for record in records]
    emb = model.encode(
        docs,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=batch_size,
    )
    emb = np.asarray(emb, dtype="float32")

    actual_backend = build_vector_index(
        emb,
        out,
        backend=backend,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
    )

    (out / "records.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )
    (out / "meta.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "num_records": len(records),
                "cases": cases_path,
                "include_distractors": include_distractors,
                "backend": actual_backend,
                "requested_backend": backend,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return len(records)


def main() -> None:
    args = build_arg_parser().parse_args()
    count = build_probe_index(
        cases_path=args.cases,
        out_dir=args.out_dir,
        model_id=args.model_id,
        include_distractors=not args.target_only,
        batch_size=args.batch_size,
        backend=args.backend,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
    )
    print(f"built MemoryOS probe index: {args.out_dir}, records={count}, backend={args.backend}")


if __name__ == "__main__":
    main()
