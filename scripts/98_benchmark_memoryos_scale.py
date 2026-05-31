#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wgram_lm.memoryos.scale_benchmark import build_scale_benchmark_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write planning records for staged MemoryOS scale checks. "
            "This does not build embeddings; it estimates index size and backend choices."
        )
    )
    parser.add_argument("--token-targets", default="1M,10M")
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=640)
    parser.add_argument("--embedding-dtype-bits", type=int, default=32)
    parser.add_argument("--available-ram-gib", type=float, default=64)
    parser.add_argument("--available-vram-gib", type=float, default=24)
    parser.add_argument("--jsonl-out", default="runs/eval/memoryos_scale_plan_1m_10m.jsonl")
    args = parser.parse_args()

    records = build_scale_benchmark_records(
        args.token_targets,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        embedding_dim=args.embedding_dim,
        embedding_dtype_bits=args.embedding_dtype_bits,
        available_ram_gib=args.available_ram_gib,
        available_vram_gib=args.available_vram_gib,
    )

    out = Path(args.jsonl_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    for record in records:
        plan = record["plan"]
        print(
            f"{record['target_label']:>4s}: chunks={plan['estimated_chunks']:,} "
            f"emb={plan['embedding_gib']:.3f}GiB backend={plan['build_backend']} "
            f"pattern={plan['serving_pattern']}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
