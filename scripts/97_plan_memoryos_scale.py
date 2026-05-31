#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from wgram_lm.memoryos.scale_plan import build_memory_scale_plan, format_plan_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate MemoryOS retrieval/index requirements before building a large memory pool."
    )
    parser.add_argument("--total-tokens", type=int, default=100_000_000)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=640)
    parser.add_argument("--embedding-dtype-bits", type=int, default=32)
    parser.add_argument("--available-ram-gib", type=float, default=64)
    parser.add_argument("--available-vram-gib", type=float, default=24)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    plan = build_memory_scale_plan(
        total_tokens=args.total_tokens,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        embedding_dim=args.embedding_dim,
        embedding_dtype_bits=args.embedding_dtype_bits,
        available_ram_gib=args.available_ram_gib,
        available_vram_gib=args.available_vram_gib,
    )
    if args.json:
        print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_plan_summary(plan))


if __name__ == "__main__":
    main()
