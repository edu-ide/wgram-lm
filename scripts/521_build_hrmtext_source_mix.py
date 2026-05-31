#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from wgram_lm.data.hrm_text_source_mix import (
    DATASET_VIEWER_BASE,
    build_hrm_text_source_mix,
    resolve_sources,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an HRM-Text/data_io-style condition/instruction/response "
            "source mix for QTRM healing and source-aligned warmup."
        )
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--verified-source", action="append", default=[])
    parser.add_argument("--max-verified-rows-per-source", type=int, default=200)
    parser.add_argument("--verified-offset", type=int, default=0)
    parser.add_argument("--dolly-rows", type=int, default=500)
    parser.add_argument("--dolly-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viewer-base", default=DATASET_VIEWER_BASE)
    args = parser.parse_args()

    stats = build_hrm_text_source_mix(
        out_dir=args.out_dir,
        verified_sources=resolve_sources(args.verified_source),
        max_verified_rows_per_source=args.max_verified_rows_per_source,
        dolly_rows=args.dolly_rows,
        seed=args.seed,
        verified_offset=args.verified_offset,
        dolly_offset=args.dolly_offset,
        viewer_base=args.viewer_base,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
