#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from wgram_lm.qwen35_full_msa_healing import run_tiny_healing_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run safe Qwen3.5 full-MSA donor-healing smoke training."
    )
    parser.add_argument(
        "--mode",
        default="tiny-smoke",
        choices=["tiny-smoke"],
        help="Only tiny-smoke is implemented; real 2B healing comes after this gate.",
    )
    parser.add_argument("--out-dir", default="runs/qwen35_full_msa_healing_tiny_smoke")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lm-weight", type=float, default=1.0)
    parser.add_argument("--donor-kl-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_tiny_healing_smoke(
        out_dir=args.out_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        lm_weight=args.lm_weight,
        donor_kl_weight=args.donor_kl_weight,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
