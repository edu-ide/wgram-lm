#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from wgram_lm.msa_qwen35 import (
    build_qwen35_full_msa_fork,
    load_json,
    write_fork_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a Qwen3.5-2B full-MSA fork config and conversion manifest."
    )
    parser.add_argument(
        "--source-config",
        default="references/model_configs/qwen35_2b_base/config.json",
        help="Path to the Qwen3.5 source config.json.",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/qwen35_2b_full_msa_fork_plan",
        help="Directory for generated config/manifest/README.",
    )
    parser.add_argument("--top-k-docs", type=int, default=8)
    parser.add_argument("--pooling-kernel-size", type=int, default=64)
    parser.add_argument(
        "--router-layer-idx",
        default="all",
        help='MSA router layer index string, e.g. "all" or "8,12,16,20".',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_config)
    source_config = load_json(source_path)
    plan = build_qwen35_full_msa_fork(
        source_config,
        msa_config={
            "top_k_docs": args.top_k_docs,
            "pooling_kernel_size": args.pooling_kernel_size,
            "router_layer_idx": args.router_layer_idx,
        },
    )
    write_fork_artifacts(args.out_dir, plan, source_config_path=source_path)
    manifest = plan.manifest
    print(f"wrote {args.out_dir}")
    print(f"layers: {manifest['num_hidden_layers']}")
    print(
        "linear->MSA replaced: "
        f"{len(manifest['linear_attention_layers_replaced_by_msa'])}"
    )
    print(
        "full-attention MSA seeds: "
        f"{len(manifest['full_attention_layers_reused_as_msa_seed'])}"
    )


if __name__ == "__main__":
    main()
