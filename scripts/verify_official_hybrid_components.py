#!/usr/bin/env python3
"""
Strict Verification Script: Official GDN2 + Official MLA in OneBodyParallelHybridBlock

This script performs rigorous checks on whether the hybrid block is actually using
official implementations when requested.

Run with:
  /home/tripleyoung/qtrm-workspace/wgram-lm/.venv/bin/python \
      scripts/verify_official_hybrid_components.py

Part of the strict sequential verification process (2026-05-30).
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from wgram_lm.config import QTRMConfig
from wgram_lm.blocks import OneBodyParallelHybridBlock


def check_environment():
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    # Check for flash-attn
    try:
        import flash_attn
        print(f"flash-attn: {flash_attn.__version__}")
    except ImportError:
        print("flash-attn: NOT INSTALLED (official MLA will likely fail)")

    # Check for FLA official path
    fla_gdn2 = ROOT / "references" / "official" / "flash-linear-attention-gdn2"
    print(f"FLA-GDN2 vendored path exists: {fla_gdn2.exists()}")

    gdn2_path = ROOT / "references" / "official" / "gated-deltanet-2"
    print(f"GatedDeltaNet-2 vendored path exists: {gdn2_path.exists()}")
    print()


def test_combination(name: str, delta_backend: str, attention_type: str):
    print("-" * 60)
    print(f"TEST: {name}")
    print(f"  delta_backend   = {delta_backend}")
    print(f"  attention_type  = {attention_type}")
    print("-" * 60)

    cfg = QTRMConfig(
        d_model=128,
        n_heads=8,
        n_kv_heads=4,
        d_ff=256,
        max_seq_len=512,
        delta_backend=delta_backend,
    )

    try:
        block = OneBodyParallelHybridBlock(
            cfg=cfg,
            recurrence_head_count=2,
            attention_head_count=2,
            attention_type=attention_type,
        )

        # Inspect what was actually instantiated
        rec_type = type(block.recurrence_heads[0]).__name__
        att_type = type(block.attention_heads[0]).__name__

        print(f"  Recurrence head class : {rec_type}")
        print(f"  Attention head class  : {att_type}")

        # Check if they are the "official" ones
        is_official_rec = "Official" in rec_type or "GatedDeltaNet2" in rec_type and "Torch" not in rec_type
        is_official_att = "MultiheadLatentAttention" in att_type and "Simplified" not in att_type and "Custom" not in att_type

        print(f"  Using OFFICIAL GDN2?  : {is_official_rec}")
        print(f"  Using OFFICIAL MLA?   : {is_official_att}")

        # Run a forward pass
        x = torch.randn(2, 8, 128)
        with torch.no_grad():
            out = block(x)

        print(f"  Forward pass          : SUCCESS (shape={tuple(out.shape)})")

    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

    print()


def main():
    check_environment()

    # Test 1: Default (should use our custom V2 + try official MLA)
    test_combination(
        "Default (custom recurrence + mla attempt)",
        delta_backend="torch_gated_delta2_v2",
        attention_type="mla",
    )

    # Test 2: Force official GDN2 + official MLA
    test_combination(
        "Force OFFICIAL everywhere",
        delta_backend="official_gated_delta2",
        attention_type="mla",
    )

    # Test 3: Official GDN2 + GQA (for comparison)
    test_combination(
        "Official GDN2 + GQA",
        delta_backend="official_gated_delta2",
        attention_type="gqa",
    )

    print("=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
