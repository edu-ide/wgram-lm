#!/usr/bin/env python3
"""
Phase 2 Skeleton: Full 5-Mechanism Composition Ablation (I→G→A)

Mechanisms:
1. Gated Thought Workspaces + Broadcast (with ALRMC-aligned selector)
2. Depth-wise Monotonic Answer Attractor (정답 정렬)
3. Equation Binding + Readback
4. LeWM Predictive Tier
5. Provenance Data World Model + Gated Register

This script is the skeleton for testing all combinations on/off.
Run on GPU and expand with real data later.

Current version: proxy synthetic runs + all 2^5 = 32 combinations (or smart subset).

Usage:
  .venv/bin/python scripts/diag_phase2_full_composition.py --seeds 2 --steps 20
"""

import argparse
import itertools
import torch
from src.qtrm_mm.config import QTRMConfig
from src.qtrm_mm.core import QTRMRecursiveCore

MECHANISMS = [
    "workspace",
    "attractor",
    "eq_binding",
    "lewm",
    "provenance",
]

def run_combination(cfg_base, combo: dict, seed: int, steps: int, device: str):
    torch.manual_seed(seed)
    cfg = QTRMConfig(**{k: v for k, v in cfg_base.items()})

    # Apply ablation flags according to combo
    cfg.core_thought_workspace_ablation_zero = not combo["workspace"]
    cfg.core_answer_attractor_ablation_zero = not combo["attractor"]
    cfg.core_equation_binding_ablation_zero = not combo["eq_binding"]
    cfg.core_lewm_ablation_zero = not combo["lewm"]
    cfg.core_provenance_register_ablation_zero = not combo["provenance"]

    core = QTRMRecursiveCore(cfg).to(device)
    ws = torch.randn(4, 12, cfg.d_model, device=device)

    # Warmup
    for _ in range(5):
        _, _, _, _ = core(ws, return_carry=True)

    _, z_h, _, _ = core(ws, return_carry=True)
    return z_h.norm().item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Phase 2 Full Composition Skeleton on {device}")

    base_cfg = dict(
        d_model=128, d_ff=512, n_heads=4, n_kv_heads=2,
        n_prelude_layers=1, n_core_layers=2, max_seq_len=16, vocab_size=8192,
        core_thought_workspace_enabled=True,
        core_answer_attractor_enabled=True,
        core_equation_binding_enabled=True,
        core_lewm_enabled=True,
        core_provenance_register_enabled=True,
    )

    # Full meaningful set for Phase 2
    combos = []
    # All on (baseline)
    combos.append({m: True for m in MECHANISMS})
    # All single ablations
    for m in MECHANISMS:
        c = {mm: True for mm in MECHANISMS}
        c[m] = False
        combos.append(c)
    # Key double ablations (especially around 정답 정렬)
    combos.append({"attractor": False, "eq_binding": False, "workspace": True, "lewm": True, "provenance": True})
    combos.append({"workspace": False, "attractor": False, "eq_binding": True, "lewm": True, "provenance": True})
    # Almost all off (only one on) for extreme contrast
    for m in MECHANISMS:
        c = {mm: False for mm in MECHANISMS}
        c[m] = True
        combos.append(c)

    results = []
    for seed in range(args.seeds):
        for i, combo in enumerate(combos):
            val = run_combination(base_cfg, combo, seed=100 + seed, steps=args.steps, device=device)
            label = "+".join([m for m in MECHANISMS if combo[m]]) or "all_off"
            results.append({"seed": 100 + seed, "combo": label, "z_h_norm": val})
            print(f"Seed {100+seed} | {label:40} | z_h_norm = {val:.2f}")

    print("\n=== Phase 2 Skeleton Results (expand this) ===")
    for r in results:
        print(r)

    print("\nNext: Turn this into full 32-combination + real data + answer margin metric.")

if __name__ == "__main__":
    main()