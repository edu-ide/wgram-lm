#!/usr/bin/env python3
"""
Reverse I→G→A - I-stage Narrow Gate Diagnostic for Stochastic Recurrent Breadth
(per research-driven-architecture-debugging skill, 2026-05-30 plan)

Tests:
- stochastic_breadth_enabled=True vs core_stochastic_breadth_ablation_zero=True
- Verifies ablation_zero produces numerically identical behavior to disabled
- Measures basic trajectory diversity effect when enabled

Run example (from workspace root):
PYTHONPATH=. python scripts/diag_reverse_iga_stochastic_breadth_narrow_gate.py --steps 20
"""

import argparse
import sys
from pathlib import Path

# Make runnable from workspace root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.qtrm_mm.config import QTRMConfig
from src.qtrm_mm.core import QTRMRecursiveCore

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    args = parser.parse_args()

    cfg = QTRMConfig(
        d_model=args.d_model,
        n_core_layers=2,
        outer_steps=args.steps,
        h_cycles=1,
        l_cycles=1,
        core_stochastic_breadth_enabled=True,
        core_stochastic_mode="delta",
        core_stochastic_scale=0.08,
        core_stochastic_breadth_ablation_zero=False,
    )

    core = QTRMRecursiveCore(cfg).eval()

    # Synthetic workspace
    workspace = torch.randn(args.batch, 16, args.d_model)

    # Run with breadth ON
    with torch.no_grad():
        z_l_on, z_h_on, _, _ = core(workspace)

    # Now force ablation_zero
    cfg2 = QTRMConfig(
        d_model=args.d_model,
        n_core_layers=2,
        outer_steps=args.steps,
        h_cycles=1,
        l_cycles=1,
        core_stochastic_breadth_enabled=True,
        core_stochastic_mode="delta",
        core_stochastic_scale=0.08,
        core_stochastic_breadth_ablation_zero=True,   # <-- force identity
    )
    core2 = QTRMRecursiveCore(cfg2).eval()

    with torch.no_grad():
        z_l_ab, z_h_ab, _, _ = core2(workspace)

    # Check numerical identity under ablation
    diff = (z_h_on - z_h_ab).abs().max().item()
    print(f"[Ablation Zero Check] max |z_h_on - z_h_ab| = {diff:.2e}")
    if diff < 1e-6:
        print("PASS: ablation_zero produces identity behavior (within float error)")
    else:
        print("FAIL: ablation_zero is leaking stochasticity!")

    # Rough diversity signal when enabled (not strict gate, just diagnostic)
    if args.batch >= 2:
        diversity = torch.cosine_similarity(z_h_on[0].mean(0), z_h_on[1].mean(0), dim=-1).abs().item()
        print(f"[Enabled Diversity Proxy] cosine between two samples' mean z_h: {diversity:.4f} (lower = more divergence from stochasticity)")

    print("\nNarrow gate diagnostic complete. See 2026-05-30-reverse-iga-stochastic-breadth-plan.md")

if __name__ == "__main__":
    main()