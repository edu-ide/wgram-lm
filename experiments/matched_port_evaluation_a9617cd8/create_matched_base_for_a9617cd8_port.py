#!/usr/bin/env python3
"""
Minimal Base Checkpoint Creator for Matched Condition Experiments

Purpose:
Create a very short, controlled "base" checkpoint using the current
ported QTRMRecursiveCore (a9617cd8 FULL PORT: stochastic breadth + 
gated equation_binding readback).

This base can later be used for true conditions-matched continuation
experiments (on vs off the new mechanisms) under FAIR_COMPARISON_PROTOCOL.md.

This is explicitly a "protocol-compliant minimal base", not a claim
of reasoning intelligence.

Usage:
    PYTHONPATH=src python scripts/create_matched_base_for_a9617cd8_port.py --steps 15 --d_model 64
"""

import argparse
import json
import time
from pathlib import Path

import torch

from qtrm_mm.config import QTRMConfig
from qtrm_mm.core import QTRMRecursiveCore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=15, help="Very small number of steps for minimal base")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--save_name", type=str, default="base_for_matched_a9617cd8_port_test.pt")
    args = parser.parse_args()

    print("=== Creating Minimal Base Checkpoint for Matched Port Validation ===\n")

    # Use the ported core with the mechanisms we want to evaluate later
    core_cfg = QTRMConfig(
        d_model=args.d_model,
        n_core_layers=2,
        outer_steps=3,
        h_cycles=1,
        l_cycles=1,
        core_stochastic_breadth_enabled=True,   # Ported feature ON in this base
        core_equation_binding_enabled=True,     # Ported feature ON in this base
        core_stochastic_mode="delta",
        core_stochastic_scale=0.06,
    )

    core = QTRMRecursiveCore(core_cfg)
    core.train()

    print(f"Core created with FULL PORT (stochastic + eq_binding) at a9617cd8")
    print(f"Training for {args.steps} steps to create minimal base...\n")

    for step in range(args.steps):
        workspace = torch.randn(args.batch, args.seq_len, args.d_model)

        # This triggers the ported stochastic breadth + any binding logic
        z_l, z_h, trajectory, halt_info = core(workspace)

        if step % 5 == 0 or step == args.steps - 1:
            print(f"Step {step:3d} | z_h norm: {z_h.norm().item():.4f}")

    # Save with rich metadata for protocol compliance
    save_path = Path(args.save_name)
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "worktree": "a9617cd8",
        "commit": "a9617cd",
        "purpose": "Minimal base checkpoint for future conditions-matched continuation experiments on stochastic + gated binding port",
        "core_flags": {
            "core_stochastic_breadth_enabled": True,
            "core_equation_binding_enabled": True,
        },
        "steps": args.steps,
        "d_model": args.d_model,
        "protocol": "FAIR_COMPARISON_PROTOCOL.md",
        "note": "This is a synthetic short base. Not for direct reasoning claims. Use only as starting point for matched on/off ablations.",
    }

    torch.save({
        "model_state_dict": core.state_dict(),
        "config": core_cfg.__dict__ if hasattr(core_cfg, "__dict__") else str(core_cfg),
        "metadata": metadata,
    }, save_path)

    print(f"\n=== Base checkpoint created ===")
    print(f"Saved to: {save_path.resolve()}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")
    print("\nThis file can now be used as the starting point for matched continuation runs.")


if __name__ == "__main__":
    main()
