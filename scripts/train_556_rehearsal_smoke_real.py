#!/usr/bin/env python3
"""
train_556_rehearsal_smoke_real.py

Minimal but realistic trainer-style loop demonstrating the full integration:

QTRMRecursiveCore + AdaptiveRehearsal.full_curriculum_rehearsal_step 
+ Stochastic Breadth (Reverse I→G→A)

This is the direct next step after:
- scripts/example_556_full_curriculum_wiring.py
- scripts/diag_556_rehearsal_curriculum_smoke.py
- The extensions to adaptive_rehearsal.py (2026-05-30)

References:
- docs/wiki/decisions/2026-05-30-deep-dive-full-556-rehearsal-curriculum.md

Run (in a proper environment with torch + the package):
PYTHONPATH=. python scripts/train_556_rehearsal_smoke_real.py --steps 30
"""

import argparse
import torch
from dataclasses import dataclass

# Real imports
from src.qtrm_mm.config import QTRMConfig
from src.qtrm_mm.core import QTRMRecursiveCore
from src.qtrm_mm.rehearsal.adaptive_rehearsal import AdaptiveRehearsal, RehearsalConfig


@dataclass
class SmokeTrainConfig:
    d_model: int = 64
    steps: int = 30
    batch: int = 4
    seq_len: int = 16
    enable_stochastic_breadth: bool = True


def build_core_and_rehearsal(cfg: SmokeTrainConfig, total_steps: int):
    core_cfg = QTRMConfig(
        d_model=cfg.d_model,
        n_core_layers=2,
        outer_steps=4,
        h_cycles=1,
        l_cycles=1,
        core_stochastic_breadth_enabled=cfg.enable_stochastic_breadth,
        core_stochastic_mode="delta",
        core_stochastic_scale=0.06,
        core_stochastic_breadth_ablation_zero=False,
    )

    core = QTRMRecursiveCore(core_cfg)

    reh_cfg = RehearsalConfig(
        enabled=True,
        scheduled_binding_decay_start=0.40,
        scheduled_binding_decay_end=0.04,
        gold_state_injection_alpha=0.25,
        protect_attractor=True,
        attractor_protection_during_rehearsal=0.7,
    )

    rehearsal = AdaptiveRehearsal(reh_cfg, core_cfg)
    rehearsal.set_total_steps(total_steps)

    return core, rehearsal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--d_model", type=int, default=64)
    args = parser.parse_args()

    print("=== 5.56 Rehearsal Curriculum + Stochastic Breadth Real Training Smoke ===\n")

    train_cfg = SmokeTrainConfig(d_model=args.d_model, steps=args.steps)
    core, rehearsal = build_core_and_rehearsal(train_cfg, total_steps=args.steps)

    # Fake gold state (proxy for 642)
    gold_state = torch.randn(train_cfg.d_model) * 0.08

    # Training-like loop
    for step in range(args.steps):
        # Simulate input workspace (in real trainer this comes from backbone)
        workspace = torch.randn(train_cfg.batch, train_cfg.seq_len, train_cfg.d_model)

        # Core forward (this is where stochastic breadth is applied internally if enabled)
        z_l, z_h, trajectory, halt_info = core(workspace)

        # Get memory buffer from core (simplified - in real code you'd maintain it properly)
        # For this smoke we use a dummy buffer derived from trajectory
        memory_buffer = [t.mean(1) for t in trajectory[-6:]] if trajectory else [z_h.mean(1)]

        # === The key call: full 5.56 curriculum step with stochastic breadth hook ===
        # (In real training this would be inside the rehearsal logic or after core update)
        z_h = rehearsal.full_curriculum_rehearsal_step(
            z_h=z_h,
            memory_buffer=memory_buffer,
            gold_state=gold_state,
            # The stochastic breadth is already handled inside core when enabled,
            # but we can also pass an explicit fn if we want more control:
            stochastic_breadth_fn=None,   # core already applies it during forward
        )

        if step % 5 == 0:
            print(f"Step {step:3d} | z_h norm: {z_h.norm().item():.4f} | "
                  f"binding_weight: {rehearsal.get_current_binding_weight():.3f}")

    print("\nReal training smoke completed.")
    print("This loop now uses the actual QTRMRecursiveCore + AdaptiveRehearsal.full_curriculum_rehearsal_step.")
    print("See the deep dive document for scaling this to real runs.")


if __name__ == "__main__":
    main()