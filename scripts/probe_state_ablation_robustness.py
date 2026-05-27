#!/usr/bin/env python3
"""
State Ablation Robustness Probe for 5.56 Curriculum Checkpoints (v2 - 2026-05-30)

This is a Phase-1 proxy for the historical "state_ablation_median" idea.

v2 Improvements:
- Structured, temporally correlated input sequences (instead of i.i.d. Gaussian)
- Proper parallel rollout + trajectory divergence metric

These changes make the model actually rely on its recurrent state, so ablation effects become measurable.

Usage (example):
    python scripts/probe_state_ablation_robustness.py \
        --ckpt local_556_real642_long_180step_20260527_1201/best.pt \
        --steps 40 --trials 6 --ablation zero
"""

import argparse
import torch
from pathlib import Path

from src.qtrm_mm.config import QTRMConfig
from src.qtrm_mm.core import QTRMRecursiveCore


def load_trainer_checkpoint(ckpt_path: str, device="cpu"):
    """Load a checkpoint saved by train_556_full_curriculum_minimal.py"""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("config", {})

    core_cfg = QTRMConfig(
        d_model=cfg_dict.get("d_model", 64),
        n_core_layers=4,
        outer_steps=6,
        h_cycles=1,
        l_cycles=2,
        core_stochastic_breadth_enabled=cfg_dict.get("enable_stochastic_breadth", False),
        core_stochastic_breadth_ablation_zero=cfg_dict.get("stochastic_breadth_ablation_zero", False),
    )

    core = QTRMRecursiveCore(core_cfg).to(device)
    core.load_state_dict(ckpt["core_state_dict"], strict=False)
    core.eval()
    return core, core_cfg


def generate_structured_workspace_sequence(batch, seq_len, d_model, total_steps, device="cpu"):
    """
    v2 improvement (2026-05-30):
    Generate a sequence of temporally correlated workspaces instead of i.i.d. noise.
    This forces the model to actually use its recurrent state to track slow changes.
    """
    torch.manual_seed(42)
    # Base pattern per batch item (slowly evolving "context")
    base = torch.randn(batch, 1, d_model, device=device) * 0.5

    workspaces = []
    current = torch.randn(batch, seq_len, d_model, device=device) * 0.3

    for t in range(total_steps):
        # Slow drift + small innovation (AR-like process)
        drift = 0.08 * torch.randn(batch, seq_len, d_model, device=device)
        current = 0.92 * current + drift + 0.03 * base

        # Add mild low-frequency modulation across time
        modulation = 0.15 * torch.sin(torch.tensor(t * 0.15, device=device))
        current = current + modulation * base

        workspaces.append(current.clone())

    return workspaces


def run_with_ablation(core, batch=4, seq_len=32, steps=40, ablation="zero", strength=0.7, device="cpu"):
    """
    Improved Phase-1 probe (2026-05-30 v2):
    - Uses structured, temporally correlated input sequences (v2).
    - Runs two parallel branches after the ablation point.
    - Measures trajectory divergence of z_h over time.

    Returns:
        divergence_score (higher = less robust after ablation)
        clean_final_norm (for reference)
    """
    total_steps = steps
    workspaces = generate_structured_workspace_sequence(batch, seq_len, core.cfg.d_model, total_steps, device)

    # === Warmup (shared) ===
    z_h = None
    for t in range(steps // 2):
        z_l, z_h, traj, halt = core(workspaces[t])

    # === Create two branches ===
    z_h_clean = z_h.clone()
    z_h_ablated = z_h.clone()

    # Apply ablation only to the ablated branch
    if ablation == "zero":
        z_h_ablated = z_h_ablated * 0.0
    elif ablation == "noise":
        noise = torch.randn_like(z_h_ablated) * strength
        z_h_ablated = z_h_ablated + noise

    # === Continue in parallel and measure divergence ===
    divergences = []
    for t in range(steps // 2, total_steps):
        # Clean branch
        _, z_h_clean, _, _ = core(workspaces[t])

        # Ablated branch
        _, z_h_ablated, _, _ = core(workspaces[t])

        dist = (z_h_clean - z_h_ablated).norm(dim=-1).mean().item()
        divergences.append(dist)

    mean_divergence = sum(divergences) / len(divergences) if divergences else 0.0
    final_clean_norm = z_h_clean.norm(dim=-1).mean().item()

    return mean_divergence, final_clean_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--ablation", choices=["zero", "noise"], default="zero")
    parser.add_argument("--strength", type=float, default=0.7)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint: {args.ckpt}")
    core, _ = load_trainer_checkpoint(args.ckpt, device=device)

    degradations = []
    for t in range(args.trials):
        deg, post = run_with_ablation(
            core,
            steps=args.steps,
            ablation=args.ablation,
            strength=args.strength,
            device=device
        )
        degradations.append(deg)
        print(f"Trial {t+1}: degradation = {deg:.4f}")

    mean_div = sum(degradations) / len(degradations)
    print(f"\n=== Proxy State Ablation Robustness (v2 - Structured Inputs + Trajectory Divergence) ===")
    print(f"Ablation type : {args.ablation}")
    print(f"Mean divergence after ablation: {mean_div:.4f}")
    print(f"(Higher = less robust. Uses temporally correlated inputs so the model must use recurrent state.)")


if __name__ == "__main__":
    main()
