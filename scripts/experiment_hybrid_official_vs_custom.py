#!/usr/bin/env python3
"""
Small-scale controlled experiment for OneBodyParallelHybridBlock.

Compares different combinations of recurrence (custom V2 vs official GDN2)
and attention (GQA vs official MLA) under the same conditions.

Metrics:
- Hidden state norm trajectory over recurrent steps (stability)
- Stochastic breadth effect size (mean change when noise is injected)
- Stochastic ablation identity (how close to identity when ablation_zero=True)

Run with the project venv, preferably on GPU with bfloat16.
"""

import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.qtrm_mm.config import QTRMConfig
from src.qtrm_mm.blocks import OneBodyParallelHybridBlock


def run_recurrent_experiment(
    name: str,
    delta_backend: str,
    attention_type: str,
    num_steps: int = 12,
    batch_size: int = 2,
    seq_len: int = 8,
    d_model: int = 128,
    seed: int = 42,
):
    torch.manual_seed(seed)

    cfg = QTRMConfig(
        d_model=d_model,
        n_heads=8,
        n_kv_heads=4,
        d_ff=256,
        max_seq_len=512,
        delta_backend=delta_backend,
    )

    block = OneBodyParallelHybridBlock(
        cfg=cfg,
        recurrence_head_count=3,
        attention_head_count=2,
        attention_type=attention_type,
    ).to(device="cuda", dtype=torch.bfloat16)

    # Force use of official MLA when requested (for clearer logging)
    if attention_type == "mla":
        print(f"  [Info] attention_type=mla → will attempt official FLA MLA")

    block.eval()

    x = torch.randn(batch_size, seq_len, d_model, device="cuda", dtype=torch.bfloat16)

    norms = []
    stochastic_effect = []

    for step in range(num_steps):
        noise = None
        if step % 3 == 0:  # inject stochastic breadth every 3 steps
            noise = torch.randn_like(x) * 0.04

        with torch.no_grad():
            x_no_noise = block(x, stochastic_breadth_noise=None)
            x_with_noise = block(x, stochastic_breadth_noise=noise) if noise is not None else x_no_noise

        norms.append(x_no_noise.norm(dim=-1).mean().item())

        if noise is not None:
            effect = (x_with_noise - x_no_noise).abs().mean().item()
            stochastic_effect.append(effect)

    # Ablation identity check at the end
    with torch.no_grad():
        block._stochastic_breadth_ablation_zero = True
        x_abl = block(x, stochastic_breadth_noise=noise if 'noise' in locals() else None)
        block._stochastic_breadth_ablation_zero = False

        ablation_diff = (x_abl - x_no_noise).abs().max().item()

    return {
        "name": name,
        "final_norm": norms[-1],
        "norm_growth": norms[-1] / norms[0],
        "avg_stochastic_effect": sum(stochastic_effect) / len(stochastic_effect) if stochastic_effect else 0.0,
        "ablation_max_diff": ablation_diff,
        "norms": norms,
    }


def main():
    print("=== Hybrid Block Official vs Custom Experiment ===\n")
    print("Using bfloat16 + CUDA. All blocks moved to device.\n")

    configs = [
        ("Custom V2 + GQA", "torch_gated_delta2_v2", "gqa"),
        ("Custom V2 + Official MLA", "torch_gated_delta2_v2", "mla"),
        ("Official GDN2 + GQA", "official_gated_delta2", "gqa"),
        ("Official GDN2 + Official MLA", "official_gated_delta2", "mla"),
    ]

    results = []

    for name, delta_backend, att_type in configs:
        print(f"Running: {name} ...")
        try:
            res = run_recurrent_experiment(name, delta_backend, att_type)
            results.append(res)
            print(f"  Final norm: {res['final_norm']:.3f} (growth x{res['norm_growth']:.2f})")
            print(f"  Avg stochastic effect: {res['avg_stochastic_effect']:.4f}")
            print(f"  Ablation max diff: {res['ablation_max_diff']:.2e}")
        except Exception as e:
            print(f"  FAILED: {e}")
        print()

    print("\n=== Summary Table ===")
    print(f"{'Configuration':<30} | {'Final Norm':>10} | {'Growth':>8} | {'Stoch Effect':>12} | {'Ablation Diff':>14}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<30} | {r['final_norm']:>10.3f} | x{r['norm_growth']:>6.2f} | {r['avg_stochastic_effect']:>12.4f} | {r['ablation_max_diff']:>14.2e}")


if __name__ == "__main__":
    main()
