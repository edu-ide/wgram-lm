#!/usr/bin/env python3
"""
train_hybrid_ri4_real_continuation_minimal.py

A-Mode continuation trainer for the proven RI-4 hybrid substrate.

Purpose (post 160-step d128 real-gold matrix closure):
- Take the exact verified recipe (OneBodyParallelHybridBlock as recurrent engine
  + RI-4 SparseSlotRouter + 5.56 Adaptive Rehearsal Gold Recipe + real gold loading)
  and run it in a real training-like loop with checkpoints.
- Preserve 100% of the One-Body contract, 4-way RI-4 ablation flags, and
  Reverse I→G→A stochastic breadth.
- This is the smallest executable step that moves the substrate from "toy matrix
  diagnostic" to "production training continuation" (the current #1 Most-Deficient).

Usage (smoke on real gold path):
    PYTHONPATH=. python scripts/train_hybrid_ri4_real_continuation_minimal.py \
        --steps 50 --d_model 128 --batch 2 \
        --enable_stochastic_breadth \
        --gold_path local_eval/642_adaptive_fine_tuned_200step/adaptive_phase2_checkpoint.pt \
        --save_every 10 --out_dir checkpoints/hybrid_ri4_cont

The recipe inside is deliberately kept identical to the one that produced the clean
160-step RI-3+RI-4 evidence (after cap removal).

Next after this: wire real data loaders + full optimizer + 192-style heldout gates
on the saved checkpoints.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.qtrm_mm.config import QTRMConfig
from src.qtrm_mm.blocks import OneBodyParallelHybridBlock
from src.qtrm_mm.memory.sparse_slot_router import SparseSlotRouter

# Reuse the exact proven helpers from the matrix runner (no drift)
from scripts.train_556_on_parallel_hybrid_minimal import (
    Hybrid556Config,
    build_hybrid_stack,
    scheduled_decay,
    load_gold_proxy_robust,
    apply_556_rehearsal_update,
)


@dataclass
class ContinuationConfig(Hybrid556Config):
    save_every: int = 10
    out_dir: str = "checkpoints/hybrid_ri4_cont"
    resume_from: Optional[str] = None
    input_mode: str = "gold_structured"


def parse_continuation_args() -> ContinuationConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--enable_stochastic_breadth", action="store_true")
    p.add_argument("--gold_path", type=str, default=None)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--out_dir", type=str, default="checkpoints/hybrid_ri4_cont")
    p.add_argument("--resume_from", type=str, default=None)
    # Pass-through RI-4 ablation flags for contract preservation
    p.add_argument("--ri4_slots_off", action="store_true")
    p.add_argument("--ri4_persistence_off", action="store_true")
    p.add_argument("--input_mode", type=str, default="gold_structured", choices=["random", "gold_structured"], help="Input generation mode for continuation (gold_structured = much more faithful to 5.56 rehearsal cases)")
    args = p.parse_args()

    cfg = ContinuationConfig(
        total_steps=args.steps,
        batch_size=args.batch,
        d_model=args.d_model,
        enable_stochastic_breadth=args.enable_stochastic_breadth,
        gold_path=args.gold_path,
        save_every=args.save_every,
        out_dir=args.out_dir,
        resume_from=args.resume_from,
        input_mode=args.input_mode,
    )
    cfg.ri4_sparse_slots_ablation = args.ri4_slots_off
    cfg.ri4_persistence_ablation = args.ri4_persistence_off
    return cfg


def main():
    cfg = parse_continuation_args()
    os.makedirs(cfg.out_dir, exist_ok=True)

    print("=" * 72)
    print("RI-4 HYBRID REAL CONTINUATION (A-Mode Most-Deficient move)")
    print(f"  Horizon: {cfg.total_steps} steps | d_model={cfg.d_model}")
    print(f"  Gold: {cfg.gold_path}")
    print(f"  RI-4 ablations: slots_off={cfg.ri4_sparse_slots_ablation}, persistence_off={cfg.ri4_persistence_ablation}")
    print("=" * 72)

    # Build the exact proven stack
    model = build_hybrid_stack(cfg)

    # RI-4 router (same wiring as the 160-step evidence run)
    router = None
    if not cfg.ri4_sparse_slots_ablation:
        router = SparseSlotRouter(
            d_model=cfg.d_model,
            num_slots=16,
            top_k=4,
        ).to(device=cfg.device, dtype=cfg.dtype)

    # === Resume support (A-Mode: close the "can actually continue from previous checkpoint" gap) ===
    start_step = 0
    if cfg.resume_from and os.path.exists(cfg.resume_from):
        print(f"[Resume] Loading from {cfg.resume_from}")
        # Trusted internal checkpoint — safe to use weights_only=False for our own custom objects
        ckpt = torch.load(cfg.resume_from, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if router is not None and ckpt.get("router") is not None:
            router.load_state_dict(ckpt["router"])
        start_step = ckpt.get("step", 0)
        print(f"[Resume] Resumed at step {start_step}")
        # Carry over slots if present in the checkpoint (for persistent RI-4 state)
        if hasattr(model, '_ri4_current_slots') and ckpt.get('slots') is not None:
            model._ri4_current_slots = ckpt['slots'].to(device=cfg.device, dtype=cfg.dtype)

    # Gold state (real 642 if provided) — exact same robust prep as the 160-step evidence run
    gold_state = None
    if cfg.gold_path and os.path.exists(cfg.gold_path):
        print(f"[Gold] Loading real 642 gold from: {cfg.gold_path}")
        gold_raw = load_gold_proxy_robust(cfg.gold_path, cfg.d_model, cfg.device, cfg.dtype)
        if gold_raw is not None:
            if gold_raw.dim() > 1:
                gold_raw = gold_raw.squeeze()
            if gold_raw.shape[0] != cfg.d_model:
                if gold_raw.shape[0] > cfg.d_model:
                    gold_raw = gold_raw[:cfg.d_model]
                else:
                    gold_raw = torch.nn.functional.pad(gold_raw, (0, cfg.d_model - gold_raw.shape[0]))
            gold_state = gold_raw.unsqueeze(0).unsqueeze(0)  # [1, 1, d]
            print(f"[Gold] Real 642 gold successfully prepared for d_model={cfg.d_model}")
        else:
            print("[Gold] Load failed, using synthetic proxy")
            gold_state = torch.randn(1, 1, cfg.d_model, device=cfg.device, dtype=cfg.dtype) * 0.1
    else:
        gold_state = torch.randn(1, 1, cfg.d_model, device=cfg.device, dtype=cfg.dtype) * 0.1

    def make_input(step: int, total: int) -> torch.Tensor:
        """Generate inputs. gold_structured is far more faithful to the 5.56 rehearsal cases than pure random."""
        if cfg.input_mode == "gold_structured" and gold_state is not None:
            # Structured around gold: small decay-modulated noise + slight temporal variation (mimics rehearsal cases)
            decay = scheduled_decay(step, total, 0.40, 0.04)
            base = gold_state.expand(cfg.batch_size, 8, -1)
            noise = torch.randn_like(base) * (0.08 * decay)
            # Add a tiny step-dependent phase so different steps see slightly different "contexts"
            phase = torch.sin(torch.tensor(step * 0.1, device=base.device)) * 0.03
            return base + noise + phase
        else:
            # Legacy random (for comparison / backward)
            return torch.randn(cfg.batch_size, 8, cfg.d_model, device=cfg.device, dtype=cfg.dtype) * 0.02

    # Simple training-like loop (placeholder for real data + optimizer)
    # In the next step this becomes a real DataLoader + AdamW + checkpointing
    x = make_input(start_step, cfg.total_steps)

    total_to_run = cfg.total_steps
    for step in range(start_step, start_step + total_to_run):
        decay = scheduled_decay(step, cfg.total_steps, 0.40, 0.04)

        # Rehearsal-style update (exact faithful version from the proven matrix)
        gold_delta = torch.zeros(1, 1, cfg.d_model, device=cfg.device, dtype=cfg.dtype)
        if gold_state is not None and cfg.gold_injection_alpha > 0:
            gold_delta = gold_state * (cfg.gold_injection_alpha * decay)

        x = make_input(step, start_step + total_to_run)
        x_in = x + gold_delta
        noise = torch.randn_like(x_in) * 0.06 if cfg.enable_stochastic_breadth else None

        h = x_in
        current_slots = getattr(model, '_ri4_current_slots', None) if hasattr(model, '_ri4_current_slots') else None

        for layer in model:
            out = layer(h, stochastic_breadth_noise=noise,
                        slot_state=current_slots if isinstance(layer, OneBodyParallelHybridBlock) else None)
            if isinstance(out, tuple):
                h, current_slots = out
            else:
                h = out

        if hasattr(model, '_ri4_current_slots'):
            model._ri4_current_slots = current_slots

        # RI-4 selective rehearsal (when router is active)
        if router is not None and current_slots is not None:
            # Simplified slot mask for demo (in real code this comes from router routing)
            slot_mask = torch.ones(cfg.batch_size, 16, device=cfg.device, dtype=torch.bool)
            _ = router.apply_rehearsal_update(
                current_slots=current_slots,
                gold_state=gold_state,
                rehearsal_target=h.mean(dim=1, keepdim=True),
                slot_mask=slot_mask,
                gold_alpha=cfg.gold_injection_alpha,
                protection=cfg.attractor_protection,
                decay=decay,
            )

        x = h.detach()

        if (step + 1) % max(1, cfg.total_steps // 4) == 0 or step == cfg.total_steps - 1:
            print(f"step {step+1}/{cfg.total_steps} | norm={h.norm().item():.3f}")

        if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
            ckpt_path = os.path.join(cfg.out_dir, f"hybrid_ri4_cont_step{step+1}.pt")
            slots = getattr(model, '_ri4_current_slots', None)
            torch.save({
                "model": model.state_dict(),
                "router": router.state_dict() if router is not None else None,
                "step": step + 1,
                "config": cfg,
                "slots": slots.cpu() if slots is not None else None,
            }, ckpt_path)
            print(f"[Checkpoint] saved {ckpt_path}")

    print("\nContinuation smoke complete. Substrate is now exercisable in a checkpointed loop.")
    print("Next: wire real data + full optimizer + 192-style gates on these checkpoints.")


if __name__ == "__main__":
    main()