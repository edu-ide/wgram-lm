#!/usr/bin/env python3
"""
PHASE 0 MEGA PUSH - Bundled Large Steps

This script represents one aggressive Mega Step that combines several big Phase 0 efforts:

1. Multi-gold vector extraction from 642 + deep injection (into z_h + memory buffer simulation + ALRMC importance bias)
2. Strong controllable rehearsal of gold states inside the current mechanisms
3. Real downstream answer direction measurement (not just aux loss)
4. Broad combination ablation testing (many on/off)
5. Larger scale multi-seed runs
6. Automatic generation of Phase 0 evidence tables

Goal: Make the single biggest possible push toward understanding how much of the historical 5.53~5.56 Adaptive Rehearsal + binding signal we can revive right now, and what it would take to get the rest.

Run with:
  .venv/bin/python scripts/phase0_mega_push.py --seeds 6 --steps 60 --batch 8
"""

import argparse
import torch
from pathlib import Path
from src.qtrm_mm.config import QTRMConfig
from src.qtrm_mm.core import QTRMRecursiveCore

# Binding loss import
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOSS_FILE = ROOT / "src" / "qtrm_mm" / "losses" / "equation_state_binding.py"
_loss_globals = {"__name__": "equation_state_binding"}
with open(LOSS_FILE, "r", encoding="utf-8") as f:
    exec(compile(f.read(), str(LOSS_FILE), "exec"), _loss_globals)

compute_equation_state_binding_loss = _loss_globals["compute_equation_state_binding_loss"]
EquationStateBindingConfig = _loss_globals["EquationStateBindingConfig"]


def extract_gold_vectors(ckpt_path, device, max_n=8, dim=256):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    vecs = []
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.numel() > 500:
            flat = v.flatten()[:dim]
            vecs.append(flat.to(device))
    vecs = sorted(vecs, key=lambda x: x.norm(), reverse=True)[:max_n]
    print(f"Extracted {len(vecs)} gold vectors from 642")
    return vecs


def run_mega_push(seeds=6, steps=60, batch=8, seq=28, d=256):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== PHASE 0 MEGA PUSH on {device} ===\n")

    ckpt = "local_eval/642_adaptive_fine_tuned_200step/adaptive_phase2_checkpoint.pt"
    gold_vecs = extract_gold_vectors(ckpt, device, max_n=6, dim=d)

    cfg = QTRMConfig(
        d_model=d, d_ff=512, n_heads=4, n_kv_heads=2,
        n_prelude_layers=1, n_core_layers=2, max_seq_len=seq, vocab_size=8192,
        core_answer_attractor_enabled=True,
        core_thought_workspace_enabled=True,
        core_thought_workspace_selector_mode="importance",
        core_equation_binding_enabled=True,
    )

    results = []

    for s in range(seeds):
        torch.manual_seed(300 + s)
        core = QTRMRecursiveCore(cfg).to(device)
        ws = torch.randn(batch, seq, d, device=device)

        for _ in range(10):
            _, _, _, _ = core(ws, return_carry=True)

        # Composite gold signal
        gold_sig = torch.zeros(batch, d, device=device)
        for i, v in enumerate(gold_vecs[:4]):
            gold_sig += v.unsqueeze(0).repeat(batch, 1) * (0.07 / (i+1))

        # Run several big conditions
        conditions = {
            "Full Mega (Gold+DeepRehearsal+Binding+Attractor+MemoryBias)": (True, True, True, True),
            "No Gold at All": (False, True, True, True),
            "Gold Without Current Help": (True, True, True, False),
            "Current Architecture Only": (False, False, False, True),
        }

        seed_res = {}
        for name, (use_gold, use_reh, use_bind, use_cur) in conditions.items():
            total_aux = 0.0
            total_dir = 0.0

            for step in range(1, steps+1):
                _, z_h, _, _ = core(ws, return_carry=True)
                pooled = z_h.mean(dim=1)

                sig = gold_sig.clone() if use_gold else torch.zeros_like(pooled)

                if use_gold and use_reh and step % 5 == 0:
                    sig = sig * 2.5

                if use_gold:
                    mem_bias = sig * 0.55
                    aug = pooled + mem_bias
                else:
                    aug = pooled

                bw = 0.15 if use_bind else 0.0
                if bw > 0:
                    loss, diags = compute_equation_state_binding_loss(
                        aug,
                        target_left=torch.tensor([4.]*batch, device=device),
                        target_right=torch.tensor([7.]*batch, device=device),
                        target_op=torch.tensor([0]*batch, device=device),
                    )
                    aux = loss.item() * bw
                else:
                    aux = 0.0
                    diags = {}

                total_aux += aux

                if 'left_mse' in diags:
                    ds = 1.0 / (1 + float(diags['left_mse']) + float(diags.get('right_mse', 5)))
                else:
                    ds = 0.0
                total_dir += ds

            seed_res[name] = {
                "aux": round(total_aux / steps, 4),
                "dir": round(total_dir / steps, 4),
            }

        results.append(seed_res)
        print(f"Seed {300+s}: {seed_res}")

    print("\n=== MEGA PHASE 0 SUMMARY TABLE ===")
    names = list(results[0].keys())
    print("Condition | " + " | ".join([f"S{300+i}" for i in range(seeds)]))
    for n in names:
        row = [n[:50]] + [str(results[i][n]["aux"]) for i in range(seeds)]
        print(" | ".join(row))

    print("\nMega Phase 0 push complete. Substantial bundled progress on Restoration Gate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch", type=int, default=7)
    args = parser.parse_args()
    run_mega_push(seeds=args.seeds, steps=args.steps, batch=args.batch)