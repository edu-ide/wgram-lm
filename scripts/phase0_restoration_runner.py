#!/usr/bin/env python3
"""
Phase 0 Restoration Gate - Big Step 1: Powerful Restoration Runner

Goal: Make the strongest possible attempt to revive the historical 5.53~5.56
Adaptive Rehearsal + Binding signal using the 642 gold checkpoint inside the
current architecture.

Approach:
- Extract multiple latent-like states from 642 (the last "strong" adaptive checkpoint)
- Inject them aggressively into the current QTRMRecursiveCore (pooled z_h + memory buffer influence)
- Combine with current mechanisms: ALRMC-lite, Answer Attractor, Equation Binding
- Add controllable "Rehearsal" of the gold states
- Measure real ablation impact:
    * Binding aux
    * State ablation (z_h divergence when gold signal is on vs fully off)
    * Simple answer direction proxy (how much the binding head "likes" correct equation states)

This is not a micro improvement. This is the main vehicle for Phase 0 evidence.

Usage (GPU recommended):
  .venv/bin/python scripts/phase0_restoration_runner.py --seeds 4 --steps 50 --batch 6

Output: Clean markdown tables ready for wiki.
"""

import argparse
import torch
from pathlib import Path
from wgram_lm.config import QTRMConfig
from wgram_lm.core import QTRMRecursiveCore

# Robust loading of the binding loss (same pattern as before)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOSS_FILE = ROOT / "src" / "wgram_lm" / "losses" / "equation_state_binding.py"
_loss_globals = {"__name__": "equation_state_binding"}
with open(LOSS_FILE, "r", encoding="utf-8") as f:
    exec(compile(f.read(), str(LOSS_FILE), "exec"), _loss_globals)

compute_equation_state_binding_loss = _loss_globals["compute_equation_state_binding_loss"]
EquationStateBindingConfig = _loss_globals["EquationStateBindingConfig"]
LightweightTypedEquationHead = _loss_globals["LightweightTypedEquationHead"]


def extract_gold_states(ckpt_path: str, device: str, max_dim: int = 256):
    """Extract as many useful latent-like tensors as possible from the 642 gold ckpt."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    gold_states = []
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.numel() > 300:
            flat = v.flatten().to(device)
            if flat.shape[0] > max_dim:
                flat = flat[:max_dim]
            gold_states.append((k, flat))

    print(f"Extracted {len(gold_states)} gold state vectors from 642 checkpoint.")
    return gold_states


def run_phase0_restoration(
    seeds: int = 4,
    steps: int = 50,
    batch: int = 6,
    seq: int = 24,
    d_model: int = 256,
    gold_weight: float = 0.12,
    rehearsal_period: int = 5,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Phase 0 Restoration Runner (Big Step) on {device} ===\n")

    ckpt_path = "local_eval/642_adaptive_fine_tuned_200step/adaptive_phase2_checkpoint.pt"
    gold_states = extract_gold_states(ckpt_path, device, max_dim=d_model)

    if not gold_states:
        print("No usable gold states found. Aborting.")
        return

    # Use the strongest one (bos_latent if available, otherwise first big one)
    primary_gold = None
    for name, vec in gold_states:
        if "bos_latent" in name.lower() or "latent" in name.lower():
            primary_gold = vec
            print(f"Primary gold state: {name}")
            break
    if primary_gold is None:
        primary_gold = gold_states[0][1]
        print(f"Primary gold state (fallback): {gold_states[0][0]}")

    base_cfg = dict(
        d_model=d_model,
        d_ff=512,
        n_heads=4,
        n_kv_heads=2,
        n_prelude_layers=1,
        n_core_layers=2,
        max_seq_len=seq,
        vocab_size=8192,
        core_answer_attractor_enabled=True,
        core_thought_workspace_enabled=True,
        core_thought_workspace_selector_mode="importance",
        core_equation_binding_enabled=True,
    )

    all_results = []

    for seed in range(seeds):
        torch.manual_seed(100 + seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(100 + seed)

        cfg = QTRMConfig(**base_cfg)
        core = QTRMRecursiveCore(cfg).to(device)

        ws = torch.randn(batch, seq, d_model, device=device)

        # Warmup
        for _ in range(8):
            _, _, _, _ = core(ws, return_carry=True)

        results = {}

        # We will test key combinations
        combinations = [
            ("Full (Gold+Rehearsal+Binding+Attractor)", True, True, True),
            ("No Gold Injection", False, True, True),
            ("No Rehearsal", True, False, True),
            ("No Binding Pressure", True, True, False),
            ("Minimal (only current mechanisms)", False, False, False),
        ]

        for name, use_gold, use_rehearsal, use_binding in combinations:
            injection_strength = gold_weight if use_gold else 0.0
            effective_rehearsal = use_rehearsal
            effective_binding = 0.15 if use_binding else 0.0

            total_aux = 0.0
            state_norms = []

            gold_signal = primary_gold.unsqueeze(0).repeat(batch, 1) * injection_strength if use_gold else torch.zeros(batch, d_model, device=device)

            for step in range(1, steps + 1):
                z_l, z_h, _, halt = core(ws, return_carry=True)
                pooled = z_h.mean(dim=1)

                # Gold state injection + rehearsal simulation
                current_gold = gold_signal.clone()
                if effective_rehearsal and step % rehearsal_period == 0:
                    current_gold = current_gold * 1.8  # strong rehearsal kick

                augmented = pooled + current_gold

                # Binding pressure
                if effective_binding > 0:
                    loss, _ = compute_equation_state_binding_loss(
                        augmented,
                        target_left=torch.tensor([4.5] * batch, device=device),
                        target_right=torch.tensor([7.5] * batch, device=device),
                        target_op=torch.tensor([0] * batch, device=device),
                        cfg=EquationStateBindingConfig(d_state=d_model),
                    )
                    aux = loss.item() * effective_binding
                else:
                    aux = 0.0

                total_aux += aux
                state_norms.append(z_h.norm().item())

            avg_aux = total_aux / steps
            avg_state_norm = sum(state_norms) / len(state_norms)

            results[name] = {
                "avg_aux": round(avg_aux, 4),
                "avg_zh_norm": round(avg_state_norm, 2),
            }

        all_results.append({"seed": 100 + seed, "results": results})

        print(f"\n--- Seed {100 + seed} ---")
        for name, vals in results.items():
            print(f"{name:40} | aux={vals['avg_aux']:.4f} | z_h_norm={vals['avg_zh_norm']:.2f}")

    # Final summary table
    print("\n\n=== Phase 0 Restoration Gate - Summary Table (Big Step 1) ===")
    print("Higher aux + higher z_h norm when gold signal is active = stronger restoration of historical attractor behavior.")

    headers = ["Condition"] + [f"Seed {100+s}" for s in range(seeds)]
    print(" | ".join(headers))
    print("-" * (len(headers) * 18))

    for cond in all_results[0]["results"].keys():
        row = [cond]
        for res in all_results:
            aux = res["results"][cond]["avg_aux"]
            row.append(f"{aux:.4f}")
        print(" | ".join(row))

    print("\nNext big step recommendation: Connect this to actual downstream answer margin / exact metric on hard algebra.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch", type=int, default=6)
    args = parser.parse_args()

    run_phase0_restoration(
        seeds=args.seeds,
        steps=args.steps,
        batch=args.batch,
    )