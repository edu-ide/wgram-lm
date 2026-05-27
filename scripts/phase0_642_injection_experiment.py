#!/usr/bin/env python3
"""
Improved B: 642 Gold State Partial Injection + Binding Pressure Experiment

Attempts to take useful state from the historical 5.53 gold checkpoint
and inject it into the current QTRMRecursiveCore as initial carry / latent.

This is the next step after simple proxy: trying to make the gold state "live" inside the new architecture.

Run:
  .venv/bin/python scripts/phase0_642_injection_experiment.py --steps 40 --binding-weight 0.15
"""

import argparse
import torch
from pathlib import Path
from src.qtrm_mm.config import QTRMConfig
from src.qtrm_mm.core import QTRMRecursiveCore

# Robust import for equation_state_binding
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOSS_FILE = ROOT / "src" / "qtrm_mm" / "losses" / "equation_state_binding.py"
_loss_globals = {"__name__": "equation_state_binding"}
with open(LOSS_FILE, "r", encoding="utf-8") as f:
    exec(compile(f.read(), str(LOSS_FILE), "exec"), _loss_globals)

compute_equation_state_binding_loss = _loss_globals["compute_equation_state_binding_loss"]
EquationStateBindingConfig = _loss_globals["EquationStateBindingConfig"]
LightweightTypedEquationHead = _loss_globals["LightweightTypedEquationHead"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--binding-weight", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-injection", action="store_true", help="Disable 642 state injection (ablation)")
    parser.add_argument("--no-rehearsal", action="store_true", help="Disable rehearsal simulation (ablation)")
    parser.add_argument("--no-binding", action="store_true", help="Disable binding pressure (ablation)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = "local_eval/642_adaptive_fine_tuned_200step/adaptive_phase2_checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    old_state = ckpt["model_state_dict"]

    print("=== Phase 0 B Improved: 642 State Injection Experiment ===")
    print(f"Loaded 642 ckpt with {len(old_state)} parameters")

    # Current architecture config (matching recent I→G→A work)
    cfg = QTRMConfig(
        d_model=256, d_ff=768, n_heads=4, n_kv_heads=2,
        n_prelude_layers=2, n_core_layers=2, max_seq_len=128, vocab_size=8192,
        core_equation_binding_enabled=True,
        core_answer_attractor_enabled=True,
    )

    core = QTRMRecursiveCore(cfg).to(device)

    # Try to extract useful initial state from old checkpoint
    # Look for latent-like tensors
    latent_candidates = {}
    for k, v in old_state.items():
        if isinstance(v, torch.Tensor) and v.numel() > 200 and 'latent' in k.lower():
            latent_candidates[k] = v

    print(f"Found {len(latent_candidates)} latent-like tensors in 642 ckpt")

    # Use bos_latent or similar as initial z_h proxy if possible
    initial_state = None
    if 'bos_latent' in old_state:
        initial_state = old_state['bos_latent'].to(device)
        print("Using 'bos_latent' from 642 as initial state signal")
    elif latent_candidates:
        key = list(latent_candidates.keys())[0]
        initial_state = latent_candidates[key].to(device)
        print(f"Using {key} as initial state signal")

    # Create input
    batch, seq, d = 4, 32, 256
    ws = torch.randn(batch, seq, d, device=device)

    # Run with injection attempt (we feed the old latent as extra conditioning if possible)
    if initial_state is not None and not args.no_injection:
        inj = initial_state[:256].view(1, 256)
        base_injection = inj.repeat(batch, 1) * 0.1
        print("Injection: ON (using 642 bos_latent)")
    else:
        base_injection = torch.zeros(batch, 256, device=device)
        print("Injection: OFF (ablation)")

    effective_binding_weight = 0.0 if args.no_binding else args.binding_weight
    print(f"Binding pressure: {'ON' if not args.no_binding else 'OFF'} (weight={effective_binding_weight})")
    print(f"Rehearsal simulation: {'ON' if not args.no_rehearsal else 'OFF'}")

    print(f"\nRunning {args.steps} steps...")

    head = LightweightTypedEquationHead(EquationStateBindingConfig(d_state=256)).to(device)
    total_aux = 0.0
    rehearsal_boost = 0.0

    for step in range(1, args.steps + 1):
        # Normal forward
        z_l, z_h, _, halt = core(ws, return_carry=True)

        # Rehearsal simulation
        if not args.no_rehearsal:
            if step % 5 == 0:
                rehearsal_boost = 0.25
            else:
                rehearsal_boost *= 0.7
        else:
            rehearsal_boost = 0.0

        # Combined effect
        current_injection = base_injection * (1.0 + rehearsal_boost) if not args.no_injection else torch.zeros_like(base_injection)
        pooled = z_h.mean(dim=1) + current_injection

        loss, diags = compute_equation_state_binding_loss(
            pooled,
            target_left=torch.tensor([5.0] * batch, device=device),
            target_right=torch.tensor([8.0] * batch, device=device),
            target_op=torch.tensor([0] * batch, device=device),
            head=head,
        )
        aux = loss.item() * effective_binding_weight
        total_aux += aux

        if step % 10 == 0:
            print(f"step {step:03d} | aux {aux:.4f} | rehearsal {rehearsal_boost:.3f}")

    mode = []
    if not args.no_injection: mode.append("Injection")
    if not args.no_rehearsal: mode.append("Rehearsal")
    if not args.no_binding: mode.append("Binding")
    mode_str = "+".join(mode) if mode else "All OFF"

    print(f"\n[Phase 0 Result]")
    print(f"Mode: {mode_str}")
    print(f"Avg aux = {total_aux / args.steps:.4f}")

if __name__ == "__main__":
    main()