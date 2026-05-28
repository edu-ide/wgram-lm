#!/usr/bin/env python3
"""
Minimal 20-step Diagnostic Run for the June 2026 Explicit Attractor Solver Substrate
(Section 7 Overhaul — Proposal Engine + Dedicated AttractorSolverModule)

This is the first practical "배선 스크립트".

It deliberately keeps the harness tiny and self-contained so you can get real signals
on the new substrate (persistent y0 injection, Parcae negative diagonal, SOT segments,
first-class Equilibrium Internalization) within minutes, without fighting the complexity
of the full 72-measurement trainer.

Key behaviors exercised:
- Proposal y0 (even a small learned projection of input acts as "rich proposal engine" for diagnostic)
- AttractorSolverModule with Parcae stability + mandatory persistent proposal injection every step
- SOTSegmentedSolverTrainer (EqR style: short segments + immediate optimizer step + detached carry)
- Equilibrium Internalization loss as first-class term
- Slow context (summary) injection from brain triple memory when available

Ready-to-run command (copy-paste):

python scripts/diag_explicit_attractor_solver_20step.py \
  --steps 20 \
  --sot_segment_length 4 \
  --attractor_solver_weight 0.12 \
  --internalization_weight 0.10 \
  --out_dir checkpoints/diag_attractor_solver_20step \
  --device cuda

Watch for (per wiki Section 7.1):
- solver_residual trending down across the 20 steps
- internalization_loss trending down (y0 getting closer to equilibrium)
- No NaN / explosion (Parcae + SOT should prevent this)

If internalization is flat or solver residual stays high → we have hit one of the critical risks
already (rich proposal engine may already be too good, or basin shaping is insufficient).

After a clean 20-step run, promote the wiring into the real trainer for native 72 RI-1 measurement.
"""

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.qtrm_mm.attractor.attractor_solver import (
    AttractorSolverModule,
    SOTSegmentedSolverTrainer,
    SOTConfig,
)


def main():
    parser = argparse.ArgumentParser(description="20-step diagnostic for explicit attractor solver substrate")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sot_segment_length", type=int, default=4)
    parser.add_argument("--attractor_solver_weight", type=float, default=0.12)
    parser.add_argument("--internalization_weight", type=float, default=0.10)
    parser.add_argument("--ri_ni_scale", type=float, default=0.05)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="checkpoints/diag_attractor_solver_20step")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_every", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 72)
    print("EXPLICIT ATTRACTOR SOLVER — FIRST 20-STEP DIAGNOSTIC (Section 7)")
    print("=" * 72)
    print(f"Config: steps={args.steps}, sot_h={args.sot_segment_length}, "
          f"solver_w={args.attractor_solver_weight}, int_w={args.internalization_weight}")

    device = torch.device(args.device)
    D = args.d_model
    B = args.batch
    T = args.seq_len

    # === Minimal but realistic Proposal Engine (for the diagnostic) ===
    # In a full wiring this would be the real OneBodyParallelHybridBlock + FastGated + BrainTriple.
    # For the absolute first 20 steps we use a small learned projection as "proposal generator".
    # This is sufficient to test the solver dynamics, persistent injection, Parcae stability, SOT, and internalization.
    proposal_proj = nn.Linear(D, D).to(device)

    # === The New Core: Dedicated Attractor Solver ===
    solver = AttractorSolverModule(
        dim=D,
        num_layers=1,
        use_parcae=True,
        max_solver_steps=10,
        residual_tol=5e-3,
    ).to(device)

    sot_cfg = SOTConfig(
        segment_length=args.sot_segment_length,
        max_segments=2,
        ri_noise=args.ri_ni_scale,
        internalization_weight=args.internalization_weight,
        use_detached_carry=True,
    )

    def primary_loss_fn(y_star, _):
        # Surrogate "next-state" prediction loss on the equilibrium.
        # Real version: decode(y_star) → CE against targets.
        target = y_star.detach() * 0.92 + 0.01 * torch.randn_like(y_star)
        return F.mse_loss(y_star, target)

    sot_trainer = SOTSegmentedSolverTrainer(solver, primary_loss_fn, sot_cfg)

    # Tiny optimizer only for the solver + proposal (in real wiring this is the main optimizer)
    opt = torch.optim.AdamW(list(solver.parameters()) + list(proposal_proj.parameters()), lr=3e-4)

    # Synthetic "data" + slow context (in real run this comes from the dataset + BrainMimeticTripleMemory)
    dummy_x = torch.randn(B, T, D, device=device)
    slow_summary = torch.randn(B, D, device=device) * 0.3   # simulate triple memory summary

    log_lines = []
    start = time.time()

    for step in range(args.steps):
        opt.zero_grad()

        # === Proposal Engine produces y0 (rich context in real system) ===
        proposal = proposal_proj(dummy_x.mean(dim=1))   # [B, D] — simulated rich proposal
        proposal = proposal.unsqueeze(1).expand(-1, T, -1)  # make it sequence-like for solver

        slow_ctx = {"summary": slow_summary}

        # === Run one SOT segment on the Dedicated Solver (this is the new core loop) ===
        logs, sot_total, equilibrium = sot_trainer.train_segment(
            y0=proposal,
            slow_context=slow_ctx,
            target_logits_or_ids=None,
        )

        # === First-class internalization + primary on equilibrium ===
        solver_contrib = sot_total * args.attractor_solver_weight
        primary_on_eq = primary_loss_fn(equilibrium, None)

        total = primary_on_eq + solver_contrib
        total.backward()
        opt.step()

        # === Quick residual probe (the most important diagnostic signal) ===
        with torch.no_grad():
            _, meta = solver.solve(proposal.detach(), slow_ctx, max_steps=6, tol=5e-3)
            res = meta.get("final_residual", 1.0)

        # === Inference Densing Signal (Densing Law informed) ===
        # Primary signal: quality per solver step (lower is better density)
        densing_signal = float(primary_on_eq) / max(1, solver_res * 10 + 1e-8)   # proxy

        line = (f"step {step+1:02d} | "
                f"primary_eq={float(primary_on_eq):.5f} | "
                f"sot={float(sot_total):.5f} | "
                f"int={float(logs.get('sot_internalization_loss', 0.0)):.5f} | "
                f"solver_res={res:.4f} | "
                f"densing_sig={densing_signal:.5f} | "
                f"total={float(total):.5f}")

        log_lines.append(line)
        if (step + 1) % args.log_every == 0 or step < 3:
            print(line)

        if torch.isnan(total) or float(total) > 50:
            print("[CRITICAL] Loss explosion / NaN — hit one of the 7.1 failure modes early.")
            break

    elapsed = time.time() - start
    print(f"\n[COMPLETE] {args.steps} steps in {elapsed:.1f}s")
    print(f"Final solver residual: {res:.4f}")

    # Persist everything
    log_path = Path(args.out_dir) / "20step_diagnostic.log"
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"Log saved → {log_path}")

    ckpt = Path(args.out_dir) / "solver_20step.pt"
    torch.save({
        "solver": solver.state_dict(),
        "proposal_proj": proposal_proj.state_dict(),
        "steps": args.steps,
        "final_residual": res,
    }, ckpt)
    print(f"Solver checkpoint saved → {ckpt}")

    print("\n" + "=" * 72)
    print("IMMEDIATE NEXT ACTIONS (Section 7.1 Success Criteria + Densing Law)")
    print("=" * 72)
    print("1. Look at the log: is solver_res going down? Is int_loss going down? Is densing_sig improving?")
    print("2. If internalization flat after step 8-10 → rich proposal may already be near equilibrium (risk #1).")
    print("3. Re-run with different --sot_segment_length 3 or 7.")
    print("4. Re-run with --attractor_solver_weight 0.05 (gentle) or 0.25 (aggressive).")
    print("5. When the 20-step curve looks sane, integrate this loop into the real trainer for native 72 RI-1 measurement.")
    print("6. Document every deviation from the 7.1 risk list with numbers, especially Inference Densing curves.")
    print("7. Next architecture iteration targets: curriculum internalization, density-aware solver gating, procedural/factual separation (see wiki Section 7 'Specific Architectural Improvement Opportunities').")
    print("8. LoopMDM intersection experiment idea: Add --selective_loop and --stochastic_core_loops flags to test selective mid-block looping + stochastic loop count during SOT (inspired by arXiv:2605.26106). Track impact on densing_sig and training efficiency.")


if __name__ == "__main__":
    main()
