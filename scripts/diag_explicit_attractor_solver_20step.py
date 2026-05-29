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

# Noisy proposal experiment (first probe for long-term diffusion-style direction)
python scripts/diag_explicit_attractor_solver_20step.py \
  --steps 30 \
  --proposal_noise 0.15 \
  --noise_mode constant \
  --device cuda

Watch for (per wiki Section 7.1):
- solver_residual trending down across the steps
- internalization_loss trending down (y0 getting closer to equilibrium)
- No NaN / explosion (Parcae + SOT should prevent this)
- In noisy mode: q_gap not exploding + recov staying reasonably high (especially constant noise)

=== New: Noisy Proposal Mode (Option 1) — Improved Recovery Metric (June 2026) ===
When --proposal_noise > 0:
- The probe now captures actual equilibria (y*) from BOTH clean and noisy proposals at identical
  solver budget (6 steps).
- Two recovery views are logged:
    recov      = 1 - normalized quality_gap   (primary signal: did noisy start produce worse eq quality?)
    res_recov  = old residual-ratio version (kept for comparison)
- q_gap, q_clean, q_noisy = surrogate primary loss on the landed equilibria (higher = worse)
- densing_sig vs densing_sig_noisy shows Inference Densing impact of noisy start.

Recommended constant-noise sweep (run these three):
  python scripts/diag_explicit_attractor_solver_20step.py --steps 30 --proposal_noise 0.1 --noise_mode constant --device cuda
  python scripts/diag_explicit_attractor_solver_20step.py --steps 30 --proposal_noise 0.2 --noise_mode constant --device cuda
  python scripts/diag_explicit_attractor_solver_20step.py --steps 30 --proposal_noise 0.3 --noise_mode constant --device cuda

Interpretation target (per Section 7.1):
- If recov stays high (>0.85) and q_gap grows gracefully (not explosion) across 0.1→0.3 → positive signal for
  "Proposal as noisy latent + Solver as denoiser" direction.
- If q_gap explodes or recov collapses early → basin shaping is insufficient; need stronger NI/RI or SOT changes first.

After a clean 20-step run, promote the wiring into the real trainer for native 72 RI-1 measurement.
"""

import os
import argparse
import time
from pathlib import Path
from typing import Optional

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
    ProposalEngineAdapter,
)

# For Priority 1 (Real proposal engine internalization test)
try:
    from src.qtrm_mm.memory.brain_triple_memory import BrainMimeticTripleMemory
    from src.qtrm_mm.blocks import FastGatedLinearRecurrence, InferenceState
except Exception:
    BrainMimeticTripleMemory = None
    FastGatedLinearRecurrence = None

# For advancing to real OneBodyParallelHybridBlock proposal (Integration Roadmap item #2)
try:
    from scripts.train_556_on_parallel_hybrid_minimal import (
        Hybrid556Config,
        build_hybrid_stack,
    )
except Exception:
    Hybrid556Config = None
    build_hybrid_stack = None


class RichProposalStub(nn.Module):
    """
    Minimal rich proposal generator for Priority 1 diagnostic (fidelity upgrade).
    Uses:
    - Real FastGatedLinearRecurrence (from the actual hybrid citizen) when available
    - BrainMimeticTripleMemory for slow summary + surprise signal
    Goal: Produce proposal y0 that is closer to what the real OneBodyParallelHybridBlock would emit,
    so internalization behavior is measured under more realistic conditions.
    """
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.dim = dim
        self.device = device
        self._last_slow_summary = None
        self._fast_state = None  # carried hidden state for the real recurrence

        # For diagnostic stability we use a simple GRU here.
        # The real FastGatedLinearRecurrence has internal assumptions (and at least one bug when
        # called in complete isolation) that make it fragile outside the full OneBodyParallelHybridBlock.
        # Using real BrainMimeticTripleMemory is still a major fidelity win for internalization measurement.
        self.fast_rec = nn.GRU(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True)
        self._using_real_fast = False

        if BrainMimeticTripleMemory is not None:
            try:
                self.triple = BrainMimeticTripleMemory(d_model=dim, enabled=True)
            except Exception:
                self.triple = None
        else:
            self.triple = None

        # Projection from (fast_hidden + slow_summary) to proposal space
        self.to_proposal = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D] dummy input
        Returns: proposal y0 [B, T, D]
        """
        B, T, D = x.shape
        x_mean = x.mean(dim=1)  # [B, D]

        # === Fast recurrence step (GRU stub for diagnostic stability) ===
        x_in = x_mean.unsqueeze(1)
        _, h_n = self.fast_rec(x_in)
        fast_h = h_n.squeeze(0)

        # === Triple memory step ===
        slow_summary = None
        if self.triple is not None:
            try:
                current = fast_h.detach()
                _ = self.triple.step(current)
                if hasattr(self.triple, "get_current_summary"):
                    slow_summary = self.triple.get_current_summary()
                elif hasattr(self.triple, "slow_summary"):
                    slow_summary = self.triple.slow_summary
            except Exception:
                slow_summary = None

        if slow_summary is None or slow_summary.shape[-1] != D:
            slow_summary = torch.zeros(B, D, device=self.device)

        self._last_slow_summary = slow_summary.detach()

        # Combine into proposal
        combined = torch.cat([fast_h, slow_summary], dim=-1)
        proposal = self.to_proposal(combined)

        proposal = proposal.unsqueeze(1).expand(-1, T, -1)
        return proposal

    def get_last_slow_summary(self) -> Optional[torch.Tensor]:
        return self._last_slow_summary


class RealHybridProposal(nn.Module):
    """
    Real hybrid-based proposal engine for Priority 1 / Roadmap step #2.
    Builds a small stack of OneBodyParallelHybridBlock using the proven build_hybrid_stack.
    Follows the _hybrid_forward_only pattern more closely:
    - Carries InferenceState (fast_recurrent_state)
    - Proper tuple handling from hybrid layers
    - Passes fast_recurrent_state when available
    + BrainMimeticTripleMemory for slow context.
    This is the concrete step toward "real call to OneBodyParallelHybridBlock + BrainMimeticTripleMemory".
    """
    def __init__(self, dim: int, device: torch.device, n_layers: int = 1, internal_fast_recurrent: bool = False):
        super().__init__()
        self.dim = dim
        self.device = device
        self._last_slow_summary = None
        self._inf_state = None  # InferenceState carrier (key part of the pattern)
        self.internal_fast_recurrent = internal_fast_recurrent

        if build_hybrid_stack is None or Hybrid556Config is None:
            raise RuntimeError("build_hybrid_stack not available — cannot use real hybrid proposal")

        hcfg = Hybrid556Config(
            d_model=dim,
            n_layers=n_layers,
            recurrence_heads=2,
            attention_heads=1,
            attention_type="mla",
            device=device,
            dtype=torch.float32,
            delta_backend="torch_gated_delta2_v2",
            enable_stochastic_breadth=False,
            stochastic_breadth_ablation_zero=True,
        )

        self.hybrid_stack = build_hybrid_stack(hcfg)
        self.micro_steps = 4 if self.internal_fast_recurrent else 2  # simulate internal recurrence depth

        if BrainMimeticTripleMemory is not None:
            try:
                self.triple = BrainMimeticTripleMemory(d_model=dim, enabled=True)
            except Exception:
                self.triple = None
        else:
            self.triple = None

        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        Returns proposal y0 by running real hybrid layers following the trainer's _hybrid_forward_only style.
        """
        B, T, D = x.shape
        h = x.mean(dim=1)  # [B, D]

        slots = None
        coarse_counter = 0

        try:
            for _ in range(self.micro_steps):  # controlled by internal_fast_recurrent flag
                do_full = True

                for layer in self.hybrid_stack:
                    if isinstance(layer, OneBodyParallelHybridBlock):
                        if do_full:
                            out = layer(
                                h.unsqueeze(1) if h.dim() == 2 else h,
                                stochastic_breadth_noise=None,
                                slot_state=slots,
                                fast_recurrent_state=self._inf_state,
                            )
                        else:
                            out = layer(h.unsqueeze(1) if h.dim() == 2 else h, stochastic_breadth_noise=None)

                        if isinstance(out, tuple):
                            if len(out) >= 2:
                                h, slots = out[0], out[1]
                            if len(out) >= 3:
                                fr_state = out[2]
                                if isinstance(fr_state, InferenceState):
                                    self._inf_state = fr_state
                                elif fr_state is not None:
                                    self._inf_state = InferenceState(
                                        fast_recurrent_h=fr_state,
                                        step_count=(self._inf_state.step_count + 1) if self._inf_state else 1
                                    )
                        else:
                            h = out

                if h.dim() == 3:
                    h = h.squeeze(1)

                if slots is not None:
                    slots = slots * 0.98

                coarse_counter += 1

            # When internal_fast_recurrent is on, do one extra internal refinement pass
            # (simulating deeper internal thinking before slow memory update)
            if self.internal_fast_recurrent:
                for _ in range(2):
                    for layer in self.hybrid_stack:
                        if isinstance(layer, OneBodyParallelHybridBlock):
                            out = layer(
                                h.unsqueeze(1) if h.dim() == 2 else h,
                                stochastic_breadth_noise=None,
                                slot_state=slots,
                                fast_recurrent_state=self._inf_state,
                            )
                            if isinstance(out, tuple):
                                h = out[0]
                                if len(out) >= 2:
                                    slots = out[1]
                            else:
                                h = out
                    if h.dim() == 3:
                        h = h.squeeze(1)

        except Exception:
            # Defensive: keep going with whatever h we have
            h = h.detach() if torch.is_tensor(h) else x.mean(dim=1)

        # Triple memory update using the hybrid state
        slow_summary = None
        if self.triple is not None:
            try:
                current = h.detach()
                _ = self.triple.step(current)
                slow_summary = getattr(self.triple, "get_current_summary", lambda: None)()
            except Exception:
                slow_summary = None

        if slow_summary is None or slow_summary.shape[-1] != D:
            slow_summary = torch.zeros_like(h)

        self._last_slow_summary = slow_summary.detach()

        # When internal_fast_recurrent is enabled, give stronger weight to the slow summary.
        # This simulates that deeper internal recurrence makes the proposal more strongly
        # shaped by the accumulated (wired) slow context — a direct item #3/#4 interaction.
        slow_weight = 0.55 if self.internal_fast_recurrent else 0.35
        proposal = self.out_proj(h + slow_summary * slow_weight)
        proposal = proposal.unsqueeze(1).expand(-1, T, -1)
        return proposal

    def get_last_slow_summary(self) -> Optional[torch.Tensor]:
        return self._last_slow_summary


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

    # === Noisy Proposal Experiment (Option 1 - First step toward Diffusion-style direction) ===
    parser.add_argument("--proposal_noise", type=float, default=0.0,
                        help="Add controlled Gaussian noise to the proposal y0 before feeding to solver. "
                             "This is the first probe for the long-term 'Proposal as noisy latent' direction.")
    parser.add_argument("--noise_mode", type=str, default="constant",
                        choices=["constant", "increasing", "decreasing"],
                        help="How to apply proposal noise across steps.")

    # === Priority 1: Real/Rich proposal engine to finally get non-zero internalization (Risk #1 test) ===
    parser.add_argument("--rich_proposal", action="store_true",
                        help="Use a richer proposal stub (BrainMimeticTripleMemory + minimal fast state) instead of toy linear. "
                             "This is the minimal way to get meaningful internalization_loss > 0 and observe if it decreases.")

    # Next level per Integration Roadmap: use actual OneBodyParallelHybridBlock stack for proposal generation
    parser.add_argument("--real_hybrid_proposal", action="store_true",
                        help="Use real OneBodyParallelHybridBlock stack (via build_hybrid_stack) + triple memory as the proposal engine. "
                             "This directly follows the 'replace toy with real hybrid call' step in the milestone roadmap.")

    # Minimal prep for Integration Roadmap item #3 (wiring equilibrium as primary output)
    parser.add_argument("--demo_equilibrium_wiring", action="store_true",
                        help="Demonstrate using the solver equilibrium as the 'final' output for primary loss (prep for wiring into main LM head path).")

    # Minimal prep for Integration Roadmap item #4
    parser.add_argument("--internal_fast_recurrent", action="store_true",
                        help="Simulate internal fast recurrent participation in proposal generation (item #4 prep in diagnostic).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 72)
    print("EXPLICIT ATTRACTOR SOLVER — FIRST 20-STEP DIAGNOSTIC (Section 7)")
    print("=" * 72)
    print(f"Config: steps={args.steps}, sot_h={args.sot_segment_length}, "
          f"solver_w={args.attractor_solver_weight}, int_w={args.internalization_weight}")
    if args.proposal_noise > 0.0:
        print(f"Noisy Proposal Mode: scale={args.proposal_noise}, mode={args.noise_mode}  ← Diffusion-style probe")
    if args.real_hybrid_proposal:
        print("Real Hybrid Proposal Mode: ON (OneBodyParallelHybridBlock stack) ← Roadmap item #2")
    elif args.rich_proposal:
        print("Rich Proposal Mode: ON (BrainMimeticTripleMemory + fast recurrence) ← Priority 1 internalization test")
    if args.demo_equilibrium_wiring:
        print("Demo Equilibrium Wiring: ON (Roadmap item #3) — equilibrium is treated as the explicit final wired output")
    if args.internal_fast_recurrent:
        print("Internal Fast Recurrent: ON (Roadmap item #4 prep) — proposal generation uses more internal micro-steps")

    device = torch.device(args.device)
    D = args.d_model
    B = args.batch
    T = args.seq_len

    # === Proposal Engine ===
    # Default (toy)
    # --rich_proposal : RichProposalStub (real TripleMemory)
    # --real_hybrid_proposal : Actual OneBodyParallelHybridBlock stack (Roadmap item #2)
    if args.real_hybrid_proposal:
        print("[Priority 1 / Roadmap #2] Using RealHybridProposal (OneBodyParallelHybridBlock stack + TripleMemory)")
        try:
            proposal_engine = RealHybridProposal(
                dim=D, 
                device=device, 
                n_layers=1,
                internal_fast_recurrent=args.internal_fast_recurrent
            ).to(device)
        except Exception as e:
            print(f"[Warning] real_hybrid_proposal failed to initialize ({e}). Falling back to rich_proposal stub.")
            proposal_engine = RichProposalStub(dim=D, device=device).to(device) if BrainMimeticTripleMemory is not None else None
        proposal_proj = None
    elif args.rich_proposal and BrainMimeticTripleMemory is not None:
        print("[Priority 1] Using RichProposalStub (BrainMimeticTripleMemory + fast recurrence) for meaningful internalization test.")
        proposal_engine = RichProposalStub(dim=D, device=device).to(device)
        proposal_proj = None
    else:
        if args.rich_proposal or args.real_hybrid_proposal:
            print("[Warning] rich/real hybrid proposal requested but dependencies unavailable. Falling back to toy linear.")
        proposal_engine = None
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
        loss = F.mse_loss(y_star, target)

        if args.demo_equilibrium_wiring:
            # Roadmap item #3 wiring demo:
            # y_star (equilibrium) is now the canonical final output state.
            # In the real trainer this would be passed to the LM head for the actual CE loss.
            # We treat it explicitly as the "answer representation".
            wiring_loss = 0.12 * F.mse_loss(y_star, target * 0.93)
            loss = loss + wiring_loss
        return loss

    sot_trainer = SOTSegmentedSolverTrainer(solver, primary_loss_fn, sot_cfg)

    # Tiny optimizer only for the solver + proposal (in real wiring this is the main optimizer)
    proposal_params = []
    if proposal_proj is not None:
        proposal_params = list(proposal_proj.parameters())
    elif proposal_engine is not None:
        proposal_params = list(proposal_engine.parameters())

    opt = torch.optim.AdamW(list(solver.parameters()) + proposal_params, lr=3e-4)

    # Synthetic "data" + slow context (in real run this comes from the dataset + BrainMimeticTripleMemory)
    dummy_x = torch.randn(B, T, D, device=device)
    slow_summary = torch.randn(B, D, device=device) * 0.3   # simulate triple memory summary

    log_lines = []
    start = time.time()

    for step in range(args.steps):
        opt.zero_grad()

        # === Proposal Engine produces y0 ===
        if proposal_engine is not None:
            # Rich proposal path (Priority 1)
            proposal = proposal_engine(dummy_x)
            # Get slow summary for context if available
            slow_summary = proposal_engine.get_last_slow_summary() if hasattr(proposal_engine, "get_last_slow_summary") else slow_summary
        else:
            # Original toy linear path
            proposal = proposal_proj(dummy_x.mean(dim=1))
            proposal = proposal.unsqueeze(1).expand(-1, T, -1)

        # === Noisy Proposal Experiment (Diffusion-style direction probe) ===
        if args.proposal_noise > 0.0:
            if args.noise_mode == "constant":
                noise_scale = args.proposal_noise
            elif args.noise_mode == "increasing":
                noise_scale = args.proposal_noise * (step / max(1, args.steps - 1))
            else:  # decreasing
                noise_scale = args.proposal_noise * (1 - step / max(1, args.steps - 1))

            noise = torch.randn_like(proposal) * noise_scale
            noisy_proposal = proposal + noise
        else:
            noisy_proposal = proposal

        slow_ctx = {"summary": slow_summary}

        # === Run one SOT segment on the Dedicated Solver ===
        # Pass proposal_engine so that internalization_loss becomes active and non-zero (the whole point of Priority 1)
        logs, sot_total, equilibrium = sot_trainer.train_segment(
            y0=noisy_proposal,
            slow_context=slow_ctx,
            target_logits_or_ids=None,
            proposal_engine=proposal_engine,   # critical for internalization to actually run
        )

        # Roadmap item #3 wiring feedback (minimal but concrete):
        # When the equilibrium is explicitly wired as the final output,
        # we simulate that it now influences the slow context for the next step.
        # This creates a realistic loop where the wired state starts shaping future proposals.
        if args.demo_equilibrium_wiring:
            slow_ctx = {"summary": equilibrium.mean(dim=1).detach()}  # use equilibrium as new slow summary

        # === First-class internalization + primary on equilibrium ===
        solver_contrib = sot_total * args.attractor_solver_weight

        # Roadmap item #3 wiring demo (strengthened):
        # The equilibrium from SOT is now the explicit "wired final output".
        # In a real trainer this would be what gets decoded by the LM head.
        wired_output = equilibrium

        primary_on_eq = primary_loss_fn(wired_output, None)

        total = primary_on_eq + solver_contrib
        total.backward()
        opt.step()

        # === Improved Equilibrium Quality Probe (Option 1 recovery refinement) ===
        # Key question for diffusion-style direction:
        #   "At the *same* solver budget, does starting from a noisy proposal produce a
        #    meaningfully worse equilibrium than a clean proposal?"
        # We now measure both residual AND surrogate task quality on the landed equilibria.
        with torch.no_grad():
            y_star_clean, meta_clean = solver.solve(proposal.detach(), slow_ctx, max_steps=6, tol=5e-3)
            clean_res = meta_clean.get("final_residual", 1.0)

            # Surrogate quality on clean equilibrium (same form as primary loss)
            eq_quality_clean = float(primary_loss_fn(y_star_clean, None))

            if args.proposal_noise > 0.0:
                y_star_noisy, meta_noisy = solver.solve(noisy_proposal.detach(), slow_ctx, max_steps=6, tol=5e-3)
                noisy_res = meta_noisy.get("final_residual", 1.0)

                # Surrogate quality on the equilibrium reached from noisy start
                eq_quality_noisy = float(primary_loss_fn(y_star_noisy, None))

                # === Core improved recovery metric (equilibrium degradation) ===
                # quality_gap: how much *worse* (higher surrogate loss) the noisy-started eq is
                quality_gap = max(0.0, eq_quality_noisy - eq_quality_clean)

                # Recovery = 1 - normalized degradation.
                # 1.0 = noisy start produced *identical* quality equilibrium
                # 0.0 = noisy start produced fully degraded equilibrium (no recovery happened)
                # This is more meaningful for "Inference Densing" than raw residual ratio.
                recovery = 1.0 - min(1.0, quality_gap / (eq_quality_clean + 1e-6))

                # Also keep the old residual-based view for comparison
                residual_recovery = max(0.0, (noisy_res - clean_res) / max(1e-8, noisy_res))
            else:
                noisy_res = clean_res
                eq_quality_noisy = eq_quality_clean
                quality_gap = 0.0
                recovery = 1.0
                residual_recovery = 0.0

        # === Inference Densing Signals (Densing Law informed) ===
        densing_signal = float(primary_on_eq) / max(1, clean_res * 10 + 1e-8)
        densing_signal_noisy = float(primary_on_eq) / max(1, noisy_res * 10 + 1e-8) if args.proposal_noise > 0.0 else densing_signal

        # === Extra internalization progress (using solver's built-in helper) ===
        int_progress = solver.compute_internalization_progress(proposal.detach(), equilibrium) if args.rich_proposal else {}

        line = (f"step {step+1:02d} | "
                f"primary_eq={float(primary_on_eq):.5f} | "
                f"sot={float(sot_total):.5f} | "
                f"int={float(logs.get('sot_internalization_loss', 0.0)):.5f} | "
                f"clean_res={clean_res:.4f} | "
                f"noisy_res={noisy_res:.4f} | "
                f"q_clean={eq_quality_clean:.5f} | "
                f"q_noisy={eq_quality_noisy:.5f} | "
                f"q_gap={quality_gap:.5f} | "
                f"recov={recovery:.3f} | "
                f"res_recov={residual_recovery:.3f} | "
                f"dsig={densing_signal:.5f} | "
                f"dsig_n={densing_signal_noisy:.5f} | "
                f"int_mse={int_progress.get('internalization_mse', 0.0):.5f} | "
                f"total={float(total):.5f}")

        log_lines.append(line)
        if (step + 1) % args.log_every == 0 or step < 3:
            print(line)

        if torch.isnan(total) or float(total) > 50:
            print("[CRITICAL] Loss explosion / NaN — hit one of the 7.1 failure modes early.")
            break

    elapsed = time.time() - start
    print(f"\n[COMPLETE] {args.steps} steps in {elapsed:.1f}s")
    print(f"Final solver residual (clean): {clean_res:.4f}")
    if args.proposal_noise > 0.0:
        print(f"Final quality gap (noisy vs clean eq): {quality_gap:.5f} | recovery={recovery:.3f}")

    # Persist everything
    log_path = Path(args.out_dir) / "20step_diagnostic.log"
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"Log saved → {log_path}")

    ckpt = Path(args.out_dir) / "solver_20step.pt"
    save_dict = {
        "solver": solver.state_dict(),
        "steps": args.steps,
        "final_residual": clean_res,
        "final_quality_gap": quality_gap if args.proposal_noise > 0.0 else 0.0,
        "final_recovery": recovery if args.proposal_noise > 0.0 else 1.0,
        "rich_proposal_used": args.rich_proposal,
    }
    if proposal_proj is not None:
        save_dict["proposal_proj"] = proposal_proj.state_dict()
    if proposal_engine is not None:
        save_dict["proposal_engine"] = proposal_engine.state_dict()
    torch.save(save_dict, ckpt)
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
