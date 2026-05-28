"""
Dedicated Attractor Solver Module + Supporting Primitives (June 2026 Substrate Overhaul Prototype)

Core papers driving this module (see wiki Section 7 + Appendices F/G):
- "Solve the Loop: Attractor Models for Language and Reasoning" (Fein-Ashley & Rashidinejad, arXiv:2605.12466)
  - Backbone (Proposal Engine) produces ỹ₀ in tied output embedding space.
  - Small attractor module solves ỹ* = T_θa(ỹ, ỹ₀) (or root of A(y, ỹ₀)=0) via root-finding (Anderson or simple iteration).
  - Persistent proposal injection at EVERY solver step.
  - Equilibrium internalization: during training ||ỹ₀ - ỹ*|| drives the proposal engine to do most of the work.
  - Training: next-token on equilibrium + implicit differentiation (or one-step phantom) → O(1) memory w.r.t. effective depth.
  - Inference: adaptive depth via residual tolerance ε.

- Parcae (Prairie et al., arXiv:2604.12946)
  - Looped recurrence as nonlinear time-variant dynamical system over residual:
      h_{t+1} = A_bar h_t + B_bar e + R_bar(h_t, e)
  - Instability root cause: spectral norm of injection matrix A_bar >= 1.
  - Fix: parameterize continuous A as negative diagonal Diag(-exp(log_A)), discretize with ZOH/Euler.
      → eigenvalues strictly negative → ρ(A_bar) < 1 guaranteed.
  - Additional: prelude/input normalization + per-sequence stochastic depth sampling.

- EqR / Equilibrium Reasoners (Huang, Geng, Kolter, arXiv:2605.21488)
  - Task-conditioned attractors: stable fixed points of z_{k+1} = f_θ(z_k; x) = solutions.
  - Depth + Breadth scaling axes (NFE ≈ D × B). Residual ||f(z)-z|| is strong correctness proxy.
  - SOT (Segmented Online Training): split long trajectory into short segments (h≈7), local loss + immediate optimizer step + detached carry.
    This interleaves parameter updates with latent dynamics → far better attractor landscape shaping than terminal-loss full unroll.
  - Training interventions: RI (randomized init of solver state) + NI (noise injection) to broaden correct basins and suppress spurious ones.

Integration contract with existing codebase:
- ProposalEngineAdapter wraps the current OneBodyParallelHybridBlock + FastGatedLinearRecurrence + BrainMimeticTripleMemory
  and returns (y0_proposal, slow_context_dict) where slow_context includes summary, surprise, provenance state.
- AttractorSolverModule is a small weight-tied network (can be 1-2 hybrid blocks or simplified).
  Its .step(y, y0, slow_ctx) MUST inject y0 persistently (concat/projection/add every call).
- The final equilibrium y* is decoded to logits (tied unembed) and used for primary CE loss.
- Slow memory updates (ChunkedSlow + TripleMemory) happen primarily from the *final equilibrium*, not intermediate states.
- SOTSegmentedSolverTrainer provides the training loop skeleton that the main trainer can call for segments.
- All components are designed to be dropped into the primary one-body forward path (no side channels).

Ablation flags (to be wired in config/trainer):
  --attractor_solver_off (or solver_weight=0.0) → bypass, fall back to proposal only (baseline behavior)
  --internalization_weight=0.0
  --parcae_negative_diag_off
  --sot_off (use vanilla full unroll or current external rehearsal)
  --ri_ni_off (no randomized init / noise in solver)

This is a *prototype skeleton*. Forward + loss + SOT loop are implemented at sufficient fidelity to run unit tests
and to be wired into train_hybrid_ri4... scripts for small-scale diagnostic runs (B=2~4, short segments).
Full Anderson acceleration, implicit_function_theorem autograd, and production one-body integration are TODOs
marked with clear comments.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Project imports (graceful fallback for prototype isolation)
try:
    from ..blocks import InferenceState, FastGatedLinearRecurrence
    from ..memory.brain_triple_memory import BrainMimeticTripleMemory
    from ..config import QTRMConfig
except Exception:
    InferenceState = None
    FastGatedLinearRecurrence = None
    BrainMimeticTripleMemory = None
    QTRMConfig = None


# =============================================================================
# 1. Parcae-style Negative Diagonal Stability Operator (core of solver recurrence)
# =============================================================================

class ParcaeNegativeDiagonalInjection(nn.Module):
    """
    Implements the stable injection operator from Parcae (arXiv:2604.12946).

    Continuous form: A := Diag(-exp(log_A))   (strictly negative eigenvalues)
    Discretization:  A_bar = ZOH(Δ * A) or Euler
    B_bar via Euler or learned.

    In the solver recurrence:
        y_{t+1} = A_bar @ y_t + B_bar @ (proposal_injection + slow_context_proj) + R(y_t, ...)
    The negative diagonal + proper Δ keeps ρ(A_bar) < 1 even at high iteration counts.
    """

    def __init__(self, dim: int, use_zoh: bool = True, learn_delta: bool = True):
        super().__init__()
        self.dim = dim
        self.use_zoh = use_zoh
        # log_A > -inf; we store raw parameter and exp inside
        self.log_A = nn.Parameter(torch.zeros(dim))          # init → A = -1 on diag (after exp)
        self.B = nn.Linear(dim, dim, bias=False)             # B_bar will be derived or learned projection
        if learn_delta:
            self.log_delta = nn.Parameter(torch.zeros(dim))  # per-channel learned step size
        else:
            self.register_buffer("log_delta", torch.zeros(dim))
        self.input_norm = nn.LayerNorm(dim)  # Parcae prelude/input norm (critical for late-stage stability)

    def forward(self, h: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        """
        h: [B, T, D] or [B, D] recurrent state
        injection: proposal + slow context projected signal (same shape)
        Returns the linear contribution A_bar @ h + B_bar @ normalized_injection
        """
        x = self.input_norm(injection)
        delta = torch.exp(self.log_delta) + 1e-8
        A_cont = -torch.exp(self.log_A)                     # [D] strictly negative
        if self.use_zoh:
            # ZOH discretization (approx for diagonal: elementwise)
            A_bar = torch.exp(delta * A_cont)               # still <1 in magnitude
        else:
            # Euler
            A_bar = 1.0 + delta * A_cont
            A_bar = torch.clamp(A_bar, max=0.999)           # safety

        A_bar = A_bar.unsqueeze(0).unsqueeze(0) if h.dim() > 2 else A_bar.unsqueeze(0)
        # Linear part
        lin = A_bar * h
        # B contribution (learned + normalized injection)
        b_contrib = self.B(x)
        return lin + b_contrib


# =============================================================================
# 2. AttractorSolverModule (Dedicated Solver with Persistent Proposal Injection)
# =============================================================================

class AttractorSolverModule(nn.Module):
    """
    The new "heavy iterative refinement" engine.

    Signature (per Solve-the-Loop + Parcae + EqR):
        y_{t+1} = f_θ(y_t, y0, slow_context)   with y0 injected *every single step*

    The module is intentionally small (1-2 blocks or even a single learned operator + Parcae injection).
    It is weight-tied across solver steps by design.

    At training time we typically run SOT (short segments) rather than one giant unroll.
    At inference we run until residual < ε (adaptive depth).
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 1,
        use_parcae: bool = True,
        use_hyperconnections: bool = False,  # stub for future φ recurrence boost
        max_solver_steps: int = 32,
        residual_tol: float = 1e-3,
    ):
        super().__init__()
        self.dim = dim
        self.max_solver_steps = max_solver_steps
        self.residual_tol = residual_tol
        self.use_parcae = use_parcae

        # Small recurrent operator (can be upgraded to 1-2 OneBodyHybridBlocks later)
        self.parcae_inject = ParcaeNegativeDiagonalInjection(dim) if use_parcae else None

        # Core refinement operator R (the "T_θa" of the paper). Minimal MLP + gated residual for prototype.
        self.refine = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.gate = nn.Linear(dim * 2, dim)  # for gated residual (helps stability)

        # Projection for slow context injection (BrainMimeticTripleMemory summary etc.)
        self.slow_proj = nn.Linear(dim, dim, bias=False)

        # Persistent proposal injection projection (critical per Attractor Models paper)
        self.proposal_inject = nn.Linear(dim, dim, bias=False)

        # Optional: learned halting / adaptive depth head (EqR style)
        self.halt_head = nn.Linear(dim, 1)

        self._step_count = 0  # for logging / diagnostics

    def step(
        self,
        y: torch.Tensor,
        y0: torch.Tensor,
        slow_context: Optional[Dict[str, torch.Tensor]] = None,
        noise: float = 0.0,
    ) -> Tuple[torch.Tensor, float]:
        """
        One solver step with *mandatory persistent proposal injection*.

        Returns:
            y_next, residual_norm
        """
        B, T, D = y.shape if y.dim() == 3 else (y.shape[0], 1, y.shape[-1])
        y = y if y.dim() == 3 else y.unsqueeze(1)

        # === Persistent proposal injection (every step) ===
        prop_inj = self.proposal_inject(y0)
        if prop_inj.dim() == 2:
            prop_inj = prop_inj.unsqueeze(1).expand_as(y)

        # Slow context (optional rich state from triple memory)
        slow_inj = 0.0
        if slow_context is not None and "summary" in slow_context:
            s = slow_context["summary"]
            if s.dim() == 2:
                s = s.unsqueeze(1)
            slow_inj = self.slow_proj(s)

        # Combine signals
        combined_inj = prop_inj + slow_inj

        # Parcae stable linear dynamics (if enabled)
        if self.use_parcae and self.parcae_inject is not None:
            lin = self.parcae_inject(y, combined_inj)
        else:
            lin = y * 0.9 + combined_inj * 0.1   # fallback decay

        # Non-linear refinement (the actual attractor dynamics)
        r = self.refine(y + lin)
        gate = torch.sigmoid(self.gate(torch.cat([y, r], dim=-1)))
        y_next = y + gate * (r - y)

        # Optional noise injection (EqR NI for basin shaping)
        if noise > 0 and self.training:
            y_next = y_next + torch.randn_like(y_next) * noise

        residual = (y_next - y).pow(2).mean().sqrt().item()

        # restore shape
        if y.dim() == 3 and y_next.dim() == 3 and T == 1:
            y_next = y_next.squeeze(1)

        return y_next, residual

    @torch.no_grad()
    def solve(
        self,
        y0: torch.Tensor,
        slow_context: Optional[Dict[str, torch.Tensor]] = None,
        max_steps: Optional[int] = None,
        tol: Optional[float] = None,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Full adaptive-depth solve at inference (or for diagnostics).
        Returns equilibrium y* and metadata (steps_taken, final_residual, trajectory if requested).
        """
        max_steps = max_steps or self.max_solver_steps
        tol = tol or self.residual_tol
        y = y0.clone()
        trajectory = [] if return_trajectory else None
        residuals = []

        for t in range(max_steps):
            y, res = self.step(y, y0, slow_context, noise=0.0)
            residuals.append(res)
            if return_trajectory:
                trajectory.append(y.clone())
            if res < tol:
                break

        meta = {
            "steps_taken": t + 1,
            "final_residual": residuals[-1] if residuals else 1.0,
            "residuals": residuals,
        }
        if return_trajectory:
            meta["trajectory"] = trajectory
        return y, meta

    def forward(
        self,
        y0: torch.Tensor,
        slow_context: Optional[Dict[str, torch.Tensor]] = None,
        num_steps: int = 8,
        noise: float = 0.0,
    ) -> torch.Tensor:
        """
        Training-friendly fixed-step forward (use SOT wrapper for long horizons).
        """
        y = y0
        for _ in range(num_steps):
            y, _ = self.step(y, y0, slow_context, noise=noise)
        return y

    def compute_internalization_progress(self, y0: torch.Tensor, y_star: torch.Tensor) -> Dict[str, float]:
        """
        Densing Law informed helper (Inference Densing).
        Returns simple metrics on how close the proposal is to equilibrium.
        Can be used to drive curriculum or density-aware losses.
        """
        mse = F.mse_loss(y0, y_star.detach()).item()
        cos = F.cosine_similarity(y0.flatten(), y_star.detach().flatten(), dim=0).item() if y0.numel() > 0 else 0.0
        return {
            "internalization_mse": mse,
            "internalization_cosine": cos,
            "internalization_gap": mse,  # primary signal for curriculum decay of solver budget
        }

    def get_densing_metrics(self, trajectory: Optional[list] = None, quality_proxy: Optional[float] = None) -> Dict[str, Any]:
        """
        Experimental hook for Inference Densing tracking.
        In real usage this would combine solver steps, residual curve, and downstream quality.
        """
        return {
            "steps_in_last_solve": getattr(self, '_last_steps', None),
            "final_residual": getattr(self, '_last_residual', None),
            "quality_proxy": quality_proxy,
            # TODO: integrate with trainer to compute quality_per_solver_step
        }

    # -------------------------------------------------------------------------
    # LoopMDM-style Selective Looping Extension (Intersection Experiment)
    # -------------------------------------------------------------------------
    # Idea: Instead of treating the entire solver as the looped unit,
    # designate a small "core_refinement_block" that is looped selectively.
    # This mirrors LoopMDM's head/mid/tail + selective mid-block looping.
    #
    # Potential benefits (Densing Law + LoopMDM):
    # - Much lower cost per solver step
    # - Still strong refinement power on hard cases
    # - Natural fit with adaptive early stopping at inference
    #
    # TODO for next iteration:
    # - Add `core_refinement_block` (small Parcae-stabilized sub-module)
    # - Modify step() to have option `selective_loop=True`
    # - In SOT trainer, support stochastic number of core loops (LoopMDM training trick)
    # - Add adaptive stopping based on delta inside the core loop

    def step_selective(self, y: torch.Tensor, y0: torch.Tensor, slow_context=None, num_core_loops: int = 1, noise: float = 0.0):
        """
        Experimental selective-loop step (LoopMDM intersection).
        Applies the main refinement logic only `num_core_loops` times on a cheap core,
        then one final full step. Placeholder for future implementation.
        """
        # Placeholder: currently just calls normal step multiple times.
        # Real version would have a lighter "core" sub-network for the inner loops.
        for _ in range(num_core_loops):
            y, _ = self.step(y, y0, slow_context, noise=noise)
        return self.step(y, y0, slow_context, noise=0.0)  # final full step

    # -------------------------------------------------------------------------
    # Long-term: Diffusion-Style Noisy Attractor Iteration
    # "Proposal as initial noisy latent → Solver as learned denoiser"
    # -------------------------------------------------------------------------
    # This direction reframes solver steps as iterative denoising of a structured
    # "noisy proposal" toward high-quality equilibria.
    #
    # Key concepts to explore:
    # - Explicit noise schedule (beta_t, alpha_t) during training and inference
    # - Noising the proposal y0 with learnable or scheduled noise
    # - Solver becomes a conditional denoiser: f(y_t, y0_clean, t, slow_context)
    # - Training losses that include denoising objectives at multiple noise levels
    #
    # Potential benefits:
    # - More powerful basin shaping via controlled noise
    # - Finer control of "thinking compute" at inference (true Inference Densing)
    # - Natural connection to diffusion LM + recurrent depth literature

    def step_with_noise_schedule(
        self,
        y: torch.Tensor,
        y0: torch.Tensor,
        slow_context: Optional[Dict] = None,
        t: int = 0,
        total_steps: int = 10,
        schedule: str = "cosine",   # "linear", "cosine", "learned"
        noise_scale: float = 0.1,
    ):
        """
        Experimental diffusion-style step with explicit timestep and schedule.
        This is a long-term research stub.
        """
        # Placeholder implementation
        # Real version would:
        # 1. Compute beta_t or alpha_t based on schedule and t
        # 2. Optionally add scheduled noise to y
        # 3. Condition the refinement on the current noise level t
        # 4. Return both the refined state and predicted noise (for denoising loss)

        # For now, just delegate to normal step with extra noise modulated by t
        effective_noise = noise_scale * (1 - t / max(1, total_steps))
        y_next, res = self.step(y, y0, slow_context, noise=effective_noise)
        return y_next, res, {"t": t, "effective_noise": effective_noise}


# =============================================================================
# 3. ProposalEngineAdapter (repurposes existing hybrid citizen as rich y0 generator)
# =============================================================================

class ProposalEngineAdapter(nn.Module):
    """
    Thin wrapper that treats the current best hybrid citizen
    (OneBodyParallelHybridBlock + FastGated + BrainMimeticTripleMemory + ChunkedSlow)
    as a high-quality proposal generator (ỹ₀ + rich slow context).

    The real "thinking" work moves to the AttractorSolverModule.
    This adapter's job is only to produce a semantically meaningful starting point
    + the slow memory state that the solver will condition on.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model  # the existing hybrid / qtrm_model

    def forward(
        self,
        input_emb: torch.Tensor,
        inference_state: Optional["InferenceState"] = None,
        slow_memory: Optional["BrainMimeticTripleMemory"] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Returns:
            y0_proposal: [B, T, D] initial guess in output embedding space (tied)
            slow_context: dict with summary, surprise, provenance, etc.
        """
        # In real wiring this would call the internal fast citizen forward
        # and extract the pre-logit representation + current slow summary.
        # For prototype we return plausible shapes and a stub context.
        B, T, D = input_emb.shape
        y0 = input_emb  # placeholder — real impl would run the fast path + projection to output emb

        slow_ctx = {
            "summary": None,
            "surprise": None,
            "provenance": None,
        }
        if slow_memory is not None:
            # In real code: slow_memory.get_current_summary() etc.
            try:
                state = slow_memory.get_state() if hasattr(slow_memory, "get_state") else None
                if state is not None:
                    slow_ctx["summary"] = state.get("slow_summary") if isinstance(state, dict) else None
            except Exception:
                pass

        return y0, slow_ctx


# =============================================================================
# 4. Equilibrium Internalization Loss (first-class objective)
# =============================================================================

class EquilibriumInternalizationLoss(nn.Module):
    """
    L_int = || y0 - stopgrad(y*) ||^2   (or cosine, Huber, etc.)

    This is the mechanism that causes "equilibrium internalization".
    As training progresses the proposal engine learns to emit y0 already very close
    to the attractor the solver would find → solver becomes skippable at inference.
    """

    def __init__(self, weight: float = 0.1, distance: str = "mse"):
        super().__init__()
        self.weight = weight
        self.distance = distance

    def forward(
        self,
        y0: torch.Tensor,
        y_star: torch.Tensor,
    ) -> torch.Tensor:
        if self.distance == "cosine":
            y0_n = F.normalize(y0, dim=-1)
            ys_n = F.normalize(y_star.detach(), dim=-1)
            loss = (1.0 - (y0_n * ys_n).sum(-1)).mean()
        else:
            loss = F.mse_loss(y0, y_star.detach())
        return self.weight * loss


# =============================================================================
# 5. SOT Segmented Solver Trainer (EqR core training primitive)
# =============================================================================

@dataclass
class SOTConfig:
    segment_length: int = 7          # h in the EqR paper
    max_segments: int = 8
    ri_noise: float = 0.05           # randomized init + noise injection scale
    internalization_weight: float = 0.15
    use_detached_carry: bool = True


class SOTSegmentedSolverTrainer:
    """
    Executes the Segmented Online Training loop for the attractor solver.

    High-level:
        for each segment:
            run solver for segment_length steps (starting from previous equilibrium or RI)
            compute primary loss on current y_segment_end (next-token or task loss)
            compute internalization loss (proposal vs current equilibrium)
            optimizer.step() immediately
            detach carry for next segment (truncates gradient, keeps memory bounded)

    This is dramatically more stable and landscape-shaping than "unroll 64 steps then one big backward".
    """

    def __init__(self, solver: AttractorSolverModule, loss_fn: Callable, cfg: SOTConfig):
        self.solver = solver
        self.loss_fn = loss_fn   # usually CE on decoded logits
        self.cfg = cfg
        self.internalization = EquilibriumInternalizationLoss(
            weight=cfg.internalization_weight
        )

    def train_segment(
        self,
        y0: torch.Tensor,
        slow_context: Dict[str, torch.Tensor],
        target_logits_or_ids: Any,
        proposal_engine: Optional[ProposalEngineAdapter] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run one SOT segment. Returns dict of losses for logging.
        """
        # RI: randomized initialization of solver state (EqR)
        y = y0.clone()
        if self.cfg.ri_noise > 0:
            y = y + torch.randn_like(y) * self.cfg.ri_noise

        # Run the solver segment (fixed steps for this prototype; adaptive later)
        y_star = self.solver(y, slow_context, num_steps=self.cfg.segment_length)

        # Primary task loss (on the equilibrium)
        primary_loss = self.loss_fn(y_star, target_logits_or_ids)

        # Internalization loss (first-class)
        int_loss = torch.tensor(0.0, device=y0.device)
        if proposal_engine is not None:
            # In real wiring: re-run proposal or cache y0
            int_loss = self.internalization(y0, y_star)

        total = primary_loss + int_loss

        logs = {
            "sot_primary_loss": primary_loss.detach(),
            "sot_internalization_loss": int_loss.detach() if isinstance(int_loss, torch.Tensor) else int_loss,
            "sot_total": total.detach(),
            "sot_steps": self.cfg.segment_length,
        }
        return logs, total, y_star.detach()   # detached carry for next segment


# =============================================================================
# 6. Factory + High-level helper (for easy wiring / tests)
# =============================================================================

def make_attractor_solver(dim: int, **kwargs) -> AttractorSolverModule:
    return AttractorSolverModule(dim=dim, **kwargs)


def solve_fixed_point_with_persistent_injection(
    solver: AttractorSolverModule,
    y0: torch.Tensor,
    slow_context: Optional[Dict] = None,
    **solve_kwargs,
) -> Tuple[torch.Tensor, Dict]:
    """Convenience wrapper matching the paper's recommended inference path."""
    return solver.solve(y0, slow_context, **solve_kwargs)


# =============================================================================
# Minimal self-test (run `python -m src.qtrm_mm.attractor.attractor_solver`)
# =============================================================================

if __name__ == "__main__":
    print("=== Attractor Solver Prototype Self-Test ===")
    B, T, D = 2, 16, 128
    y0 = torch.randn(B, T, D)
    slow = {"summary": torch.randn(B, D)}

    solver = AttractorSolverModule(dim=D, num_layers=1, use_parcae=True, max_solver_steps=12)
    eq, meta = solver.solve(y0, slow, max_steps=8, tol=1e-2, return_trajectory=False)
    print(f"Solve completed in {meta['steps_taken']} steps, final residual {meta['final_residual']:.4f}")

    sot_cfg = SOTConfig(segment_length=5, internalization_weight=0.1)
    sot = SOTSegmentedSolverTrainer(solver, lambda y, t: F.mse_loss(y, torch.zeros_like(y)), sot_cfg)
    logs, total, carry = sot.train_segment(y0, slow, None)
    print("SOT segment logs:", {k: float(v) for k, v in logs.items() if torch.is_tensor(v)})

    print("Prototype OK — ready for trainer integration and wiki-aligned experiments.")
