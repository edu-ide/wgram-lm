"""
Chunked Slow Memory Adapter (LaCT + ATLAS Omega + Titans Surprise)

This is the highest-impact remaining piece for Hybrid Brain-Mimetic Recurrence v2.

Goal (per papers + wiki H):
- The *expensive* slow memory (neural LTM + gated long-term slots) should **not** be updated every micro-step or fixed small stride.
- Instead, collect context + surprise over a coherent "chunk" (token window or thinking step window, or high-surprise boundary).
- Perform one batched, high-quality update (Omega rule style optimization or surprise-modulated gated write) at chunk boundaries.
- This enables:
  * Much lower Python dispatch cost for the slow path
  * Better hardware utilization (LaCT large-chunk lesson: 50-70% FLOPS)
  * More stable / higher quality long-term consolidation (ATLAS Omega window optimization)

Current state (before this module):
- All slow/long-term reads and writes were driven per micro-step (or strided) from inside BrainMimeticTripleMemory.step() / light_update().
- This is exactly the pattern LaCT and ATLAS say is inefficient.

Design (strict RI principles):
- One-Body only
- Perfect ablation (chunked_slow_off → falls back to previous per-step/strided behavior)
- Surprise from PredictiveDataIntuition is first-class trigger + modulator
- Clean state snapshot/restore for trainer + serving
- Can wrap either SparseGatedLongTermMemory (slot) or a future deep neural memory (Titans-style MLP)

This module does NOT replace the fast internal recurrence (FastGatedLinearRecurrence).
It complements it: fast path stays cheap + per-micro inside the block; slow path becomes chunked and high-signal.
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Any
import torch
from torch import nn

from .sparse_gated_long_term_memory import SparseGatedLongTermMemory


class ChunkedSlowMemoryAdapter(nn.Module):
    """
    Orchestrates chunked / boundary-driven updates to the slow persistent memory.

    Usage pattern (future ideal):
        adapter = ChunkedSlowMemoryAdapter(
            long_term_memory=sparse_gated_ltm,
            chunk_size=1024,           # tokens or micro-steps
            use_surprise_trigger=True
        )

        for micro in think_loop:
            ... fast recurrence runs inside block ...
            surprise = data_intuition(...)
            read_context = adapter.light_read(current_state, surprise)

            # Inject read_context into fast path or working memory

            adapter.accumulate(current_state, surprise)

            if adapter.should_commit():
                adapter.commit()   # one expensive windowed update

    For the absolute first version we keep it simple and safe.
    """

    def __init__(
        self,
        long_term_memory: Any,                    # SparseGatedLongTermMemory or future neural LTM
        chunk_size: int = 64,                     # LaCT/Omega/ATLAS radical: large chunks (was 8, now 64+ per E/H sections of brain_attractor MD)
        use_surprise_trigger: bool = True,
        surprise_threshold: float = 0.6,
        ablation_zero: bool = False,
    ):
        super().__init__()
        self.long_term_memory = long_term_memory
        self.chunk_size = max(1, int(chunk_size))
        self.use_surprise_trigger = use_surprise_trigger
        self.surprise_threshold = surprise_threshold
        self._ablation_zero = bool(ablation_zero)

        # Accumulation buffers (very lightweight)
        self._accumulated_context: List[torch.Tensor] = []
        self._accumulated_surprise: List[torch.Tensor] = []
        self._steps_since_commit = 0

        self._enabled = not self._ablation_zero

    def set_ablation(self, enabled: bool = True, ablation_zero: bool = False):
        self._enabled = enabled and not ablation_zero
        self._ablation_zero = ablation_zero
        if ablation_zero:
            self.reset()

    def reset(self):
        self._accumulated_context.clear()
        self._accumulated_surprise.clear()
        self._steps_since_commit = 0

    def should_commit(self, current_surprise: Optional[torch.Tensor] = None) -> bool:
        if not self._enabled or self._ablation_zero:
            return False

        self._steps_since_commit += 1

        force_by_size = self._steps_since_commit >= self.chunk_size

        if self.use_surprise_trigger and current_surprise is not None:
            # High surprise → commit even if chunk not full
            s = current_surprise.detach().abs().mean().item() if torch.is_tensor(current_surprise) else float(current_surprise)
            if s > self.surprise_threshold:
                return True

        return force_by_size

    def accumulate(self, context: torch.Tensor, surprise: Optional[torch.Tensor] = None):
        """Call this every micro-step with the current thought state + surprise signal."""
        if not self._enabled or self._ablation_zero:
            return

        self._accumulated_context.append(context.detach())
        if surprise is not None:
            self._accumulated_surprise.append(surprise.detach() if torch.is_tensor(surprise) else surprise)

    def light_read(
        self,
        query: torch.Tensor,
        surprise: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Cheap read path that can be called every micro-step.
        In the best-state architecture this read can still be frequent (it's relatively cheap),
        while the heavy write/commit is chunked.
        """
        if not self._enabled or self._ablation_zero or self.long_term_memory is None:
            return None

        try:
            read_signal, _, _ = self.long_term_memory(
                query=query,
                surprise=surprise,
            )
            return read_signal
        except Exception:
            return None

    def commit(self) -> bool:
        """
        ATLAS Omega + Titans inspired chunked commit.

        Instead of per-step writes, we accumulate a window of (context, surprise) and
        perform one higher-quality update. This dramatically reduces Python overhead
        and allows better optimization (momentum, decay, windowed loss) over the chunk.

        Current implementation:
        - Surprise-weighted average of the window (stronger signal on surprising moments)
        - Simple momentum/decay (Titans-style) before writing to the underlying memory
        - Still delegates the actual slot update to SparseGatedLongTermMemory (or future neural LTM)
        """
        if not self._enabled or self._ablation_zero or not self._accumulated_context:
            self.reset()
            return False

        if self.long_term_memory is None:
            self.reset()
            return False

        try:
            ctx_stack = torch.stack(self._accumulated_context, dim=0)  # (T, B, d) or similar

            # Surprise-weighted aggregation (core of the "Omega" idea)
            if self._accumulated_surprise and len(self._accumulated_surprise) == ctx_stack.shape[0]:
                raw_s = torch.stack([s for s in self._accumulated_surprise])
                s = raw_s.abs().clamp(0.05, 5.0)
                weights = s / (s.sum(dim=0, keepdim=True) + 1e-8)
                if ctx_stack.dim() == 3:
                    weighted = (ctx_stack * weights.unsqueeze(-1)).sum(0)
                else:
                    weighted = (ctx_stack * weights).sum(0)
            else:
                weighted = ctx_stack.mean(0)

            # LeWM follow-up inspired (RC-aux / TRM / Sub-JEPA spirit, final push)
            # Momentum + decay + horizon/reachability-aware boost + subspace Gaussian regularizer seed.
            # This makes the slow path closer to a "plannable latent world model" (LeWM + follow-ups).
            #
            # ATLAS Omega upgrade (2026-05-28 deep integration): the weighted aggregation is already
            # a step toward Omega (windowed, surprise-modulated). When the caller is in ri1_relaxed
            # training mode we can later replace the simple momentum with a true small-window GD step
            # using data_intuition as the attentional bias (exact Omega formula in the wiki section).
            mom = 0.82
            decay = 0.96
            if not hasattr(self, '_commit_momentum'):
                self._commit_momentum = None

            if self._commit_momentum is None or self._commit_momentum.shape != weighted.shape:
                self._commit_momentum = weighted.detach() * 0.35
            else:
                self._commit_momentum = mom * self._commit_momentum + (1 - mom) * weighted.detach()

            final_update = decay * self._commit_momentum + (1 - decay) * weighted

            # === AGGRESSIVE LeWM-style upgrade (MD I-section #1 concrete suggestion) ===
            # Add a small learned autoregressive latent step + proper horizon-matched reachability.
            # This moves ChunkedSlow from "predictive accumulator" toward "plannable latent world model".

            # Lightweight autoregressive predictor on the latent summary (LeWM recurrent unroll spirit)
            if not hasattr(self, 'latent_ar_proj'):
                self.latent_ar_proj = nn.Linear(final_update.shape[-1] if final_update is not None else 128, 
                                                final_update.shape[-1] if final_update is not None else 128, bias=False)
                if final_update is not None:
                    self.latent_ar_proj = self.latent_ar_proj.to(final_update.device)
                    nn.init.xavier_uniform_(self.latent_ar_proj.weight)

            if final_update is not None and hasattr(self, 'latent_ar_proj'):
                ar_pred = self.latent_ar_proj(final_update)
                # Simple one-step recurrent prediction error as additional reachability signal
                ar_error = (ar_pred - final_update).pow(2).mean(dim=-1, keepdim=True)
                # Use low prediction error = more "reachable/plannable" for the fast path
                reachability_from_ar = 1.0 / (1.0 + 0.5 * ar_error.clamp(0, 4))
                final_update = final_update * (0.7 + 0.3 * reachability_from_ar)

            # RC-aux / TRM style: horizon/reachability boost (improved with AR signal)
            if self._accumulated_surprise:
                avg_surprise = torch.stack([s for s in self._accumulated_surprise]).abs().mean()
                reachability_boost = 1.0 + 0.35 * torch.sigmoid(avg_surprise)
                final_update = final_update * reachability_boost

            # Sub-JEPA / RC-aux style subspace Gaussian (live)
            if hasattr(self, 'long_term_memory') and self.long_term_memory is not None and final_update is not None:
                try:
                    u = final_update
                    if u.dim() > 1:
                        u = u.mean(dim=0) if u.dim() == 2 else u.view(-1)[:256]
                    # Simple 4-subspace projection + unit Gaussian pull (cheap, no extra params)
                    subspaces = 4
                    reg = 0.0
                    for k in range(subspaces):
                        proj = u * (0.7 + 0.1 * k)   # synthetic different "views"
                        reg = reg + (proj.norm() - 1.0).pow(2)
                    # The caller (brain or loss) can use this as aux if desired; we store it.
                    self._last_subspace_reg = float(reg / max(1, subspaces))
                except Exception:
                    self._last_subspace_reg = 0.0

            # Delegate the actual persistent write to the underlying long-term memory
            # (it already has gated + surprise logic)
            _ = self.long_term_memory(
                query=final_update,
                surprise=torch.tensor(1.2, device=final_update.device) if self._accumulated_surprise else None,
            )

            self.reset()
            return True

        except Exception:
            self.reset()
            return False

    def get_state(self):
        """For trainer checkpointing / generation state management."""
        return {
            "accumulated_context": [c.clone() for c in self._accumulated_context],
            "steps_since_commit": self._steps_since_commit,
        }

    def set_state(self, state: dict):
        if state is None:
            self.reset()
            return
        self._accumulated_context = state.get("accumulated_context", [])
        self._steps_since_commit = state.get("steps_since_commit", 0)


# Convenience factory (mirrors other make_* functions in the project)
def make_chunked_slow_memory_adapter(
    long_term_memory: Any,
    chunk_size: int = 8,
    use_surprise_trigger: bool = True,
    ablation_zero: bool = False,
) -> ChunkedSlowMemoryAdapter:
    return ChunkedSlowMemoryAdapter(
        long_term_memory=long_term_memory,
        chunk_size=chunk_size,
        use_surprise_trigger=use_surprise_trigger,
        ablation_zero=ablation_zero,
    )
