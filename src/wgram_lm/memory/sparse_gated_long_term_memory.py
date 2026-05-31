"""
Sparse Gated Long-Term Memory (Raven + LM² + MSA 2026 + Surprise driven)

This module provides a clean, first-class long-term persistent memory layer
designed to work with the BrainMimeticTripleMemory + PredictiveDataIntuition system.

Core ideas drawn from latest literature (2025-2026):
- Raven / Routing State Model: Persistent slots + learned sparse router. Unselected slots have near-perfect persistence (anti-interference).
- LM² / G-MemLLM: Explicit gated read/write (input/forget/output style gates) on memory slots.
- MSA (Memory Sparse Attention, 2026): Scalable sparse attention patterns over memory segments + tiered thinking about memory.
- Titans / ATLAS + surprise mechanisms: Surprise (prediction error / novelty) as a first-class signal for what to attend to and what to consolidate.
- EM-LLM style episodic thinking: High-surprise periods are more likely to be written to long-term memory.

Design goals (strict, per project RI principles):
- One-Body only: Everything ultimately flows into the main recurrent state → LM head path.
- Perfect, cheap ablations.
- Surprise from Predictive Data Intuition is a first-class citizen for both read attention and write decisions.
- Clean state management for trainers (get latest state, set state, etc.).
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import nn

from .sparse_slot_router import SparseSlotRouter


class SparseGatedLongTermMemory(nn.Module):
    """
    First-class long-term memory layer for the BMSAM (Brain-Mimetic Sparse Attractor Memory) architecture.

    Responsibilities:
    - Maintain a pool of persistent memory slots.
    - Perform sparse, gated, surprise-modulated reads and writes.
    - Serve as the "slow, high-capacity, low-interference" memory that complements the fast ActiveWorkingMemory + Attractor + Predictive Data Intuition system.
    """

    def __init__(
        self,
        d_model: int,
        num_slots: int = 64,
        top_k: int = 12,
        router_hidden: Optional[int] = None,
        use_gated_update: bool = True,
        use_surprise: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.top_k = min(top_k, num_slots)

        # Core router (Raven + LM2 gated + surprise support already implemented)
        self.router = SparseSlotRouter(
            d_model=d_model,
            num_slots=num_slots,
            top_k=top_k,
            router_hidden=router_hidden,
        )

        if use_gated_update:
            self.router.enable_gated_memory_update(True)

        if use_surprise:
            self.router.enable_surprise_write_trigger(True, surprise_scale=1.5)

        # Internal persistent state (B, num_slots, d) — managed across steps
        self.register_buffer("_slots", None)  # lazily initialized

        self._use_surprise = use_surprise

    def _ensure_slots(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        if self._slots is None or self._slots.shape[0] != batch_size or self._slots.device != device or self._slots.dtype != dtype:
            self._slots = self.router.slot_memory.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype).clone()

    def _ensure_same_device(self, tensor: torch.Tensor):
        """Force the long-term memory module and internal states to the tensor's device/dtype."""
        device = tensor.device
        dtype = tensor.dtype
        if next(self.parameters()).device != device or next(self.parameters()).dtype != dtype:
            self.to(device=device, dtype=dtype)
        if self._slots is not None and (self._slots.device != device or self._slots.dtype != dtype):
            self._slots = self._slots.to(device=device, dtype=dtype)

    def forward(
        self,
        query: torch.Tensor,                    # (B, d) — usually pooled current thought state
        surprise: Optional[torch.Tensor] = None, # (B, 1) or scalar — from PredictiveDataIntuition
        stochastic_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a sparse read from long-term memory.

        Returns:
            read_signal: (B, d) — weighted read from selected slots
            mask: (B, num_slots) — which slots were selected
            current_slots: (B, num_slots, d)
        """
        self._ensure_same_device(query)
        self._ensure_slots(query.shape[0], query.device, query.dtype)

        # If surprise is provided, we can bias the router scores (future: make this more sophisticated)
        # For now we pass it through stochastic_noise path or let the caller modulate the query.
        read_signal, mask, slots = self.router(
            query,
            stochastic_noise=stochastic_noise,
            slot_state=self._slots,
        )

        # Update internal state reference (read does not mutate slots, only write does)
        self._slots = slots

        return read_signal, mask, slots

    def write(
        self,
        update_signal: torch.Tensor,            # (B, d)
        mask: torch.Tensor,                     # (B, num_slots) from a previous forward()
        surprise: Optional[torch.Tensor] = None, # (B, 1) — modulates write strength
    ) -> torch.Tensor:
        """
        Perform a gated, surprise-modulated write into long-term memory.

        This is the consolidation step: "data intuition" decides how aggressively to commit the current thought.
        """
        self._ensure_same_device(update_signal)
        self._ensure_slots(update_signal.shape[0], update_signal.device, update_signal.dtype)

        learning_rate = 0.08
        if surprise is not None:
            surprise_factor = (1.0 + 1.2 * surprise.mean().item())
            learning_rate *= surprise_factor

        updated_slots = self.router.update_slots(
            slot_state=self._slots,
            update_signal=update_signal,
            slot_mask=mask,
            learning_rate=learning_rate,
        )

        self._slots = updated_slots
        return updated_slots

    def get_state(self) -> torch.Tensor:
        """Returns the current persistent slot state for external saving."""
        return self._slots.clone() if self._slots is not None else None

    def set_state(self, state: torch.Tensor):
        """Restore persistent slot state (e.g. from checkpoint)."""
        self._slots = state.clone()

    def set_ablation(self, enabled: bool = True, ablation_zero: bool = False):
        self.router.set_ablation(enabled=enabled, ablation_zero=ablation_zero)

    def enable_fast_eval(self, enabled: bool = True, cache_interval: int = 4, decay: float = 0.9):
        """Forward fast eval optimization to the router (used in native 72 push)."""
        self.router.enable_fast_eval(enabled, cache_interval, decay)