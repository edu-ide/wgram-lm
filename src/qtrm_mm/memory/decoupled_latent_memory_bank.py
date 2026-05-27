"""
Decoupled Latent Memory Bank (2026-06 proposal)

Inspired by:
- MELT (arXiv:2605.07721, 2026): Decoupling compute (loop depth) from memory in LoopLM-style recurrent models via shared evolving cache + learnable gating.
- G-MemLLM (arXiv:2602.00015): Trainable Latent Memory Bank with gated updates (preserve/overwrite).
- LM2 (arXiv:2502.06049): Auxiliary memory module with explicit input/forget/output gates + cross-attention retrieval.
- Titans/ATLAS lineage: Utility/surprise-driven writes instead of every-step updates.

Core idea for this codebase:
Current RI-4 embeds slots inside every hybrid recurrence micro-step (tight coupling).
This makes learning "when a write is actually useful" very difficult because write opportunities are constant and high-frequency.

This module proposes a **decoupled bank**:
- The recurrent core (OneBodyParallelHybridBlock / answer_state_loop) can query the bank.
- Writes are controlled by an explicit lightweight controller + utility signal (not automatic per-step).
- This changes the causal route: memory is no longer a synchronous side-effect of every thinking step.

Design goals (strictly enforced):
- One-Body compliant: Bank context is injected into the normal recurrent state path that feeds the LM head.
- Full ablation support: bank_on vs bank_off must be clean (no behavior change when off).
- Compatible with existing 5.56 rehearsal + RI-4 4-way contract (can be used orthogonally or as evolution of slots).
- Minimal at first: single shared bank + small controller. Scale later.

This is the skill-mandated next Big Jump candidate after repeated falsification of embedded per-block slot designs.
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoupledLatentMemoryBank(nn.Module):
    """
    A separate, addressable latent memory bank that the recurrent thinker can
    query and selectively update via a controller.

    Unlike the current embedded SparseSlotRouter (updated inside every block forward),
    this bank lives at a higher scope and is interacted with on a more controlled schedule.

    Slots: (B, num_slots, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_slots: int = 16,
        top_k: int = 4,
        controller_hidden: int = 128,
        persistence: float = 0.92,  # strong default non-selected persistence (Raven-style)
    ):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.top_k = top_k
        self.persistence = persistence

        # Bank content
        self.slots = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)

        # Lightweight controller for write decisions
        # Input: current recurrent state summary + optional utility/surprise signal
        # Output: write gate (how much to write) + which slots (soft selection)
        self.controller = nn.Sequential(
            nn.Linear(d_model + 1, controller_hidden),  # +1 for optional utility scalar
            nn.GELU(),
            nn.Linear(controller_hidden, num_slots + 1),  # num_slots for selection logits + 1 global write strength
        )

        # Optional value projection for the bank (like a small key-value memory)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)

        self._enabled = True
        self._ablation_zero = False

    def set_ablation(self, enabled: bool = True, ablation_zero: bool = False):
        """Clean ablation interface matching existing RI-4 contract."""
        self._enabled = enabled
        self._ablation_zero = ablation_zero

    def forward_read(
        self,
        query: torch.Tensor,           # (B, d) or (B, 1, d) — current recurrent state as query
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read from the bank using content-based addressing.

        Returns:
            memory_context: (B, d) weighted sum of selected slots
            slot_weights: (B, num_slots) attention weights over slots
        """
        if not self._enabled or self._ablation_zero:
            b = query.shape[0]
            return torch.zeros(b, self.d_model, device=query.device, dtype=query.dtype), \
                   torch.zeros(b, self.num_slots, device=query.device)

        if query.dim() == 3:
            query = query.mean(dim=1)  # (B, d)

        slots = self.slots.expand(query.size(0), -1, -1)  # (B, num_slots, d)
        keys = slots

        scores = torch.matmul(query.unsqueeze(1), keys.transpose(-2, -1)).squeeze(1) / (self.d_model ** 0.5)
        k = top_k or self.top_k
        topk_scores, topk_idx = torch.topk(scores, k=min(k, self.num_slots), dim=-1)

        weights = torch.zeros_like(scores)
        weights.scatter_(1, topk_idx, F.softmax(topk_scores, dim=-1))

        values = self.value_proj(slots)
        context = torch.matmul(weights.unsqueeze(1), values).squeeze(1)

        return context, weights

    def controller_write(
        self,
        current_state: torch.Tensor,      # (B, d) summary of current recurrent thinking
        utility_signal: Optional[torch.Tensor] = None,  # scalar or (B,) surprise/utility from rehearsal or internal prediction error
        rehearsal_target: Optional[torch.Tensor] = None,
        write_strength: float = 0.15,
    ) -> Dict[str, torch.Tensor]:
        """
        Learned controller decides whether and how strongly to write.

        This is the key decoupling: the bank is NOT updated automatically inside every
        recurrence micro-step. The controller (or external rehearsal loop) calls this
        at more meaningful moments (e.g., after a burst of thinking, or during 5.56 rehearsal).

        Returns metadata + the updated slots (for external assignment).
        """
        if not self._enabled or self._ablation_zero:
            return {"updated_slots": self.slots.expand(current_state.size(0), -1, -1), "write_gate": torch.zeros(current_state.size(0), 1)}

        b = current_state.shape[0]
        state_summary = current_state.mean(dim=1) if current_state.dim() == 3 else current_state

        util = utility_signal
        if util is None:
            util = torch.zeros(b, 1, device=state_summary.device)
        else:
            if util.dim() == 0:
                util = util.unsqueeze(0).unsqueeze(1)   # scalar → (1,1)
            elif util.dim() == 1:
                util = util.unsqueeze(1)                # (B,) → (B,1)
            # if already (B,1) or more, leave it

        # Ensure util has batch dim matching state_summary
        if util.shape[0] != b:
            util = util.expand(b, -1)

        ctrl_in = torch.cat([state_summary, util], dim=-1)
        ctrl_out = self.controller(ctrl_in)  # (B, num_slots + 1)

        slot_logits = ctrl_out[:, :self.num_slots]
        global_write = torch.sigmoid(ctrl_out[:, -1:])

        # Soft top-k selection for write
        probs = F.softmax(slot_logits / 0.8, dim=-1)  # temperature for exploration
        k = min(self.top_k, self.num_slots)
        topk_p, topk_idx = torch.topk(probs, k, dim=-1)

        write_mask = torch.zeros_like(probs)
        write_mask.scatter_(1, topk_idx, topk_p)

        # Compute update content (simple for minimal version: current state as candidate)
        update_content = state_summary.unsqueeze(1).expand(-1, self.num_slots, -1)

        # Gated write (inspired by G-MemLLM / LM2 style)
        effective_write = global_write.unsqueeze(1) * write_strength * write_mask.unsqueeze(-1)

        current_slots = self.slots.expand(b, -1, -1)
        new_slots = current_slots * (1.0 - effective_write) + update_content * effective_write

        # Strong persistence on non-written slots (core Raven/MSA idea preserved)
        persistence_mask = 1.0 - write_mask.unsqueeze(-1)
        new_slots = new_slots * (persistence_mask * self.persistence + (1.0 - persistence_mask) * 0.3) + \
                    current_slots * (1.0 - (persistence_mask * self.persistence + (1.0 - persistence_mask) * 0.3))

        # Store back (in real use this would be managed by caller or a higher-level state)
        with torch.no_grad():
            self.slots.copy_(new_slots.mean(dim=0, keepdim=True))

        return {
            "updated_slots": new_slots,
            "write_gate": global_write,
            "slot_selection": write_mask,
        }

    def get_bank_state(self) -> torch.Tensor:
        return self.slots.detach().clone()


def make_decoupled_latent_memory_bank(
    d_model: int,
    num_slots: int = 16,
    top_k: int = 4,
    **kwargs
) -> DecoupledLatentMemoryBank:
    return DecoupledLatentMemoryBank(d_model, num_slots, top_k, **kwargs)
