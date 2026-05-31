"""
Latent Episode Memory (LEM) - 2026-06 Radical Architecture Shift

Core Hypothesis (after repeated failures of per-step and even decoupled-bank approaches):
The fundamental problem is not "how good the memory mechanism is", but the **temporal granularity** of memory access inside a high-frequency recurrent loop.

In current designs (embedded slots or even decoupled banks called during every micro-step or every rehearsal step):
- Memory write opportunities are extremely frequent (almost continuous).
- The model never experiences strong pressure to be selective, because "I can always write later in the next micro-step".
- Utility signal for any individual write is extremely diluted.

Radical change:
Make memory access **episode-based and sparse in thinking time**.

- The recurrent core (OneBodyParallelHybridBlock) runs "episodes" of variable micro-steps.
- During an active episode, memory writes are heavily restricted or disabled (transient fast thinking only).
- At episode boundaries (decided by a learned commit/halt mechanism), a summary or selected states from the episode are written to persistent long-term memory via an explicit controller.
- Retrieval from long-term memory happens at the start of new episodes or when the current fast state signals high uncertainty/novelty.

This directly attacks the "too many write opportunities" root cause identified across many experiments.

This is inspired by:
- Complementary Learning Systems (fast hippocampus vs slow neocortex)
- Global Workspace Theory (limited access to the "stage")
- MELT-style decoupling, but pushed further into temporal structure rather than just cache sharing.
- Human "thinking in chunks" rather than continuous micro-updates.

Design goals for this module:
- Preserve One-Body: final answer still comes from the normal recurrent state → LM head path.
- Strong ablation contract (episode_memory on/off must be clean).
- Compatible with existing hybrid recurrence (can be used on top of or instead of previous slot/bank systems).
- Explicit "commit points" so the training signal for selectivity becomes much stronger.

This is a genuine causal route change, not another layer of gating on the old route.
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentEpisodeMemory(nn.Module):
    """
    Episode-based persistent memory for latent recurrent reasoning.

    Key concepts:
    - fast_transient_state: what the recurrence is currently using (not persisted to long-term unless committed).
    - long_term_bank: the actual persistent selective memory (Raven/MSA style slots or more advanced).
    - episode_commit_controller: decides at episode end whether and what to commit from the recent trajectory.
    - retrieval_controller: decides when and how much to pull from long_term_bank into the current fast state.

    During an episode:
    - The hybrid block mostly operates with fast transient dynamics.
    - Persistent memory influence is limited (e.g., only initial retrieval at episode start).

    At episode commit:
    - A compact summary (or selected key states) of the episode is fed to the commit controller.
    - The controller produces a write mask + strength for the long_term_bank.
    """

    def __init__(
        self,
        d_model: int,
        num_slots: int = 16,
        top_k: int = 4,
        controller_hidden: int = 128,
        max_episode_length: int = 16,  # safety cap; real episodes can be shorter via commit gate
    ):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.top_k = top_k
        self.max_episode_length = max_episode_length

        # Long-term selective memory bank (can be upgraded later to more advanced structures)
        self.long_term_slots = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)

        # Episode commit controller
        # Input: summary of recent fast trajectory + current fast state + optional uncertainty signal
        # Output: commit strength + which slots to update
        self.commit_controller = nn.Sequential(
            nn.Linear(d_model * 2 + 1, controller_hidden),  # fast_state + episode_summary + uncertainty
            nn.GELU(),
            nn.Linear(controller_hidden, num_slots + 1),   # slot selection logits + global commit gate
        )

        # Retrieval controller (when to pull from long-term memory)
        self.retrieve_controller = nn.Sequential(
            nn.Linear(d_model + 1, controller_hidden),
            nn.GELU(),
            nn.Linear(controller_hidden, num_slots + 1),   # retrieval weights + strength
        )

        # Simple running episode buffer (for minimal version we keep a short fixed-size buffer of recent states)
        self.register_buffer("episode_buffer", torch.zeros(1, max_episode_length, d_model))
        self.register_buffer("episode_length", torch.zeros(1, dtype=torch.long))

        self._enabled = True
        self._ablation_zero = False

    def set_ablation(self, enabled: bool = True, ablation_zero: bool = False):
        self._enabled = enabled
        self._ablation_zero = ablation_zero

    def reset_episode(self, batch_size: int, device=None, dtype=None):
        """Call this at the start of a new thinking episode."""
        if device is None:
            device = self.long_term_slots.device
        if dtype is None:
            dtype = self.long_term_slots.dtype
        self.episode_buffer = torch.zeros(batch_size, self.max_episode_length, self.d_model, device=device, dtype=dtype)
        self.episode_length = torch.zeros(batch_size, dtype=torch.long, device=device)

    def step_fast_state(self, fast_state: torch.Tensor):
        """
        Call this every micro-step of the hybrid recurrence during an episode.
        We just accumulate the fast state for later summarization / commit decision.
        """
        if not self._enabled or self._ablation_zero:
            return

        b = fast_state.shape[0]
        if self.episode_buffer.shape[0] != b:
            self.reset_episode(b, fast_state.device, fast_state.dtype)

        idx = self.episode_length.clamp(max=self.max_episode_length - 1)
        # Simple rolling buffer (overwrite oldest when full)
        for i in range(b):
            pos = idx[i].item()
            self.episode_buffer[i, pos] = fast_state[i].detach()
        self.episode_length = (self.episode_length + 1).clamp(max=self.max_episode_length)

    def retrieve_at_episode_start(
        self,
        current_fast_state: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        At the beginning of a new episode (or when uncertainty is high), decide whether
        and what to retrieve from long-term memory into the fast thinking state.
        """
        if not self._enabled or self._ablation_zero:
            return torch.zeros_like(current_fast_state), torch.zeros(current_fast_state.shape[0], self.num_slots, device=current_fast_state.device)

        b = current_fast_state.shape[0]
        state = current_fast_state.mean(dim=1) if current_fast_state.dim() == 3 else current_fast_state

        util = uncertainty if uncertainty is not None else torch.zeros(b, 1, device=state.device)
        if util.dim() == 1:
            util = util.unsqueeze(1)

        ctrl_in = torch.cat([state, util], dim=-1)
        ctrl_out = self.retrieve_controller(ctrl_in)

        slot_logits = ctrl_out[:, :self.num_slots]
        strength = torch.sigmoid(ctrl_out[:, -1:])

        probs = F.softmax(slot_logits, dim=-1)
        k = min(self.top_k, self.num_slots)
        topk_p, topk_idx = torch.topk(probs, k, dim=-1)

        weights = torch.zeros_like(probs)
        weights.scatter_(1, topk_idx, topk_p * strength)

        slots = self.long_term_slots.expand(b, -1, -1)
        values = self.value_proj(slots)
        context = torch.matmul(weights.unsqueeze(1), values).squeeze(1)

        return context, weights

    def _ensure_dtype(self, x: torch.Tensor) -> torch.Tensor:
        """Defensive cast to the module's compute dtype (important for bf16 models)."""
        if not self._enabled or self._ablation_zero:
            return x
        try:
            target_dtype = next(self.parameters()).dtype
            if x.dtype != target_dtype:
                x = x.to(target_dtype)
        except StopIteration:
            pass
        return x

    def commit_episode(
        self,
        current_fast_state: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        write_strength_scale: float = 0.2,
    ) -> Dict[str, torch.Tensor]:
        """
        At the end of a coherent thinking episode, decide what to commit to long-term memory.
        This is the key selective write point.
        """
        if not self._enabled or self._ablation_zero:
            return {"committed": False}

        b = current_fast_state.shape[0]
        state = current_fast_state.mean(dim=1) if current_fast_state.dim() == 3 else current_fast_state
        state = self._ensure_dtype(state)

        # Simple episode summary: mean of buffered states
        valid_len = self.episode_length.clamp(min=1)
        summary = self.episode_buffer.sum(dim=1) / valid_len.unsqueeze(1)
        summary = self._ensure_dtype(summary)

        util = uncertainty if uncertainty is not None else torch.zeros(b, 1, device=state.device, dtype=state.dtype)
        if util.dim() == 1:
            util = util.unsqueeze(1)
        util = self._ensure_dtype(util)

        ctrl_in = torch.cat([state, summary, util], dim=-1)
        ctrl_out = self.commit_controller(ctrl_in)

        slot_logits = ctrl_out[:, :self.num_slots]
        global_commit = torch.sigmoid(ctrl_out[:, -1:])

        probs = F.softmax(slot_logits, dim=-1)
        k = min(self.top_k, self.num_slots)
        topk_p, topk_idx = torch.topk(probs, k, dim=-1)

        write_mask = torch.zeros_like(probs)
        write_mask.scatter_(1, topk_idx, topk_p)

        effective_write = global_commit.unsqueeze(1) * write_strength_scale * write_mask.unsqueeze(-1)

        current_slots = self.long_term_slots.expand(b, -1, -1)
        update_content = summary.unsqueeze(1).expand(-1, self.num_slots, -1)

        new_slots = current_slots * (1.0 - effective_write) + update_content * effective_write

        # Strong persistence on non-written slots (core selective memory bias)
        persistence = 0.92
        non_write_mask = 1.0 - write_mask.unsqueeze(-1)
        new_slots = new_slots * (non_write_mask * persistence + (1.0 - non_write_mask) * 0.3) + \
                    current_slots * (1.0 - (non_write_mask * persistence + (1.0 - non_write_mask) * 0.3))

        with torch.no_grad():
            self.long_term_slots.copy_(new_slots.mean(dim=0, keepdim=True))

        # Reset episode buffer after commit
        self.episode_length.zero_()

        return {
            "committed": True,
            "global_commit_strength": global_commit,
            "slot_write_mask": write_mask,
        }

    def get_long_term_state(self) -> torch.Tensor:
        return self.long_term_slots.detach().clone()


def make_latent_episode_memory(d_model: int, num_slots: int = 16, top_k: int = 4, **kwargs) -> LatentEpisodeMemory:
    return LatentEpisodeMemory(d_model, num_slots, top_k, **kwargs)
