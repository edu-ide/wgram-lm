from __future__ import annotations
from typing import Optional
import torch
from torch import nn

from .config import QTRMConfig
from .blocks import QTRMBlockStack
from .stability import StableInject
from .norm import RMSNorm


class QTRMRecursiveCore(nn.Module):
    """TRM-style z_L/z_H recurrent latent workspace core."""

    def __init__(self, cfg: QTRMConfig):
        super().__init__()
        self.cfg = cfg
        self.fast_stack = QTRMBlockStack(cfg, cfg.n_core_layers, causal=False, attn_every=cfg.attn_every)
        self.slow_stack = QTRMBlockStack(cfg, cfg.n_core_layers, causal=False, attn_every=cfg.attn_every)
        self.inject_l = StableInject(cfg.d_model) if cfg.use_stable_inject else None
        self.inject_h = StableInject(cfg.d_model) if cfg.use_stable_inject else None
        self.z_l_init = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.z_h_init = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.norm_l = RMSNorm(cfg.d_model)
        self.norm_h = RMSNorm(cfg.d_model)
        self.halt_head = nn.Linear(cfg.d_model, 2) if cfg.core_halt_enabled else None

    def forward(
        self,
        workspace: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        enable_halt: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        b, w, d = workspace.shape
        z_l = workspace + self.z_l_init
        z_h = workspace + self.z_h_init
        trajectory = []
        q_halt_steps = []
        q_continue_steps = []
        halted = torch.zeros(b, device=workspace.device, dtype=torch.bool)
        enable_halt = bool(self.cfg.core_halt_enabled if enable_halt is None else enable_halt)
        loop_id = 0
        for outer in range(self.cfg.outer_steps):
            for h in range(self.cfg.h_cycles):
                for l in range(self.cfg.l_cycles):
                    source = z_h + workspace
                    if self.inject_l is not None:
                        source = self.inject_l(z_l, source, loop_id=loop_id)
                    z_l = self.norm_l(z_l + source)
                    z_l = self.fast_stack(z_l, attention_mask=attention_mask)
                    loop_id += 1
                source_h = z_l
                if self.inject_h is not None:
                    source_h = self.inject_h(z_h, source_h, loop_id=loop_id)
                z_h = self.norm_h(z_h + source_h)
                z_h = self.slow_stack(z_h, attention_mask=attention_mask)
                loop_id += 1
            trajectory.append(z_h)
            if self.halt_head is not None:
                q_logits = self.halt_head(z_h[:, 0, :]).to(torch.float32)
                q_halt = q_logits[..., 0]
                q_continue = q_logits[..., 1]
                q_halt_steps.append(q_halt)
                q_continue_steps.append(q_continue)
                if enable_halt and (outer + 1) >= max(1, int(self.cfg.core_halt_min_steps)):
                    if self.cfg.core_halt_use_continue:
                        halted = q_halt > q_continue
                    else:
                        halted = q_halt > 0
                    if bool(halted.all().detach().cpu().item()):
                        break
            if self.cfg.truncated_recurrence and self.training:
                z_l = z_l.detach()
                z_h = z_h.detach()
        steps = len(trajectory)
        if q_halt_steps:
            q_halt_logits = torch.stack(q_halt_steps, dim=1)
            q_continue_logits = torch.stack(q_continue_steps, dim=1)
        else:
            q_halt_logits = workspace.new_empty((b, 0), dtype=torch.float32)
            q_continue_logits = workspace.new_empty((b, 0), dtype=torch.float32)
        halt_info = {
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "halted": halted if enable_halt else torch.zeros_like(halted),
            "steps": torch.full((b,), steps, device=workspace.device, dtype=torch.long),
        }
        return z_l, z_h, trajectory, halt_info
