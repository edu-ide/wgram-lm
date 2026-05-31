from __future__ import annotations
from typing import Optional
import torch
from torch import nn

from .attention import CrossAttention
from .ffn import SwiGLU
from .norm import RMSNorm


class LatentWorkspaceLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        ff_mult: int = 0,
        include_latents_in_kv: bool = False,
        memory_gate_enabled: bool = False,
        memory_gate_init_bias: float = -2.0,
    ):
        super().__init__()
        self.include_latents_in_kv = include_latents_in_kv
        self.memory_gate_enabled = memory_gate_enabled
        self.cross_norm_q = RMSNorm(d_model)
        self.cross_norm_context = RMSNorm(d_model)
        self.cross = CrossAttention(d_model=d_model, n_heads=n_heads)
        self.gate_norm_prev = RMSNorm(d_model) if memory_gate_enabled else None
        self.gate_norm_update = RMSNorm(d_model) if memory_gate_enabled else None
        self.update_gate = nn.Linear(d_model * 2, d_model) if memory_gate_enabled else None
        self.reset_gate = nn.Linear(d_model * 2, d_model) if memory_gate_enabled else None
        self.candidate = nn.Linear(d_model * 2, d_model) if memory_gate_enabled else None
        if self.update_gate is not None:
            nn.init.constant_(self.update_gate.bias, float(memory_gate_init_bias))
        self.out_norm = RMSNorm(d_model)
        self.ff_norm = RMSNorm(d_model) if ff_mult > 0 else None
        self.ff = SwiGLU(d_model, d_model * ff_mult) if ff_mult > 0 else None

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        *,
        return_info: bool = False,
        disable_memory_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        kv = context
        kv_mask = context_mask
        if self.include_latents_in_kv:
            kv = torch.cat([context, latents], dim=1)
            if context_mask is not None:
                latent_mask = torch.ones(
                    latents.shape[:2],
                    device=context_mask.device,
                    dtype=context_mask.dtype,
                )
                kv_mask = torch.cat([context_mask, latent_mask], dim=1)

        prev_latents = latents
        cross_updated = latents + self.cross(
            self.cross_norm_q(latents),
            self.cross_norm_context(kv),
            context_mask=kv_mask,
        )
        update_gate = None
        if self.memory_gate_enabled and not disable_memory_gate:
            if (
                self.gate_norm_prev is None
                or self.gate_norm_update is None
                or self.update_gate is None
                or self.reset_gate is None
                or self.candidate is None
            ):
                raise RuntimeError("memory_gate_enabled=True but gate modules are missing")
            gate_input = torch.cat(
                [
                    self.gate_norm_prev(prev_latents),
                    self.gate_norm_update(cross_updated),
                ],
                dim=-1,
            )
            update_gate = torch.sigmoid(self.update_gate(gate_input))
            reset_gate = torch.sigmoid(self.reset_gate(gate_input))
            candidate_input = torch.cat([reset_gate * prev_latents, cross_updated], dim=-1)
            candidate = torch.tanh(self.candidate(candidate_input))
            latents = (1.0 - update_gate) * prev_latents + update_gate * candidate
        else:
            latents = cross_updated
        if self.ff is not None and self.ff_norm is not None:
            latents = latents + self.ff(self.ff_norm(latents))
        latents = self.out_norm(latents)
        if return_info:
            gate_mean = update_gate.mean(dim=(1, 2)) if update_gate is not None else None
            return latents, gate_mean
        return latents


class LatentWorkspace(nn.Module):
    def __init__(
        self,
        d_model: int,
        workspace_tokens: int,
        n_heads: int,
        *,
        layers: int = 1,
        ff_mult: int = 0,
        include_latents_in_kv: bool = False,
        memory_gate_enabled: bool = False,
        memory_gate_init_bias: float = -2.0,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("LatentWorkspace requires at least one layer")
        self.workspace_tokens = workspace_tokens
        self.memory_gate_enabled = memory_gate_enabled
        self.workspace = nn.Parameter(torch.randn(workspace_tokens, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [
                LatentWorkspaceLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_mult=ff_mult,
                    include_latents_in_kv=include_latents_in_kv,
                    memory_gate_enabled=memory_gate_enabled,
                    memory_gate_init_bias=memory_gate_init_bias,
                )
                for _ in range(layers)
            ]
        )

    def forward(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        *,
        return_info: bool = False,
        disable_memory_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b = context.shape[0]
        q = self.workspace.unsqueeze(0).expand(b, -1, -1)
        update_gate_means = []
        for layer in self.layers:
            if return_info:
                q, gate_mean = layer(
                    q,
                    context,
                    context_mask=context_mask,
                    return_info=True,
                    disable_memory_gate=disable_memory_gate,
                )
                if gate_mean is not None:
                    update_gate_means.append(gate_mean)
            else:
                q = layer(q, context, context_mask=context_mask, disable_memory_gate=disable_memory_gate)
        if return_info:
            if update_gate_means:
                update_gate_mean = torch.stack(update_gate_means, dim=1)
            else:
                update_gate_mean = q.new_empty((b, 0))
            return q, {"update_gate_mean": update_gate_mean}
        return q
