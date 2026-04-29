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
    ):
        super().__init__()
        self.include_latents_in_kv = include_latents_in_kv
        self.cross_norm_q = RMSNorm(d_model)
        self.cross_norm_context = RMSNorm(d_model)
        self.cross = CrossAttention(d_model=d_model, n_heads=n_heads)
        self.out_norm = RMSNorm(d_model)
        self.ff_norm = RMSNorm(d_model) if ff_mult > 0 else None
        self.ff = SwiGLU(d_model, d_model * ff_mult) if ff_mult > 0 else None

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        latents = latents + self.cross(
            self.cross_norm_q(latents),
            self.cross_norm_context(kv),
            context_mask=kv_mask,
        )
        if self.ff is not None and self.ff_norm is not None:
            latents = latents + self.ff(self.ff_norm(latents))
        return self.out_norm(latents)


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
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("LatentWorkspace requires at least one layer")
        self.workspace_tokens = workspace_tokens
        self.workspace = nn.Parameter(torch.randn(workspace_tokens, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [
                LatentWorkspaceLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_mult=ff_mult,
                    include_latents_in_kv=include_latents_in_kv,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, context: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = context.shape[0]
        q = self.workspace.unsqueeze(0).expand(b, -1, -1)
        for layer in self.layers:
            q = layer(q, context, context_mask=context_mask)
        return q
