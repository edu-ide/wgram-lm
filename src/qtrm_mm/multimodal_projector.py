from __future__ import annotations
from typing import Optional
import torch
from torch import nn

from .norm import RMSNorm
from .attention import CrossAttention


class MultimodalProjector(nn.Module):
    """Project visual features to model dimension and fuse with text context.

    visual_features: [B, V, visual_dim]
    text_states: [B, T, d_model]
    output: concatenated context states [B, T+V', d_model]
    """

    def __init__(self, d_model: int, visual_dim: int, max_visual_tokens: int, n_heads: int):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, d_model, bias=False)
        self.visual_pos = nn.Parameter(torch.randn(max_visual_tokens, d_model) * 0.02)
        self.norm = RMSNorm(d_model)
        self.max_visual_tokens = max_visual_tokens
        self.visual_resampler_tokens = nn.Parameter(torch.randn(max_visual_tokens, d_model) * 0.02)
        self.cross = CrossAttention(d_model=d_model, n_heads=n_heads)

    def forward(
        self,
        text_states: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if visual_features is None:
            return text_states, text_mask
        b, v, _ = visual_features.shape
        v = min(v, self.max_visual_tokens)
        visual = self.visual_proj(visual_features[:, :v])
        visual = visual + self.visual_pos[:v].unsqueeze(0)
        visual = self.norm(visual)
        # Keep simple: concatenate projected visual tokens.
        context = torch.cat([visual, text_states], dim=1)
        if text_mask is None:
            mask = None
        else:
            vmask = torch.ones(text_mask.shape[0], v, device=text_mask.device, dtype=text_mask.dtype)
            mask = torch.cat([vmask, text_mask], dim=1)
        return context, mask
