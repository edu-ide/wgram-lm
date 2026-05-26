from __future__ import annotations
from typing import Optional
from torch import nn
import torch

from .norm import RMSNorm
from .attention import GroupedQueryAttention
from .ffn import SwiGLU
from .mixers import build_delta_mixer
from .config import QTRMConfig


CANONICAL_LT2_ATTN_EVERY = 4


class QTRMBlock(nn.Module):
    """Hybrid Qwen/Kimi-style block.

    If use_attention=True, use GQA exact attention. Otherwise, use delta/recurrent
    mixer backend. The canonical LT2-style path is attn_every=4, giving three
    GatedDelta/GDN blocks followed by one full-attention sync block.
    """

    def __init__(self, cfg: QTRMConfig, use_attention: bool, causal: bool):
        super().__init__()
        self.use_attention = use_attention
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        if use_attention:
            self.mixer = GroupedQueryAttention(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                max_seq_len=cfg.max_seq_len,
                rope_theta=cfg.rope_theta,
                dropout=cfg.dropout,
                causal=causal,
                backend=cfg.attention_backend,
                strict=cfg.strict_backends,
            )
        else:
            self.mixer = build_delta_mixer(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                backend=cfg.delta_backend,
                strict=cfg.strict_backends,
                dropout=cfg.dropout,
                head_dim=cfg.delta_head_dim or (cfg.d_model // cfg.n_heads),
                num_v_heads=cfg.delta_num_v_heads or cfg.n_heads,
                expand_v=cfg.delta_expand_v,
                mode=cfg.delta_mode,
                use_short_conv=cfg.delta_use_short_conv,
                conv_size=cfg.delta_conv_size,
                norm_eps=cfg.delta_norm_eps,
            )
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff, dropout=cfg.dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x), attention_mask=attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class QTRMBlockStack(nn.Module):
    def __init__(self, cfg: QTRMConfig, n_layers: int, causal: bool, attn_every: int):
        super().__init__()
        layers = []
        for i in range(n_layers):
            use_attention = (i + 1) % attn_every == 0
            layers.append(QTRMBlock(cfg, use_attention=use_attention, causal=causal))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x
