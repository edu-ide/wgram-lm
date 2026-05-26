from __future__ import annotations

import torch
from torch import nn


class NextImplicitByteProjector(nn.Module):
    """Predict the next supervised byte embedding from decoder hidden."""

    def __init__(self, d_model: int, hidden_dim: int = 0) -> None:
        super().__init__()
        width = int(hidden_dim)
        if width > 0:
            self.net = nn.Sequential(
                nn.Linear(int(d_model), width),
                nn.SiLU(),
                nn.Linear(width, int(d_model)),
            )
        else:
            self.net = nn.Linear(int(d_model), int(d_model))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class BLTDLocalDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        n_heads: int,
        dropout: float,
        cross_attention: bool,
    ) -> None:
        super().__init__()
        self.cross_attention = bool(cross_attention)
        self.self_norm = nn.LayerNorm(int(d_model))
        self.self_attn = nn.MultiheadAttention(
            int(d_model),
            int(n_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        if self.cross_attention:
            self.cross_norm = nn.LayerNorm(int(d_model))
            self.cross_attn = nn.MultiheadAttention(
                int(d_model),
                int(n_heads),
                dropout=float(dropout),
                batch_first=True,
            )
        else:
            self.cross_norm = None
            self.cross_attn = None
        self.ff_norm = nn.LayerNorm(int(d_model))
        self.ff = nn.Sequential(
            nn.Linear(int(d_model), int(d_model) * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model) * 4, int(d_model)),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        self_attn_mask: torch.Tensor | None,
        context: torch.Tensor | None,
        context_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.self_norm(x)
        self_out, _ = self.self_attn(
            h,
            h,
            h,
            attn_mask=self_attn_mask,
            need_weights=False,
        )
        x = x + self_out
        if self.cross_attention and context is not None:
            assert self.cross_norm is not None
            assert self.cross_attn is not None
            h = self.cross_norm(x)
            cross_out, _ = self.cross_attn(
                h,
                context,
                context,
                key_padding_mask=context_key_padding_mask,
                need_weights=False,
            )
            x = x + cross_out
        return x + self.ff(self.ff_norm(x))


class BLTDLocalDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        *,
        patch_size: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        causal: bool,
        cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.causal = bool(causal)
        self.cross_attention = bool(cross_attention)
        if self.cross_attention:
            self.layers = nn.ModuleList(
                [
                    BLTDLocalDecoderLayer(
                        d_model=int(d_model),
                        n_heads=int(n_heads),
                        dropout=float(dropout),
                        cross_attention=True,
                    )
                    for _ in range(int(n_layers))
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=int(d_model),
                        nhead=int(n_heads),
                        dim_feedforward=int(d_model) * 4,
                        dropout=float(dropout),
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(int(n_layers))
                ]
            )
        self.norm = nn.LayerNorm(int(d_model))
        self.head = nn.Linear(int(d_model), int(vocab_size), bias=False)

    def _causal_mask(self, device: torch.device) -> torch.Tensor | None:
        if not self.causal:
            return None
        mask = torch.full(
            (self.patch_size, self.patch_size),
            float("-inf"),
            device=device,
        )
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        context_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head(
            self.forward_hidden(
                x,
                context=context,
                context_key_padding_mask=context_key_padding_mask,
            )
        )

    def forward_hidden(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        context_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = self._causal_mask(x.device)
        h = x
        for layer in self.layers:
            if self.cross_attention:
                h = layer(
                    h,
                    self_attn_mask=mask,
                    context=context,
                    context_key_padding_mask=context_key_padding_mask,
                )
            else:
                h = layer(h, src_mask=mask)
        return self.norm(h)
