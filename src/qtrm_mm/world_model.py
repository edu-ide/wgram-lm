from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .norm import RMSNorm


def build_transition_mask(attention_mask: Optional[torch.Tensor], horizon: int, seq_len: int) -> torch.Tensor:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if seq_len <= horizon:
        if attention_mask is None:
            return torch.ones(1, 0, dtype=torch.bool)
        return attention_mask.new_zeros((attention_mask.shape[0], 0), dtype=torch.bool)
    if attention_mask is None:
        return torch.ones(1, seq_len - horizon, dtype=torch.bool)
    mask = attention_mask.to(torch.bool)
    return mask[:, :-horizon] & mask[:, horizon:]


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer from LeWorldModel."""

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        if knots < 2:
            raise ValueError("knots must be >= 2")
        if num_proj < 1:
            raise ValueError("num_proj must be >= 1")
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, emb: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if emb.numel() == 0:
            return emb.sum() * 0.0
        if emb.dim() != 3:
            raise ValueError("emb must have shape [B, T, D]")
        if mask is None:
            proj = emb.transpose(0, 1).float()
        else:
            valid = emb[mask.to(torch.bool)]
            if valid.shape[0] <= 1:
                return emb.sum() * 0.0
            proj = valid.unsqueeze(0).float()

        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        a = a / a.norm(p=2, dim=0).clamp_min(1e-6)
        t = self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)
        x_t = (proj @ a).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)

        attn_mask = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        if key_padding_mask is not None:
            pad_mask = key_padding_mask[:, None, None, :].to(torch.bool)
            attn_mask = attn_mask[None, None, :, :] | pad_mask
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.out(out.transpose(1, 2).contiguous().view(b, t, d))


class AdaLNZeroBlock(nn.Module):
    """LeWM-style causal transformer block with zero-initialized action conditioning."""

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = CausalSelfAttention(dim=dim, heads=heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(cond).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), key_padding_mask=key_padding_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class ActionConditionedFuturePredictor(nn.Module):
    """LeWM-style autoregressive next-embedding predictor adapted to QTRM latents."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_actions: int,
        predictor_layers: int,
        predictor_dim: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if predictor_dim % n_heads != 0:
            raise ValueError("predictor_dim must be divisible by n_heads")
        self.context_proj = nn.Linear(d_model, predictor_dim)
        self.action_proj = nn.Linear(num_actions, predictor_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, predictor_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                AdaLNZeroBlock(
                    dim=predictor_dim,
                    heads=n_heads,
                    mlp_dim=4 * predictor_dim,
                    dropout=dropout,
                )
                for _ in range(predictor_layers)
            ]
        )
        self.norm = nn.LayerNorm(predictor_dim)
        self.out_proj = nn.Linear(predictor_dim, d_model)

    def forward(
        self,
        context_latents: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, t, _ = context_latents.shape
        if t == 0:
            return context_latents
        if t > self.pos_embedding.shape[1]:
            raise ValueError("context length exceeds predictor max_seq_len")

        x = self.context_proj(context_latents) + self.pos_embedding[:, :t]
        x = self.dropout(x)
        if actions is None:
            cond = x.new_zeros(b, t, self.action_proj.in_features)
        else:
            if actions.dim() == 2:
                actions = actions[:, None, :].expand(b, t, -1)
            if actions.shape[:2] != (b, t):
                raise ValueError("actions must have shape [B, T, A] or [B, A]")
            cond = actions.to(x.dtype)
        cond = self.action_proj(cond)

        key_padding_mask = None
        if context_mask is not None:
            key_padding_mask = ~context_mask.to(torch.bool)
        for block in self.blocks:
            x = block(x, cond, key_padding_mask=key_padding_mask)
        return self.out_proj(self.norm(x))


class JepaWorldModelHead(nn.Module):
    """LeWM-style next-latent world model head for QTRM token latents."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_actions: int,
        predictor_layers: int = 2,
        predictor_dim: Optional[int] = None,
        max_seq_len: int = 512,
        horizon: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        predictor_dim = predictor_dim or d_model
        self.horizon = horizon
        self.online_norm = RMSNorm(d_model)
        self.predictor = ActionConditionedFuturePredictor(
            d_model=d_model,
            n_heads=n_heads,
            num_actions=num_actions,
            predictor_layers=predictor_layers,
            predictor_dim=predictor_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

    def forward(
        self,
        online_latents: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        b, t, d = online_latents.shape
        horizon = self.horizon
        if t <= horizon:
            empty = online_latents.new_zeros((b, 0, d))
            mask = online_latents.new_zeros((b, 0), dtype=torch.bool)
            latent_mask = online_latents.new_ones((b, t), dtype=torch.bool)
            if attention_mask is not None:
                latent_mask = attention_mask.to(torch.bool)
            return {"pred": empty, "target": empty, "latents": online_latents, "latent_mask": latent_mask, "mask": mask}

        latents = self.online_norm(online_latents)
        context = latents[:, :-horizon]
        target = latents[:, horizon:]
        if attention_mask is None:
            context_mask = torch.ones(b, t - horizon, dtype=torch.bool, device=latents.device)
            transition_mask = context_mask
            latent_mask = torch.ones(b, t, dtype=torch.bool, device=latents.device)
        else:
            valid = attention_mask.to(torch.bool)
            context_mask = valid[:, :-horizon]
            transition_mask = valid[:, :-horizon] & valid[:, horizon:]
            latent_mask = valid

        action_context = None
        if actions is not None:
            action_context = actions[:, :-horizon] if actions.dim() == 3 else actions

        pred = self.predictor(context, context_mask=context_mask, actions=action_context)
        return {"pred": pred, "target": target, "latents": latents, "latent_mask": latent_mask, "mask": transition_mask}
