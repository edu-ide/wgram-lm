from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F

from .rotary import build_rope_cache, apply_rope


class GroupedQueryAttention(nn.Module):
    """GQA attention with optional FlashAttention fallback.

    This module uses PyTorch SDPA by default. If attention_backend='flash_attn', it
    attempts to use flash_attn_varlen_func-compatible code paths. The SDPA path is
    always available and trainable.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        causal: bool = True,
        backend: str = "sdpa",
        strict: bool = False,
    ):
        super().__init__()
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        if self.head_dim * n_heads != d_model:
            raise ValueError("d_model must be divisible by n_heads")
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.dropout_p = dropout
        self.causal = causal
        self.backend = backend
        self.strict = strict

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        if backend == "flash_attn":
            try:
                import flash_attn  # noqa: F401
                self._flash_available = True
            except Exception as exc:
                self._flash_available = False
                if strict:
                    raise RuntimeError("attention_backend=flash_attn requested, but flash_attn is unavailable") from exc
        else:
            self._flash_available = False

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = build_rope_cache(t, self.head_dim, self.rope_theta, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        repeat = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        mask = None
        if attention_mask is not None:
            # attention_mask: [B, T] with 1 for valid.
            # Convert to additive key mask [B, 1, 1, T].
            mask = (1.0 - attention_mask[:, None, None, :].to(q.dtype)) * torch.finfo(q.dtype).min
        dropout_p = self.dropout_p if self.training else 0.0
        if self.causal and mask is None:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
            out = out.transpose(1, 2).contiguous().view(b, t, d)
            return self.o_proj(out)

        if self.causal:
            causal = torch.zeros((t, t), dtype=q.dtype, device=q.device)
            causal = causal.masked_fill(
                torch.ones((t, t), dtype=torch.bool, device=q.device).triu(1),
                torch.finfo(q.dtype).min,
            )
            causal = causal[None, None, :, :]
            mask = causal if mask is None else mask + causal

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.o_proj(out)


class CrossAttention(nn.Module):
    """Workspace query attends to context keys/values."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, query: torch.Tensor, context: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, w, d = query.shape
        t = context.shape[1]
        q = self.q_proj(query).view(b, w, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        if context_mask is not None:
            mask = (1.0 - context_mask[:, None, None, :].to(q.dtype)) * torch.finfo(q.dtype).min
        else:
            mask = None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(b, w, d)
        return self.o_proj(out)


# =============================================================================
# Multi-Head Latent Attention (MLA) - Simplified version for hybrid experiments
# Inspired by DeepSeek-V2 MLA (arXiv:2405.04434)
#
# Key ideas implemented here (minimal viable for our scale):
# - Low-rank latent compression for KV (kv_lora_rank << d_model)
# - Separate projection for Q
# - Standard RoPE applied after latent expansion (for compatibility)
# - This gives better KV cache efficiency than GQA while keeping One-Body path.
#
# Note: Full DeepSeek MLA has "decoupled RoPE" and absorption tricks.
# This is a practical starting point for the Parallel Hybrid attention branch.
# =============================================================================

class MultiHeadLatentAttention(nn.Module):
    """Simplified Multi-Head Latent Attention for the hybrid block attention branch.

    Use this as a stronger alternative to plain GQA in the attention side of
    OneBodyParallelHybridBlock when we want better compression/quality.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kv_lora_rank: int = 64,           # Latent dimension for KV compression (main MLA knob)
        q_lora_rank: Optional[int] = None, # Optional low-rank for Q (can be None for full Q)
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        causal: bool = True,
        backend: str = "sdpa",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.dropout_p = dropout
        self.causal = causal

        # Q projection (can be low-rank or full)
        if q_lora_rank is not None:
            self.q_a_proj = nn.Linear(d_model, q_lora_rank, bias=False)
            self.q_b_proj = nn.Linear(q_lora_rank, n_heads * self.head_dim, bias=False)
        else:
            self.q_a_proj = None
            self.q_b_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)

        # KV latent compression (core of MLA)
        self.kv_a_proj = nn.Linear(d_model, kv_lora_rank, bias=False)           # input -> latent
        self.kv_b_proj = nn.Linear(kv_lora_rank, 2 * n_heads * self.head_dim, bias=False)  # latent -> K+V

        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, d = x.shape

        # Q path
        if self.q_a_proj is not None:
            q = self.q_b_proj(self.q_a_proj(x))
        else:
            q = self.q_b_proj(x)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        # KV latent path (MLA compression)
        kv_latent = self.kv_a_proj(x)                    # (B, T, kv_lora_rank)
        kv = self.kv_b_proj(kv_latent)                   # (B, T, 2 * n_heads * head_dim)

        # Split into K and V
        k = kv[..., :self.n_heads * self.head_dim].view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = kv[..., self.n_heads * self.head_dim:].view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE (simplified - full MLA often decouples this)
        cos, sin = build_rope_cache(t, self.head_dim, self.rope_theta, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Attention
        dropout_p = self.dropout_p if self.training else 0.0

        if self.causal:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=False,
            )

        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.o_proj(out)
