from __future__ import annotations
import math
import torch


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]
    cos = cos[None, None, :, :].repeat_interleave(2, dim=-1)
    sin = sin[None, None, :, :].repeat_interleave(2, dim=-1)
    return x * cos + rotate_half(x) * sin
