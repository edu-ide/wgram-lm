#!/usr/bin/env python3
"""Minimal QTRM-native Encode-Think-Decode probe.

This is not a Qwen adapter and does not use donor hidden states. It tests the
pure native contract:

    tokens -> encode block -> repeated shared thinking block -> decode block
    -> LM logits -> autoregressive answer tokens

The task is a tiny synthetic modular-program language. It is intentionally
small enough to falsify the architecture path before any large pretraining.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from qtrm_mm.blocks import QTRMBlockStack
from qtrm_mm.config import QTRMConfig
from qtrm_mm.mixers import OfficialMamba3Mixer, build_delta_mixer
from qtrm_mm.training_optimizers import (
    MEMORY_EFFICIENT_OPTIMIZERS,
    build_memory_efficient_optimizer,
)


PAD = 0
BOS = 1
EOS = 2
START = 3
ANS = 4
OP_BASE = 5

OP_SPECS: tuple[tuple[str, int], ...] = (
    ("noop", 0),
    ("add", 1),
    ("add", 3),
    ("add", 5),
    ("mul", 2),
    ("mul", 3),
    ("affine", 2),
    ("affine", 3),
)
NOOP_OP_ID = 0


@dataclass(frozen=True)
class NativeCase:
    case_id: str
    start: int
    op_ids: tuple[int, ...]
    answer: int


def value_base() -> int:
    return OP_BASE + len(OP_SPECS)


def vocab_size(modulus: int) -> int:
    return value_base() + int(modulus)


def op_token(op_id: int) -> int:
    return OP_BASE + int(op_id)


def value_token(value: int) -> int:
    return value_base() + int(value)


def token_value(token_id: int) -> int | None:
    value = int(token_id) - value_base()
    return value if value >= 0 else None


def apply_op(value: int, op_id: int, modulus: int) -> int:
    name, param = OP_SPECS[int(op_id)]
    if name == "noop":
        return int(value) % int(modulus)
    if name == "add":
        return (int(value) + int(param)) % int(modulus)
    if name == "mul":
        return (int(value) * int(param)) % int(modulus)
    if name == "affine":
        return ((int(value) * int(param)) + int(param + 1)) % int(modulus)
    raise ValueError(f"unknown op: {name}")


def build_cases(
    *,
    count: int,
    seed: int,
    program_len: int,
    modulus: int,
) -> list[NativeCase]:
    rng = random.Random(int(seed))
    rows: list[NativeCase] = []
    for index in range(int(count)):
        start = rng.randrange(int(modulus))
        op_ids = tuple(rng.randrange(1, len(OP_SPECS)) for _ in range(int(program_len)))
        value = start
        for op_id in op_ids:
            value = apply_op(value, op_id, int(modulus))
        rows.append(
            NativeCase(
                case_id=f"native-{seed}-{index:06d}",
                start=start,
                op_ids=op_ids,
                answer=value,
            )
        )
    return rows


def case_with_active_program_len(
    case: NativeCase,
    *,
    active_len: int,
    modulus: int,
) -> NativeCase:
    active = max(0, min(int(active_len), len(case.op_ids)))
    op_ids = tuple(case.op_ids[:active]) + tuple(
        NOOP_OP_ID for _ in range(len(case.op_ids) - active)
    )
    value = int(case.start)
    for op_id in op_ids:
        value = apply_op(value, int(op_id), int(modulus))
    return NativeCase(
        case_id=f"{case.case_id}-active{active}",
        start=int(case.start),
        op_ids=op_ids,
        answer=int(value),
    )


def active_program_len_for_step(
    *,
    step: int,
    total_steps: int,
    program_len: int,
    min_active_len: int,
    warmup_fraction: float,
) -> int:
    full_len = max(1, int(program_len))
    min_len = max(1, min(int(min_active_len), full_len))
    warmup_steps = max(1, int(round(max(0.0, float(warmup_fraction)) * int(total_steps))))
    if int(step) >= warmup_steps:
        return full_len
    if warmup_steps <= 1 or full_len <= min_len:
        return full_len
    progress = max(0.0, min(1.0, (int(step) - 1) / max(1, warmup_steps - 1)))
    bucket_count = full_len - min_len + 1
    bucket = min(bucket_count - 1, int(progress * bucket_count))
    return min(full_len, min_len + bucket)


def case_prompt_tokens(case: NativeCase) -> list[int]:
    return [BOS, START, value_token(case.start)] + [
        op_token(op_id) for op_id in case.op_ids
    ] + [ANS]


def case_full_tokens(case: NativeCase) -> list[int]:
    return case_prompt_tokens(case) + [value_token(case.answer), EOS]


def cases_to_batch(
    cases: list[NativeCase],
    *,
    device: torch.device,
    ablation: str = "none",
) -> tuple[torch.Tensor, torch.Tensor]:
    prompts: list[list[int]] = []
    full: list[list[int]] = []
    for index, case in enumerate(cases):
        op_ids = list(case.op_ids)
        if ablation == "op_zero":
            op_ids = [NOOP_OP_ID for _ in op_ids]
        elif ablation == "op_shuffle":
            rng = random.Random(100000 + index)
            rng.shuffle(op_ids)
        elif ablation != "none":
            raise ValueError(f"unknown token ablation: {ablation}")
        ablated = NativeCase(
            case_id=case.case_id,
            start=case.start,
            op_ids=tuple(op_ids),
            answer=case.answer,
        )
        prompts.append(case_prompt_tokens(ablated))
        full.append(case_full_tokens(ablated))
    return (
        torch.tensor(prompts, dtype=torch.long, device=device),
        torch.tensor(full, dtype=torch.long, device=device),
    )


def depth_target_tokens(
    cases: list[NativeCase],
    *,
    max_depth: int,
    modulus: int,
    device: torch.device,
) -> torch.Tensor:
    rows: list[list[int]] = []
    for case in cases:
        value = int(case.start)
        row: list[int] = []
        for depth in range(max(1, int(max_depth))):
            if depth < len(case.op_ids):
                value = apply_op(value, int(case.op_ids[depth]), int(modulus))
            row.append(value_token(value))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long, device=device)


class NativeETDBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, *, causal_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm_attn(x)
        attn, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + attn
        x = x + self.ff(self.norm_ff(x))
        return x


class NativeMamba3Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, *, strict_backends: bool):
        super().__init__()
        self.norm_mixer = nn.LayerNorm(d_model)
        self.mixer = OfficialMamba3Mixer(
            d_model=d_model,
            n_heads=n_heads,
            strict=bool(strict_backends),
            fallback_dropout=dropout,
            d_state=max(64, int(d_model)),
            expand=2,
            headdim=max(16, int(d_model) // max(1, int(n_heads))),
            ngroups=1,
            chunk_size=64,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, *, causal_mask: torch.Tensor) -> torch.Tensor:
        del causal_mask
        x = x + self.mixer(self.norm_mixer(x), attention_mask=None)
        x = x + self.ff(self.norm_ff(x))
        return x


class NativeRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(d_model)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight.to(dtype=x.dtype)


class NativeSwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        hidden = max(1, int(d_ff))
        self.gate_up = nn.Linear(int(d_model), 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, int(d_model), bias=False)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.down(self.dropout(F.silu(gate) * value))


def _find_multiple(a: int, b: int) -> int:
    return (-(int(a) // -int(b))) * int(b)


class NativeOfficialSwiGLU(nn.Module):
    """SwiGLU width rule used by the official TRM implementation."""

    def __init__(self, d_model: int, expansion: float = 4.0) -> None:
        super().__init__()
        hidden = _find_multiple(round(float(expansion) * int(d_model) * 2 / 3), 256)
        self.gate_up = nn.Linear(int(d_model), 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, int(d_model), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = int(q.shape[1])
    head_dim = int(q.shape[-1])
    positions = torch.arange(seq_len, dtype=torch.float32, device=q.device)
    inv_freq = 1.0 / (
        float(theta)
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=q.device) / head_dim)
    )
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().view(1, seq_len, 1, head_dim)
    sin = emb.sin().view(1, seq_len, 1, head_dim)
    q_dtype = q.dtype
    k_dtype = k.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    return (
        (q * cos + _rotate_half(q) * sin).to(q_dtype),
        (k * cos + _rotate_half(k) * sin).to(k_dtype),
    )


class NativeOfficialAttention(nn.Module):
    """Official TRM attention projection layout with causal-LM masking."""

    def __init__(self, d_model: int, n_heads: int, *, rope_theta: float = 10000.0) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.head_dim = self.d_model // self.n_heads
        self.rope_theta = float(rope_theta)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, x: torch.Tensor, *, causal_mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3 * self.n_heads, self.head_dim)
        q = qkv[:, :, : self.n_heads]
        k = qkv[:, :, self.n_heads : 2 * self.n_heads]
        v = qkv[:, :, 2 * self.n_heads :]
        q, k = _apply_rotary_pos_emb(q, k, theta=self.rope_theta)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        mask = causal_mask.view(1, 1, seq_len, seq_len)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.permute(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        return self.out(out)


class NativeTRMOfficialBlock(nn.Module):
    """Official-TRM-style post-norm Attention/SwiGLU block for causal LM probes."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = NativeOfficialAttention(int(d_model), int(n_heads))
        self.mlp = NativeOfficialSwiGLU(int(d_model), expansion=4.0)
        self.norm_attn = NativeRMSNorm(int(d_model))
        self.norm_mlp = NativeRMSNorm(int(d_model))

    def forward(self, x: torch.Tensor, *, causal_mask: torch.Tensor) -> torch.Tensor:
        attn = self.attn(x, causal_mask=causal_mask)
        x = self.norm_attn(x + attn)
        x = self.norm_mlp(x + self.mlp(x))
        return x


class NativeTRMOfficialStack(nn.Module):
    """Official TRM L_level analogue: input injection followed by L_layers blocks.

    The official TRM config uses L_layers=2 and reuses that same L_level module
    for both z_L and z_H updates. Input injection is handled by the caller via
    `z_L + z_H + encoded` or `z_H + z_L`, matching the official call shape.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        *,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                NativeTRMOfficialBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(max(1, int(layers)))
            ]
        )

    def forward(self, x: torch.Tensor, *, causal_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, causal_mask=causal_mask)
        return x


class NativeTRMOfficialPreNormBlock(nn.Module):
    """Pre-norm variant of the official TRM block for causal LM stability ablations."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = NativeOfficialAttention(int(d_model), int(n_heads))
        self.mlp = NativeOfficialSwiGLU(int(d_model), expansion=4.0)
        self.norm_attn = NativeRMSNorm(int(d_model))
        self.norm_mlp = NativeRMSNorm(int(d_model))

    def forward(self, x: torch.Tensor, *, causal_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), causal_mask=causal_mask)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class NativeTRMOfficialPreNormStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        *,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                NativeTRMOfficialPreNormBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(max(1, int(layers)))
            ]
        )

    def forward(self, x: torch.Tensor, *, causal_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, causal_mask=causal_mask)
        return x


class NativeTRMGatedAttentionBlock(nn.Module):
    """Official-TRM attention block with learnable residual update strength."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            int(d_model),
            int(n_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.mlp = NativeSwiGLU(int(d_model), int(d_ff), float(dropout))
        self.norm_attn = NativeRMSNorm(int(d_model))
        self.norm_mlp = NativeRMSNorm(int(d_model))
        self.attn_gate_logit = nn.Parameter(torch.tensor(1.0))
        self.mlp_gate_logit = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, *, causal_mask: torch.Tensor) -> torch.Tensor:
        attn, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        attn_gate = torch.sigmoid(self.attn_gate_logit).to(dtype=x.dtype)
        x = self.norm_attn(x + attn_gate * attn)
        mlp_gate = torch.sigmoid(self.mlp_gate_logit).to(dtype=x.dtype)
        x = self.norm_mlp(x + mlp_gate * self.mlp(x))
        return x


class NativeTRMQwenAttentionBlock(nn.Module):
    """Qwen-style causal attention block: RMSNorm, bias-free QKV, QK-norm, SwiGLU."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        if int(d_model) % int(n_heads) != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.input_norm = NativeRMSNorm(self.d_model)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.q_norm = NativeRMSNorm(self.head_dim)
        self.k_norm = NativeRMSNorm(self.head_dim)
        self.attn_dropout = nn.Dropout(float(dropout))
        self.out_dropout = nn.Dropout(float(dropout))
        self.mlp_norm = NativeRMSNorm(self.d_model)
        self.mlp = NativeSwiGLU(self.d_model, int(d_ff), float(dropout))

    def forward(self, x: torch.Tensor, *, causal_mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        residual = x
        h = self.input_norm(x)
        q = self.q_proj(h).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        scores = scores + causal_mask.view(1, 1, seq_len, seq_len).to(dtype=scores.dtype)
        attn = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        attn = self.attn_dropout(attn)
        mixed = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        x = residual + self.out_dropout(self.o_proj(mixed))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class NativeTRMMixerLayer(nn.Module):
    """Official-TRM shell with a selectable causal mixer."""

    def __init__(
        self,
        *,
        mixer_kind: str,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        strict_backends: bool,
        delta_backend: str,
        delta_head_dim: int | None,
        delta_num_v_heads: int | None,
        delta_expand_v: float,
        delta_mode: str,
        delta_use_short_conv: bool,
        delta_conv_size: int,
        delta_norm_eps: float,
        norm_style: str = "post",
    ) -> None:
        super().__init__()
        self.mixer_kind = str(mixer_kind)
        self.norm_style = str(norm_style)
        if self.norm_style not in {"post", "pre"}:
            raise ValueError(f"unknown norm_style: {self.norm_style}")
        if self.mixer_kind == "attention":
            self.mixer = nn.MultiheadAttention(
                int(d_model),
                int(n_heads),
                dropout=float(dropout),
                batch_first=True,
            )
        elif self.mixer_kind == "mamba3":
            self.mixer = OfficialMamba3Mixer(
                d_model=int(d_model),
                n_heads=int(n_heads),
                strict=bool(strict_backends),
                fallback_dropout=float(dropout),
                d_state=max(64, int(d_model)),
                expand=2,
                headdim=max(16, int(d_model) // max(1, int(n_heads))),
                ngroups=1,
                chunk_size=64,
            )
        elif self.mixer_kind == "gated_delta":
            self.mixer = build_delta_mixer(
                d_model=int(d_model),
                n_heads=int(n_heads),
                backend=str(delta_backend),
                strict=bool(strict_backends),
                dropout=float(dropout),
                head_dim=int(delta_head_dim) if delta_head_dim else int(d_model) // max(1, int(n_heads)),
                num_v_heads=int(delta_num_v_heads) if delta_num_v_heads else int(n_heads),
                expand_v=float(delta_expand_v),
                mode=str(delta_mode),
                use_short_conv=bool(delta_use_short_conv),
                conv_size=int(delta_conv_size),
                norm_eps=float(delta_norm_eps),
            )
        else:
            raise ValueError(f"unknown TRM mixer kind: {self.mixer_kind}")
        self.mlp = NativeSwiGLU(int(d_model), int(d_ff), float(dropout))
        self.norm_mixer = NativeRMSNorm(int(d_model))
        self.norm_mlp = NativeRMSNorm(int(d_model))

    def forward(self, x: torch.Tensor, *, causal_mask: torch.Tensor) -> torch.Tensor:
        mixer_input = self.norm_mixer(x) if self.norm_style == "pre" else x
        if self.mixer_kind == "attention":
            mixed, _ = self.mixer(
                mixer_input,
                mixer_input,
                mixer_input,
                attn_mask=causal_mask,
                need_weights=False,
            )
        else:
            del causal_mask
            mixed = self.mixer(mixer_input, attention_mask=None)
        if self.norm_style == "pre":
            x = x + mixed
            x = x + self.mlp(self.norm_mlp(x))
            return x
        x = self.norm_mixer(x + mixed)
        x = self.norm_mlp(x + self.mlp(x))
        return x


class NativeTRMMixerBlock(nn.Module):
    """Stack multiple causal mixers inside the same official-TRM residual shell."""

    def __init__(
        self,
        *,
        mixer_kinds: tuple[str, ...],
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        strict_backends: bool,
        delta_backend: str,
        delta_head_dim: int | None,
        delta_num_v_heads: int | None,
        delta_expand_v: float,
        delta_mode: str,
        delta_use_short_conv: bool,
        delta_conv_size: int,
        delta_norm_eps: float,
        norm_style: str = "post",
    ) -> None:
        super().__init__()
        self.mixer_kinds = tuple(str(kind) for kind in mixer_kinds)
        self.layers = nn.ModuleList(
            [
                NativeTRMMixerLayer(
                    mixer_kind=kind,
                    d_model=int(d_model),
                    n_heads=int(n_heads),
                    d_ff=int(d_ff),
                    dropout=float(dropout),
                    strict_backends=bool(strict_backends),
                    delta_backend=str(delta_backend),
                    delta_head_dim=delta_head_dim,
                    delta_num_v_heads=delta_num_v_heads,
                    delta_expand_v=float(delta_expand_v),
                    delta_mode=str(delta_mode),
                    delta_use_short_conv=bool(delta_use_short_conv),
                    delta_conv_size=int(delta_conv_size),
                    delta_norm_eps=float(delta_norm_eps),
                    norm_style=str(norm_style),
                )
                for kind in self.mixer_kinds
            ]
        )

    def forward(self, x: torch.Tensor, *, causal_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, causal_mask=causal_mask)
        return x


class NativeTRMMamba3Block(NativeTRMMixerBlock):
    def __init__(self, **kwargs) -> None:
        super().__init__(mixer_kinds=("mamba3",), **kwargs)


class NativeTRMGatedDeltaBlock(NativeTRMMixerBlock):
    def __init__(self, **kwargs) -> None:
        super().__init__(mixer_kinds=("gated_delta",), **kwargs)


class NativeTRMQwen35HybridBlock(NativeTRMMixerBlock):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            mixer_kinds=("gated_delta", "gated_delta", "gated_delta", "attention"),
            **kwargs,
        )


class NativeTRMMamba3AttentionHybridBlock(NativeTRMMixerBlock):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            mixer_kinds=("mamba3", "mamba3", "mamba3", "attention"),
            **kwargs,
        )


class NativeTRMGatedDeltaAttentionHybridBlock(NativeTRMMixerBlock):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            mixer_kinds=("gated_delta", "gated_delta", "gated_delta", "attention"),
            **kwargs,
        )


class NativeTRMMamba3AttentionHybridPreNormBlock(NativeTRMMixerBlock):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            mixer_kinds=("mamba3", "mamba3", "mamba3", "attention"),
            norm_style="pre",
            **kwargs,
        )


class NativeTRMGatedDeltaAttentionHybridPreNormBlock(NativeTRMMixerBlock):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            mixer_kinds=("gated_delta", "gated_delta", "gated_delta", "attention"),
            norm_style="pre",
            **kwargs,
        )


class NativeTRMTriMixerBlock(NativeTRMMixerBlock):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            mixer_kinds=("gated_delta", "mamba3", "gated_delta", "attention"),
            **kwargs,
        )


SUPPORTED_BACKBONES = (
    "mha_etd",
    "qtrm_hybrid_3to1",
    "mamba3",
    "trm_official",
    "trm_official_prenorm",
    "trm_gated_attention",
    "trm_qwen_attention",
    "trm_mamba3",
    "trm_gated_delta",
    "trm_qwen35_3to1",
    "trm_tri_mixer",
)
SUPPORTED_CARRIER_STATE_MODES = (
    "gru",
    "encoded",
    "state_mean",
    "state_delta",
    "encoded_state_mean",
)
SUPPORTED_THINK_STRUCTURES = (
    "single",
    "single_core_carrier",
    "single_order_router",
    "single_order_router_residual_scale",
    "single_order_router_time_conditioned",
    "single_order_router_time_gate",
    "single_order_router_state_stream",
    "trm_dual_z",
    "trm_dual_z_hrm_separate",
    "trm_dual_z_interactive",
    "trm_dual_z_interactive_transition_gate",
    "trm_dual_z_diffusive",
    "trm_dual_z_diffusive_reversed_hybrid_3to1",
    "trm_dual_z_diffusive_reversed_hybrid_3to1_joint_readout",
    "trm_dual_z_reversed_hybrid_3to1",
    "trm_dual_z_reversed_hybrid_3to1_prenorm",
    "trm_dual_z_reversed_hybrid_3to1_joint_readout",
    "trm_dual_z_reversed_hybrid_3to1_core_gated_readout",
    "trm_dual_z_reversed_hybrid_3to1_semantic_carry",
    "trm_dual_z_reversed_hybrid_3to1_order_router",
    "trm_dual_z_reversed_hybrid_3to1_recent_order_router",
    "trm_dual_z_reversed_hybrid_3to1_state_gru_order_router",
    "trm_dual_z_reversed_hybrid_3to1_transition_state_order_router",
    "trm_dual_z_official_schedule_split_mixer_3to1",
    "trm_dual_z_reversed_mha_etd",
    "trm_dual_z_nested_reversed_mha_etd",
    "trm_dual_z_nested_reversed_mha_etd_joint_readout",
    "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
    "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
    "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
    "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
    "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router",
    "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router",
    "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router",
    "trm_dual_z_nested_reversed_hybrid_3to1",
    "trm_dual_z_nested_official_schedule_split_mixer_3to1",
    "trm_dual_z_interactive_residual_readout",
    "trm_dual_z_interactive_prefix_scratch",
    "trm_dual_z_interactive_core_carrier",
    "trm_dual_z_gated",
    "trm_dual_z_residual",
    "trm_dual_z_coupled",
    "trm_dual_z_coupled_residual",
    "trm_dual_z_coupled_delta_l_only",
    "trm_dual_z_coupled_mamba_h_only",
    "trm_dual_z_coupled_gated_proposal",
    "trm_dual_z_coupled_hybrid_router",
    "trm_dual_z_coupled_cross_attention",
    "trm_dual_z_coupled_step_conditioned_attention",
)
COUPLED_THINK_STRUCTURES = tuple(
    structure for structure in SUPPORTED_THINK_STRUCTURES if structure.startswith("trm_dual_z_coupled")
)
DUAL_Z_THINK_STRUCTURES = tuple(
    structure for structure in SUPPORTED_THINK_STRUCTURES if structure.startswith("trm_dual_z")
)
SINGLE_ORDER_ROUTER_THINK_STRUCTURES = (
    "single_order_router",
    "single_order_router_residual_scale",
    "single_order_router_time_conditioned",
    "single_order_router_time_gate",
    "single_order_router_state_stream",
)
TRM_NESTED_RECURRENT_LAYERSCALE_STRUCTURES = (
    "trm_dual_z_nested_reversed_mha_etd",
    "trm_dual_z_nested_reversed_mha_etd_joint_readout",
    "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
    "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
    "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
    "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
)


def applicable_ablation_names(think_structure: str) -> tuple[str, ...]:
    """Return only ablations that exist for the selected runtime structure."""
    names: list[str] = ["state_reset", "op_zero"]
    structure = str(think_structure)
    if (
        structure in COUPLED_THINK_STRUCTURES
        or structure
        in {
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router",
        }
    ):
        names.append("coupling_off")
    if structure in DUAL_Z_THINK_STRUCTURES:
        names.extend(["z_l_zero", "z_h_zero"])
    if structure in {
        "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
        "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
        "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
        "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router",
        "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router",
        "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router",
    }:
        names.append("carrier_off")
    return tuple(names)


class NativeQTRMETDLM(nn.Module):
    def __init__(
        self,
        *,
        vocab: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        backbone: str = "mha_etd",
        encode_backbone: str | None = None,
        think_backbone: str | None = None,
        decode_backbone: str | None = None,
        think_structure: str = "single",
        trm_l_cycles: int = 1,
        trm_no_grad_inner_cycles: bool = True,
        n_kv_heads: int = 2,
        hybrid_layers: int = 4,
        attn_every: int = 4,
        delta_backend: str = "torch_gated_delta",
        delta_head_dim: int | None = None,
        delta_num_v_heads: int | None = None,
        delta_expand_v: float = 1.0,
        delta_mode: str = "chunk",
        delta_use_short_conv: bool = True,
        delta_conv_size: int = 4,
        delta_norm_eps: float = 1e-6,
        attention_backend: str = "sdpa",
        strict_backends: bool = False,
        rope_theta: float = 100000.0,
        position_embedding_mode: str = "learned",
        op_order_embedding_mode: str = "none",
        op_order_max_positions: int = 32,
        op_token_ids: Iterable[int] | None = None,
        value_codec: str = "learned",
        value_token_ids: Iterable[int] | None = None,
        halt_pooling: str = "last",
        carrier_gate_init: float = -1.0,
        carrier_state_mode: str = "gru",
        trm_recurrent_layerscale_mode: str = "none",
        trm_recurrent_layerscale_init: float = 1.0,
        tie_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.vocab = int(vocab)
        self.max_seq_len = int(max_seq_len)
        self.backbone = str(backbone)
        self.encode_backbone = str(encode_backbone or backbone)
        self.think_backbone = str(think_backbone or backbone)
        self.decode_backbone = str(decode_backbone or backbone)
        self.think_structure = str(think_structure)
        self.trm_l_cycles = max(1, int(trm_l_cycles))
        self.trm_no_grad_inner_cycles = bool(trm_no_grad_inner_cycles)
        self.halt_pooling = str(halt_pooling)
        self.carrier_gate_init = float(carrier_gate_init)
        self.carrier_state_mode = str(carrier_state_mode)
        self.trm_recurrent_layerscale_mode = str(trm_recurrent_layerscale_mode)
        self.trm_recurrent_layerscale_init = float(trm_recurrent_layerscale_init)
        self.tie_embeddings = bool(tie_embeddings)
        self.delta_backend = str(delta_backend)
        self.delta_head_dim = int(delta_head_dim) if delta_head_dim else None
        self.delta_num_v_heads = int(delta_num_v_heads) if delta_num_v_heads else None
        self.delta_expand_v = float(delta_expand_v)
        self.delta_mode = str(delta_mode)
        self.delta_use_short_conv = bool(delta_use_short_conv)
        self.delta_conv_size = int(delta_conv_size)
        self.delta_norm_eps = float(delta_norm_eps)
        self.position_embedding_mode = str(position_embedding_mode)
        self.op_order_embedding_mode = str(op_order_embedding_mode)
        self.op_order_max_positions = max(1, int(op_order_max_positions))
        self.value_codec = str(value_codec)
        if value_token_ids is None:
            value_token_ids = range(value_base(), int(vocab))
        value_ids = tuple(int(token_id) for token_id in value_token_ids)
        if not value_ids and self.value_codec == "circular":
            raise ValueError("value_token_ids must not be empty for circular codec")
        invalid_value_ids = [
            token_id for token_id in value_ids if token_id < 0 or token_id >= int(vocab)
        ]
        if invalid_value_ids:
            raise ValueError(f"value_token_ids outside vocabulary: {invalid_value_ids}")
        if len(set(value_ids)) != len(value_ids):
            raise ValueError("value_token_ids must be unique")
        self.modulus = max(1, len(value_ids))
        value_lookup = torch.full((int(vocab),), -1, dtype=torch.long)
        for value_index, token_id in enumerate(value_ids):
            value_lookup[int(token_id)] = int(value_index)
        self.register_buffer(
            "_value_token_ids",
            torch.tensor(value_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("_value_id_lookup", value_lookup, persistent=False)
        if self.position_embedding_mode not in {"learned", "none", "randomized"}:
            raise ValueError(
                f"unknown position_embedding_mode: {self.position_embedding_mode}"
            )
        if self.op_order_embedding_mode not in {"none", "learned"}:
            raise ValueError(
                f"unknown op_order_embedding_mode: {self.op_order_embedding_mode}"
            )
        if self.value_codec not in {"learned", "circular"}:
            raise ValueError(f"unknown value_codec: {self.value_codec}")
        self.token_embed = nn.Embedding(int(vocab), int(d_model))
        self.pos_embed = nn.Embedding(int(max_seq_len), int(d_model))
        op_token_lookup = torch.zeros((int(vocab),), dtype=torch.bool)
        if op_token_ids is not None:
            for token_id in op_token_ids:
                if 0 <= int(token_id) < int(vocab):
                    op_token_lookup[int(token_id)] = True
        self.register_buffer("_op_token_lookup", op_token_lookup, persistent=False)
        if self.op_order_embedding_mode == "learned":
            self.op_order_embed = nn.Embedding(self.op_order_max_positions, int(d_model))
        self.value_feature_dim = 6
        if self.value_codec == "circular":
            self.value_code_proj = nn.Linear(self.value_feature_dim, int(d_model), bias=False)
            self.value_logit_scale = nn.Parameter(torch.tensor(1.0))
        if self.backbone not in set(SUPPORTED_BACKBONES):
            raise ValueError(f"unknown backbone: {self.backbone}")
        if self.think_structure not in set(SUPPORTED_THINK_STRUCTURES):
            raise ValueError(f"unknown think_structure: {self.think_structure}")
        if self.halt_pooling not in {"last", "mean", "dedicated"}:
            raise ValueError(f"unknown halt_pooling: {self.halt_pooling}")
        if self.carrier_state_mode not in SUPPORTED_CARRIER_STATE_MODES:
            raise ValueError(f"unknown carrier_state_mode: {self.carrier_state_mode}")
        if self.trm_recurrent_layerscale_mode not in {"none", "scalar", "channel"}:
            raise ValueError(
                "unknown trm_recurrent_layerscale_mode: "
                f"{self.trm_recurrent_layerscale_mode}"
            )
        stage_backbones = {
            self.encode_backbone,
            self.think_backbone,
            self.decode_backbone,
        }
        unknown = sorted(stage_backbones - set(SUPPORTED_BACKBONES))
        if unknown:
            raise ValueError(f"unknown stage backbone(s): {unknown}")
        cfg = None
        layers = max(1, int(hybrid_layers))
        if "qtrm_hybrid_3to1" in stage_backbones:
            cfg = QTRMConfig(
                vocab_size=int(vocab),
                d_model=int(d_model),
                n_heads=int(n_heads),
                n_kv_heads=int(n_kv_heads),
                d_ff=int(d_ff),
                max_seq_len=int(max_seq_len),
                dropout=float(dropout),
                rope_theta=float(rope_theta),
                attn_every=int(attn_every),
                delta_backend=str(delta_backend),
                delta_head_dim=int(delta_head_dim) if delta_head_dim else None,
                delta_num_v_heads=int(delta_num_v_heads) if delta_num_v_heads else None,
                delta_expand_v=float(delta_expand_v),
                delta_mode=str(delta_mode),
                delta_use_short_conv=bool(delta_use_short_conv),
                delta_conv_size=int(delta_conv_size),
                delta_norm_eps=float(delta_norm_eps),
                attention_backend=str(attention_backend),
                strict_backends=bool(strict_backends),
            )
        self.encode = self._build_stage(
            self.encode_backbone,
            d_model=int(d_model),
            n_heads=int(n_heads),
            d_ff=int(d_ff),
            dropout=float(dropout),
            cfg=cfg,
            layers=layers,
            attn_every=int(attn_every),
            strict_backends=bool(strict_backends),
        )
        self.think = self._build_stage(
            self.think_backbone,
            d_model=int(d_model),
            n_heads=int(n_heads),
            d_ff=int(d_ff),
            dropout=float(dropout),
            cfg=cfg,
            layers=layers,
            attn_every=int(attn_every),
            strict_backends=bool(strict_backends),
        )
        if self.think_structure == "trm_dual_z_hrm_separate":
            self.hrm_h_think = self._build_stage(
                self.think_backbone,
                d_model=int(d_model),
                n_heads=int(n_heads),
                d_ff=int(d_ff),
                dropout=float(dropout),
                cfg=cfg,
                layers=layers,
                attn_every=int(attn_every),
                strict_backends=bool(strict_backends),
            )
        self.decode = self._build_stage(
            self.decode_backbone,
            d_model=int(d_model),
            n_heads=int(n_heads),
            d_ff=int(d_ff),
            dropout=float(dropout),
            cfg=cfg,
            layers=layers,
            attn_every=int(attn_every),
            strict_backends=bool(strict_backends),
        )
        dual_think_structures = {
            "trm_dual_z",
            "trm_dual_z_hrm_separate",
            "trm_dual_z_interactive",
            "trm_dual_z_interactive_transition_gate",
            "trm_dual_z_diffusive",
            "trm_dual_z_diffusive_reversed_hybrid_3to1",
            "trm_dual_z_diffusive_reversed_hybrid_3to1_joint_readout",
            "trm_dual_z_reversed_hybrid_3to1",
            "trm_dual_z_reversed_hybrid_3to1_prenorm",
            "trm_dual_z_reversed_hybrid_3to1_joint_readout",
            "trm_dual_z_reversed_hybrid_3to1_core_gated_readout",
            "trm_dual_z_reversed_hybrid_3to1_semantic_carry",
            "trm_dual_z_reversed_hybrid_3to1_order_router",
            "trm_dual_z_reversed_hybrid_3to1_recent_order_router",
            "trm_dual_z_reversed_hybrid_3to1_state_gru_order_router",
            "trm_dual_z_reversed_hybrid_3to1_transition_state_order_router",
            "trm_dual_z_official_schedule_split_mixer_3to1",
            "trm_dual_z_reversed_mha_etd",
            "trm_dual_z_nested_reversed_mha_etd",
            "trm_dual_z_nested_reversed_mha_etd_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router",
            "trm_dual_z_nested_reversed_hybrid_3to1",
            "trm_dual_z_nested_official_schedule_split_mixer_3to1",
            "trm_dual_z_interactive_residual_readout",
            "trm_dual_z_interactive_prefix_scratch",
            "trm_dual_z_interactive_core_carrier",
            "trm_dual_z_gated",
            "trm_dual_z_residual",
            "trm_dual_z_coupled",
            "trm_dual_z_coupled_residual",
            "trm_dual_z_coupled_delta_l_only",
            "trm_dual_z_coupled_mamba_h_only",
            "trm_dual_z_coupled_gated_proposal",
            "trm_dual_z_coupled_hybrid_router",
            "trm_dual_z_coupled_cross_attention",
            "trm_dual_z_coupled_step_conditioned_attention",
        }
        if self.think_structure in dual_think_structures:
            self.z_l_init = nn.Parameter(torch.empty(1, 1, int(d_model)))
            self.z_h_init = nn.Parameter(torch.empty(1, 1, int(d_model)))
            self.trm_l_post_norm = nn.LayerNorm(int(d_model))
            self.trm_h_post_norm = nn.LayerNorm(int(d_model))
        if self.think_structure == "trm_dual_z_coupled_step_conditioned_attention":
            step_slots = max(64, int(max_seq_len) * 4)
            self.trm_step_embed = nn.Embedding(step_slots, int(d_model))
        diffusive_structures = {
            "trm_dual_z_diffusive",
            "trm_dual_z_diffusive_reversed_hybrid_3to1",
            "trm_dual_z_diffusive_reversed_hybrid_3to1_joint_readout",
        }
        if self.think_structure in diffusive_structures:
            step_slots = max(64, int(max_seq_len) * 4)
            self.trm_step_embed = nn.Embedding(step_slots, int(d_model))
            self.trm_diffusion_time_mlp = nn.Sequential(
                nn.Linear(2, int(d_model)),
                nn.SiLU(),
                nn.Linear(int(d_model), int(d_model)),
            )
            self.trm_diffusion_input_norm = nn.LayerNorm(int(d_model))
            self.trm_diffusion_l_update_gate = nn.Linear(3 * int(d_model), int(d_model))
            self.trm_diffusion_h_update_gate = nn.Linear(3 * int(d_model), int(d_model))
        if self.think_structure in {
            "trm_dual_z_reversed_hybrid_3to1",
            "trm_dual_z_reversed_hybrid_3to1_prenorm",
            "trm_dual_z_reversed_hybrid_3to1_joint_readout",
            "trm_dual_z_reversed_hybrid_3to1_core_gated_readout",
            "trm_dual_z_reversed_hybrid_3to1_semantic_carry",
            "trm_dual_z_reversed_hybrid_3to1_order_router",
            "trm_dual_z_reversed_hybrid_3to1_recent_order_router",
            "trm_dual_z_reversed_hybrid_3to1_state_gru_order_router",
            "trm_dual_z_reversed_hybrid_3to1_transition_state_order_router",
            "trm_dual_z_official_schedule_split_mixer_3to1",
            "trm_dual_z_reversed_mha_etd",
            "trm_dual_z_nested_reversed_mha_etd",
            "trm_dual_z_nested_reversed_mha_etd_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router",
            "trm_dual_z_nested_reversed_hybrid_3to1",
            "trm_dual_z_nested_official_schedule_split_mixer_3to1",
        }:
            self.trm_reversed_hybrid_input_norm = nn.LayerNorm(int(d_model))
            self.trm_reversed_hybrid_l_update_gate = nn.Linear(3 * int(d_model), int(d_model))
            self.trm_reversed_hybrid_h_update_gate = nn.Linear(3 * int(d_model), int(d_model))
        if self.think_structure in {
            "trm_dual_z_reversed_mha_etd",
            "trm_dual_z_nested_reversed_mha_etd",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router",
        }:
            self.trm_reversed_mha_readout_alpha = nn.Parameter(torch.tensor(0.0))
        if self.think_structure in {
            "trm_dual_z_nested_reversed_mha_etd",
            "trm_dual_z_nested_reversed_mha_etd_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router",
            "trm_dual_z_nested_reversed_hybrid_3to1",
            "trm_dual_z_nested_official_schedule_split_mixer_3to1",
        }:
            self.trm_nested_l_update_norm = nn.LayerNorm(4 * int(d_model))
            self.trm_nested_h_update_norm = nn.LayerNorm(4 * int(d_model))
            self.trm_nested_l_optimizer = nn.Sequential(
                nn.Linear(4 * int(d_model), int(d_model)),
                nn.SiLU(),
                nn.Linear(int(d_model), int(d_model)),
            )
            self.trm_nested_h_optimizer = nn.Sequential(
                nn.Linear(4 * int(d_model), int(d_model)),
                nn.SiLU(),
                nn.Linear(int(d_model), int(d_model)),
            )
            self.trm_nested_l_update_gate_logit = nn.Parameter(torch.tensor(-5.0))
            self.trm_nested_h_update_gate_logit = nn.Parameter(torch.tensor(-5.0))
        if (
            self.think_structure
            == "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange"
        ):
            self.trm_nested_cross_to_l_norm = nn.LayerNorm(int(d_model))
            self.trm_nested_cross_to_h_norm = nn.LayerNorm(int(d_model))
            self.trm_nested_cross_exchange_gate_logit = nn.Parameter(torch.tensor(-5.0))
        if (
            self.think_structure
            == "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned"
        ):
            step_slots = max(64, int(max_seq_len) * 4)
            self.trm_nested_step_embed = nn.Embedding(step_slots, int(d_model))
            self.trm_nested_step_l_update_norm = nn.LayerNorm(4 * int(d_model))
            self.trm_nested_step_h_update_norm = nn.LayerNorm(4 * int(d_model))
            self.trm_nested_step_l_optimizer = nn.Sequential(
                nn.Linear(4 * int(d_model), int(d_model)),
                nn.SiLU(),
                nn.Linear(int(d_model), int(d_model)),
            )
            self.trm_nested_step_h_optimizer = nn.Sequential(
                nn.Linear(4 * int(d_model), int(d_model)),
                nn.SiLU(),
                nn.Linear(int(d_model), int(d_model)),
            )
            self.trm_nested_step_update_gate_logit = nn.Parameter(torch.tensor(-5.0))
        if (
            self.think_structure
            in {
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router",
            }
        ):
            self.trm_nested_order_router = nn.Linear(int(d_model), 2)
        if (
            self.think_structure
            == "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router"
        ):
            self.trm_nested_route1_order_query_norm = nn.LayerNorm(int(d_model))
            self.trm_nested_route1_order_attn = nn.MultiheadAttention(
                int(d_model),
                int(n_heads),
                dropout=float(dropout),
                batch_first=True,
            )
            self.trm_nested_route1_order_norm = nn.LayerNorm(int(d_model))
            self.trm_nested_route1_order_gate_logit = nn.Parameter(torch.tensor(-1.0))
            self.trm_nested_route1_order_to_l = nn.Linear(
                int(d_model),
                int(d_model),
                bias=False,
            )
            self.trm_nested_route1_order_to_h = nn.Linear(
                int(d_model),
                int(d_model),
                bias=False,
            )
        if (
            self.trm_recurrent_layerscale_mode != "none"
            and self.think_structure in TRM_NESTED_RECURRENT_LAYERSCALE_STRUCTURES
        ):
            scale_shape = (
                (1, 1, int(d_model))
                if self.trm_recurrent_layerscale_mode == "channel"
                else (1,)
            )
            init = float(self.trm_recurrent_layerscale_init)
            self.trm_nested_l_recurrent_layerscale = nn.Parameter(
                torch.full(scale_shape, init)
            )
            self.trm_nested_h_recurrent_layerscale = nn.Parameter(
                torch.full(scale_shape, init)
            )
        if self.think_structure == "trm_dual_z_reversed_hybrid_3to1_semantic_carry":
            self.trm_semantic_l_carry_gate = nn.Linear(3 * int(d_model), int(d_model))
            self.trm_semantic_h_carry_gate = nn.Linear(3 * int(d_model), int(d_model))
        if self.think_structure in {
            "trm_dual_z_reversed_hybrid_3to1_order_router",
            "trm_dual_z_reversed_hybrid_3to1_recent_order_router",
            "trm_dual_z_reversed_hybrid_3to1_state_gru_order_router",
            "trm_dual_z_reversed_hybrid_3to1_transition_state_order_router",
        }:
            self.trm_order_router = nn.Linear(int(d_model), 2)
        if self.think_structure == "trm_dual_z_reversed_hybrid_3to1_state_gru_order_router":
            self.trm_order_state_gru = nn.GRU(
                int(d_model),
                int(d_model),
                num_layers=1,
                batch_first=True,
            )
            self.trm_order_state_norm = nn.LayerNorm(int(d_model))
        if self.think_structure == "trm_dual_z_reversed_hybrid_3to1_transition_state_order_router":
            self.trm_order_transition_seed = nn.Linear(int(d_model), int(d_model), bias=False)
            self.trm_order_transition_in = nn.Linear(3 * int(d_model), int(d_model))
            self.trm_order_transition_cell = nn.GRUCell(int(d_model), int(d_model))
            self.trm_order_transition_norm = nn.LayerNorm(int(d_model))
            self.trm_order_transition_gate_logit = nn.Parameter(torch.tensor(-1.0))
        if self.think_structure in {
            "trm_dual_z_diffusive_reversed_hybrid_3to1_joint_readout",
            "trm_dual_z_reversed_hybrid_3to1_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router",
            "trm_dual_z_reversed_hybrid_3to1_core_gated_readout",
        }:
            self.trm_joint_readout_norm = nn.LayerNorm(int(d_model))
            self.trm_joint_readout_proj = nn.Linear(3 * int(d_model), int(d_model), bias=False)
        if self.think_structure in {
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router",
        }:
            self.trm_nested_mha_joint_readout_alpha = nn.Parameter(torch.tensor(0.0))
        if self.think_structure == "trm_dual_z_reversed_hybrid_3to1_core_gated_readout":
            self.trm_core_readout_alpha = nn.Parameter(torch.tensor(0.25))
        if self.think_structure == "trm_dual_z_interactive_prefix_scratch":
            self.trm_scratch_proj = nn.Linear(3 * int(d_model), int(d_model), bias=False)
            self.trm_scratch_norm = nn.LayerNorm(int(d_model))
            self.trm_scratch_gate_logit = nn.Parameter(torch.tensor(-2.0))
        if self.think_structure == "trm_dual_z_interactive_core_carrier":
            self.trm_carrier_in = nn.Linear(3 * int(d_model), int(d_model), bias=False)
            self.trm_carrier_rnn = nn.GRU(
                int(d_model),
                int(d_model),
                num_layers=1,
                batch_first=True,
            )
            self.trm_carrier_norm = nn.LayerNorm(int(d_model))
            self.trm_carrier_gate_logit = nn.Parameter(torch.tensor(self.carrier_gate_init))
        if (
            self.think_structure
            in {
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router",
            }
        ):
            self.trm_nested_core_carrier_in = nn.Linear(
                3 * int(d_model),
                int(d_model),
                bias=False,
            )
            self.trm_nested_core_carrier_rnn = nn.GRU(
                int(d_model),
                int(d_model),
                num_layers=1,
                batch_first=True,
            )
            self.trm_nested_core_carrier_l_norm = nn.LayerNorm(int(d_model))
            self.trm_nested_core_carrier_h_norm = nn.LayerNorm(int(d_model))
            self.trm_nested_core_carrier_gate_logit = nn.Parameter(
                torch.tensor(self.carrier_gate_init)
            )
        if self.think_structure == "single_core_carrier":
            self.single_carrier_in = nn.Linear(2 * int(d_model), int(d_model), bias=False)
            self.single_carrier_rnn = nn.GRU(
                int(d_model),
                int(d_model),
                num_layers=1,
                batch_first=True,
            )
            self.single_carrier_norm = nn.LayerNorm(int(d_model))
            self.single_carrier_gate_logit = nn.Parameter(torch.tensor(self.carrier_gate_init))
        if self.think_structure in SINGLE_ORDER_ROUTER_THINK_STRUCTURES:
            self.trm_order_router = nn.Linear(int(d_model), 2)
            self.single_order_route1_gru = nn.GRU(
                int(d_model),
                int(d_model),
                num_layers=1,
                batch_first=True,
            )
            self.single_order_route1_context_norm = nn.LayerNorm(int(d_model))
            self.single_order_route1_input_norm = nn.LayerNorm(int(d_model))
            self.single_order_route1_update_gate = nn.Linear(3 * int(d_model), int(d_model))
            if self.think_structure == "single_order_router_time_conditioned":
                self.single_order_time_condition = nn.Linear(
                    2,
                    int(d_model),
                    bias=False,
                )
            if self.think_structure == "single_order_router_time_gate":
                self.single_order_time_gate = nn.Linear(
                    2,
                    int(d_model),
                    bias=False,
                )
            if self.think_structure == "single_order_router_state_stream":
                self.single_order_state_stream_in = nn.Linear(
                    2 * int(d_model),
                    int(d_model),
                    bias=False,
                )
                self.single_order_state_stream_norm = nn.LayerNorm(int(d_model))
            if self.think_structure == "single_order_router_residual_scale":
                # Preserve the previous router path at initialization while
                # allowing training to damp unstable recurrent deltas.
                self.single_order_recurrent_layerscale = nn.Parameter(
                    torch.ones(1, 1, int(d_model))
                )
        if self.think_structure == "trm_dual_z_interactive_transition_gate":
            self.trm_l_transition_gate_logit = nn.Parameter(torch.tensor(1.3862944))
            self.trm_h_transition_gate_logit = nn.Parameter(torch.tensor(1.3862944))
        if self.think_structure in {
            *diffusive_structures,
            "trm_dual_z_reversed_hybrid_3to1",
            "trm_dual_z_reversed_hybrid_3to1_prenorm",
            "trm_dual_z_reversed_hybrid_3to1_joint_readout",
            "trm_dual_z_reversed_hybrid_3to1_core_gated_readout",
            "trm_dual_z_reversed_hybrid_3to1_semantic_carry",
            "trm_dual_z_reversed_hybrid_3to1_order_router",
            "trm_dual_z_reversed_hybrid_3to1_recent_order_router",
            "trm_dual_z_reversed_hybrid_3to1_state_gru_order_router",
            "trm_dual_z_reversed_hybrid_3to1_transition_state_order_router",
            "trm_dual_z_official_schedule_split_mixer_3to1",
            "trm_dual_z_reversed_mha_etd",
            "trm_dual_z_nested_reversed_mha_etd",
            "trm_dual_z_nested_reversed_mha_etd_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router",
            "trm_dual_z_nested_reversed_hybrid_3to1",
            "trm_dual_z_nested_official_schedule_split_mixer_3to1",
        }:
            self.trm_init_l_proj = nn.Linear(int(d_model), int(d_model), bias=False)
            self.trm_init_h_proj = nn.Linear(int(d_model), int(d_model), bias=False)
        if self.think_structure in {
            "trm_dual_z_gated",
            "trm_dual_z_residual",
            "trm_dual_z_interactive_residual_readout",
            "trm_dual_z_coupled_residual",
        }:
            self.trm_init_l_proj = nn.Linear(int(d_model), int(d_model), bias=False)
            self.trm_init_h_proj = nn.Linear(int(d_model), int(d_model), bias=False)
            self.trm_readout_norm = nn.LayerNorm(int(d_model))
            self.trm_readout_alpha = nn.Parameter(torch.tensor(0.5))
        if self.think_structure == "trm_dual_z_gated":
            step_slots = max(64, int(max_seq_len) * 4)
            self.trm_step_embed = nn.Embedding(step_slots, int(d_model))
            self.trm_l_input_norm = nn.LayerNorm(int(d_model))
            self.trm_h_input_norm = nn.LayerNorm(int(d_model))
            self.trm_l_update_gate = nn.Linear(3 * int(d_model), int(d_model))
            self.trm_h_update_gate = nn.Linear(3 * int(d_model), int(d_model))
        coupled_no_readout_structures = {
            "trm_dual_z_coupled",
            "trm_dual_z_coupled_delta_l_only",
            "trm_dual_z_coupled_mamba_h_only",
            "trm_dual_z_coupled_gated_proposal",
            "trm_dual_z_coupled_hybrid_router",
            "trm_dual_z_coupled_cross_attention",
            "trm_dual_z_coupled_step_conditioned_attention",
        }
        if self.think_structure in coupled_no_readout_structures:
            self.trm_l_to_h = nn.Linear(int(d_model), int(d_model), bias=False)
            self.trm_h_to_l = nn.Linear(int(d_model), int(d_model), bias=False)
            self.trm_coupling_alpha = nn.Parameter(torch.tensor(0.05))
        if self.think_structure == "trm_dual_z_coupled_cross_attention":
            self.trm_l_cross_attn = nn.MultiheadAttention(
                int(d_model),
                int(n_heads),
                dropout=float(dropout),
                batch_first=True,
            )
            self.trm_h_cross_attn = nn.MultiheadAttention(
                int(d_model),
                int(n_heads),
                dropout=float(dropout),
                batch_first=True,
            )
            self.trm_cross_alpha = nn.Parameter(torch.tensor(0.05))
        proposal_structures = {
            "trm_dual_z_coupled_residual",
            "trm_dual_z_coupled_delta_l_only",
            "trm_dual_z_coupled_mamba_h_only",
            "trm_dual_z_coupled_gated_proposal",
            "trm_dual_z_coupled_hybrid_router",
            "trm_dual_z_diffusive_reversed_hybrid_3to1",
            "trm_dual_z_diffusive_reversed_hybrid_3to1_joint_readout",
            "trm_dual_z_reversed_hybrid_3to1",
            "trm_dual_z_reversed_hybrid_3to1_prenorm",
            "trm_dual_z_reversed_hybrid_3to1_joint_readout",
            "trm_dual_z_reversed_hybrid_3to1_core_gated_readout",
            "trm_dual_z_reversed_hybrid_3to1_semantic_carry",
            "trm_dual_z_reversed_hybrid_3to1_order_router",
            "trm_dual_z_reversed_hybrid_3to1_recent_order_router",
            "trm_dual_z_reversed_hybrid_3to1_state_gru_order_router",
            "trm_dual_z_reversed_hybrid_3to1_transition_state_order_router",
            "trm_dual_z_official_schedule_split_mixer_3to1",
            "trm_dual_z_nested_reversed_hybrid_3to1",
            "trm_dual_z_nested_official_schedule_split_mixer_3to1",
        }
        if self.think_structure in proposal_structures:
            proposal_kwargs = {
                "d_model": int(d_model),
                "n_heads": int(n_heads),
                "d_ff": int(d_ff),
                "dropout": float(dropout),
                "strict_backends": bool(strict_backends),
                "delta_backend": self.delta_backend,
                "delta_head_dim": self.delta_head_dim,
                "delta_num_v_heads": self.delta_num_v_heads,
                "delta_expand_v": self.delta_expand_v,
                "delta_mode": self.delta_mode,
                "delta_use_short_conv": self.delta_use_short_conv,
                "delta_conv_size": self.delta_conv_size,
                "delta_norm_eps": self.delta_norm_eps,
            }
        if self.think_structure in {
            "trm_dual_z_coupled_residual",
            "trm_dual_z_coupled_delta_l_only",
            "trm_dual_z_coupled_gated_proposal",
            "trm_dual_z_coupled_hybrid_router",
        }:
            self.trm_l_delta_proposal = NativeTRMGatedDeltaBlock(**proposal_kwargs)
            if self.think_structure not in {
                "trm_dual_z_coupled_gated_proposal",
                "trm_dual_z_coupled_hybrid_router",
            }:
                self.trm_delta_alpha = nn.Parameter(torch.tensor(0.05))
        if self.think_structure in {
            "trm_dual_z_coupled_residual",
            "trm_dual_z_coupled_mamba_h_only",
            "trm_dual_z_coupled_gated_proposal",
            "trm_dual_z_coupled_hybrid_router",
        }:
            self.trm_h_mamba_proposal = NativeTRMMamba3Block(**proposal_kwargs)
            if self.think_structure not in {
                "trm_dual_z_coupled_gated_proposal",
                "trm_dual_z_coupled_hybrid_router",
            }:
                self.trm_mamba_alpha = nn.Parameter(torch.tensor(0.05))
        if self.think_structure == "trm_dual_z_coupled_residual":
            self.trm_l_to_h = nn.Linear(int(d_model), int(d_model), bias=False)
            self.trm_h_to_l = nn.Linear(int(d_model), int(d_model), bias=False)
            self.trm_coupling_alpha = nn.Parameter(torch.tensor(0.05))
        if self.think_structure == "trm_dual_z_coupled_gated_proposal":
            self.trm_delta_gate_logit = nn.Parameter(torch.tensor(-4.0))
            self.trm_mamba_gate_logit = nn.Parameter(torch.tensor(-4.0))
        if self.think_structure == "trm_dual_z_coupled_hybrid_router":
            self.trm_l_hybrid_router = nn.Linear(3 * int(d_model), 3)
            self.trm_h_hybrid_router = nn.Linear(3 * int(d_model), 3)
        if self.think_structure in {
            "trm_dual_z_diffusive_reversed_hybrid_3to1",
            "trm_dual_z_diffusive_reversed_hybrid_3to1_joint_readout",
            "trm_dual_z_reversed_hybrid_3to1",
            "trm_dual_z_reversed_hybrid_3to1_joint_readout",
            "trm_dual_z_reversed_hybrid_3to1_core_gated_readout",
            "trm_dual_z_reversed_hybrid_3to1_semantic_carry",
            "trm_dual_z_reversed_hybrid_3to1_order_router",
            "trm_dual_z_reversed_hybrid_3to1_recent_order_router",
            "trm_dual_z_reversed_hybrid_3to1_state_gru_order_router",
            "trm_dual_z_reversed_hybrid_3to1_transition_state_order_router",
            "trm_dual_z_official_schedule_split_mixer_3to1",
            "trm_dual_z_nested_reversed_hybrid_3to1",
            "trm_dual_z_nested_official_schedule_split_mixer_3to1",
        }:
            self.trm_l_mamba3_attention_hybrid = NativeTRMMamba3AttentionHybridBlock(
                **proposal_kwargs
            )
            self.trm_h_gated_delta_attention_hybrid = NativeTRMGatedDeltaAttentionHybridBlock(
                **proposal_kwargs
            )
        if self.think_structure == "trm_dual_z_reversed_hybrid_3to1_prenorm":
            self.trm_l_mamba3_attention_hybrid = NativeTRMMamba3AttentionHybridPreNormBlock(
                **proposal_kwargs
            )
            self.trm_h_gated_delta_attention_hybrid = (
                NativeTRMGatedDeltaAttentionHybridPreNormBlock(**proposal_kwargs)
            )
        if self.halt_pooling == "dedicated":
            self.halt_state_init = nn.Linear(int(d_model), int(d_model))
            self.halt_state_update = nn.GRUCell(int(d_model), int(d_model))
            self.halt_state_norm = nn.LayerNorm(int(d_model))
        self.core_halt_head = nn.Linear(int(d_model), 1)
        self.norm = nn.LayerNorm(int(d_model))
        self.lm_head = nn.Linear(int(d_model), int(vocab), bias=False)
        if self.tie_embeddings:
            self.lm_head.weight = self.token_embed.weight
        self.reset_parameters()

    def _build_stage(
        self,
        stage_backbone: str,
        *,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        cfg: QTRMConfig | None,
        layers: int,
        attn_every: int,
        strict_backends: bool,
    ) -> nn.Module:
        if stage_backbone == "mha_etd":
            return NativeETDBlock(d_model, n_heads, d_ff, dropout)
        if stage_backbone == "trm_official":
            return NativeTRMOfficialStack(d_model, n_heads, d_ff, dropout, layers=2)
        if stage_backbone == "trm_official_prenorm":
            return NativeTRMOfficialPreNormStack(d_model, n_heads, d_ff, dropout, layers=2)
        if stage_backbone == "trm_gated_attention":
            return NativeTRMGatedAttentionBlock(d_model, n_heads, d_ff, dropout)
        if stage_backbone == "trm_qwen_attention":
            return NativeTRMQwenAttentionBlock(d_model, n_heads, d_ff, dropout)
        if stage_backbone == "mamba3":
            return NativeMamba3Block(
                d_model,
                n_heads,
                d_ff,
                dropout,
                strict_backends=bool(strict_backends),
            )
        trm_mixer_kwargs = {
            "d_model": int(d_model),
            "n_heads": int(n_heads),
            "d_ff": int(d_ff),
            "dropout": float(dropout),
            "strict_backends": bool(strict_backends),
            "delta_backend": self.delta_backend,
            "delta_head_dim": self.delta_head_dim,
            "delta_num_v_heads": self.delta_num_v_heads,
            "delta_expand_v": self.delta_expand_v,
            "delta_mode": self.delta_mode,
            "delta_use_short_conv": self.delta_use_short_conv,
            "delta_conv_size": self.delta_conv_size,
            "delta_norm_eps": self.delta_norm_eps,
        }
        if stage_backbone == "trm_mamba3":
            return NativeTRMMamba3Block(**trm_mixer_kwargs)
        if stage_backbone == "trm_gated_delta":
            return NativeTRMGatedDeltaBlock(**trm_mixer_kwargs)
        if stage_backbone == "trm_qwen35_3to1":
            return NativeTRMQwen35HybridBlock(**trm_mixer_kwargs)
        if stage_backbone == "trm_tri_mixer":
            return NativeTRMTriMixerBlock(**trm_mixer_kwargs)
        if cfg is None:
            raise ValueError("qtrm_hybrid_3to1 stage requires QTRMConfig")
        return QTRMBlockStack(cfg, layers, causal=True, attn_every=int(attn_every))

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        if hasattr(self, "op_order_embed"):
            nn.init.zeros_(self.op_order_embed.weight)
        if hasattr(self, "z_l_init"):
            nn.init.normal_(self.z_l_init, mean=0.0, std=0.02)
        if hasattr(self, "z_h_init"):
            nn.init.normal_(self.z_h_init, mean=0.0, std=0.02)
        if hasattr(self, "trm_step_embed"):
            nn.init.normal_(self.trm_step_embed.weight, mean=0.0, std=0.02)
        if hasattr(self, "trm_init_l_proj"):
            nn.init.eye_(self.trm_init_l_proj.weight)
        if hasattr(self, "trm_init_h_proj"):
            nn.init.eye_(self.trm_init_h_proj.weight)
        if hasattr(self, "trm_l_update_gate"):
            nn.init.zeros_(self.trm_l_update_gate.weight)
            nn.init.constant_(self.trm_l_update_gate.bias, -2.0)
        if hasattr(self, "trm_h_update_gate"):
            nn.init.zeros_(self.trm_h_update_gate.weight)
            nn.init.constant_(self.trm_h_update_gate.bias, -2.0)
        if hasattr(self, "trm_scratch_proj"):
            nn.init.xavier_uniform_(self.trm_scratch_proj.weight)
            self.trm_scratch_gate_logit.data.fill_(-2.0)
        if hasattr(self, "trm_carrier_in"):
            nn.init.xavier_uniform_(self.trm_carrier_in.weight)
            for name, parameter in self.trm_carrier_rnn.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
            self.trm_carrier_gate_logit.data.fill_(self.carrier_gate_init)
        if hasattr(self, "trm_nested_core_carrier_in"):
            nn.init.xavier_uniform_(self.trm_nested_core_carrier_in.weight)
            for name, parameter in self.trm_nested_core_carrier_rnn.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
            self.trm_nested_core_carrier_gate_logit.data.fill_(self.carrier_gate_init)
        if hasattr(self, "single_carrier_in"):
            nn.init.xavier_uniform_(self.single_carrier_in.weight)
            for name, parameter in self.single_carrier_rnn.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
            self.single_carrier_gate_logit.data.fill_(self.carrier_gate_init)
        if hasattr(self, "single_order_route1_gru"):
            for name, parameter in self.single_order_route1_gru.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
            nn.init.zeros_(self.single_order_route1_update_gate.weight)
            nn.init.constant_(self.single_order_route1_update_gate.bias, -6.0)
        if hasattr(self, "single_order_time_condition"):
            nn.init.zeros_(self.single_order_time_condition.weight)
        if hasattr(self, "single_order_time_gate"):
            nn.init.zeros_(self.single_order_time_gate.weight)
        if hasattr(self, "single_order_state_stream_in"):
            nn.init.zeros_(self.single_order_state_stream_in.weight)
        if hasattr(self, "single_order_recurrent_layerscale"):
            self.single_order_recurrent_layerscale.data.fill_(1.0)
        if hasattr(self, "trm_l_transition_gate_logit"):
            self.trm_l_transition_gate_logit.data.fill_(1.3862944)
        if hasattr(self, "trm_h_transition_gate_logit"):
            self.trm_h_transition_gate_logit.data.fill_(1.3862944)
        if hasattr(self, "trm_diffusion_time_mlp"):
            for module in self.trm_diffusion_time_mlp:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            nn.init.zeros_(self.trm_diffusion_l_update_gate.weight)
            nn.init.constant_(self.trm_diffusion_l_update_gate.bias, -1.0)
            nn.init.zeros_(self.trm_diffusion_h_update_gate.weight)
            nn.init.constant_(self.trm_diffusion_h_update_gate.bias, -1.0)
        if hasattr(self, "trm_reversed_hybrid_l_update_gate"):
            nn.init.zeros_(self.trm_reversed_hybrid_l_update_gate.weight)
            nn.init.constant_(self.trm_reversed_hybrid_l_update_gate.bias, -1.0)
            nn.init.zeros_(self.trm_reversed_hybrid_h_update_gate.weight)
            nn.init.constant_(self.trm_reversed_hybrid_h_update_gate.bias, -1.0)
            if self.think_structure in {
                "trm_dual_z_reversed_mha_etd",
                "trm_dual_z_nested_reversed_mha_etd",
                "trm_dual_z_nested_reversed_mha_etd_joint_readout",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router",
                "trm_dual_z_nested_reversed_hybrid_3to1",
                "trm_dual_z_nested_official_schedule_split_mixer_3to1",
            }:
                nn.init.constant_(self.trm_reversed_hybrid_l_update_gate.bias, -6.0)
                nn.init.constant_(self.trm_reversed_hybrid_h_update_gate.bias, -6.0)
        if hasattr(self, "trm_nested_l_optimizer"):
            for optimizer in (
                self.trm_nested_l_optimizer,
                self.trm_nested_h_optimizer,
            ):
                for module in optimizer:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        nn.init.zeros_(module.bias)
                last = optimizer[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)
            self.trm_nested_l_update_gate_logit.data.fill_(-5.0)
            self.trm_nested_h_update_gate_logit.data.fill_(-5.0)
        if hasattr(self, "trm_nested_step_l_optimizer"):
            nn.init.normal_(self.trm_nested_step_embed.weight, mean=0.0, std=0.02)
            for optimizer in (
                self.trm_nested_step_l_optimizer,
                self.trm_nested_step_h_optimizer,
            ):
                for module in optimizer:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        nn.init.zeros_(module.bias)
                last = optimizer[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)
            self.trm_nested_step_update_gate_logit.data.fill_(-5.0)
        if hasattr(self, "trm_semantic_l_carry_gate"):
            nn.init.zeros_(self.trm_semantic_l_carry_gate.weight)
            nn.init.constant_(self.trm_semantic_l_carry_gate.bias, -3.0)
            nn.init.zeros_(self.trm_semantic_h_carry_gate.weight)
            nn.init.constant_(self.trm_semantic_h_carry_gate.bias, -3.0)
        if hasattr(self, "trm_order_router"):
            nn.init.zeros_(self.trm_order_router.weight)
            bias = (
                torch.tensor([8.0, -8.0], dtype=self.trm_order_router.bias.dtype)
                if self.think_structure in SINGLE_ORDER_ROUTER_THINK_STRUCTURES
                else torch.tensor([2.0, -2.0], dtype=self.trm_order_router.bias.dtype)
            )
            self.trm_order_router.bias.data.copy_(bias)
        if hasattr(self, "trm_nested_order_router"):
            nn.init.zeros_(self.trm_nested_order_router.weight)
            self.trm_nested_order_router.bias.data.copy_(
                torch.tensor([2.0, -2.0], dtype=self.trm_nested_order_router.bias.dtype)
            )
        if hasattr(self, "trm_nested_route1_order_gate_logit"):
            self.trm_nested_route1_order_gate_logit.data.fill_(-1.0)
            nn.init.eye_(self.trm_nested_route1_order_to_l.weight)
            nn.init.eye_(self.trm_nested_route1_order_to_h.weight)
        if hasattr(self, "trm_order_transition_seed"):
            nn.init.xavier_uniform_(self.trm_order_transition_seed.weight)
            nn.init.xavier_uniform_(self.trm_order_transition_in.weight)
            nn.init.zeros_(self.trm_order_transition_in.bias)
            for name, parameter in self.trm_order_transition_cell.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
            self.trm_order_transition_gate_logit.data.fill_(-1.0)
        if hasattr(self, "trm_joint_readout_proj"):
            nn.init.xavier_uniform_(self.trm_joint_readout_proj.weight)
        if self.think_structure == "trm_dual_z_gated":
            self.trm_readout_alpha.data.fill_(0.1)
        elif self.think_structure in {
            "trm_dual_z_residual",
            "trm_dual_z_interactive_residual_readout",
        }:
            self.trm_readout_alpha.data.fill_(0.5)
        elif self.think_structure == "trm_dual_z_coupled_residual":
            self.trm_readout_alpha.data.fill_(0.5)
            nn.init.eye_(self.trm_l_to_h.weight)
            nn.init.eye_(self.trm_h_to_l.weight)
            self.trm_delta_alpha.data.fill_(0.05)
            self.trm_mamba_alpha.data.fill_(0.05)
            self.trm_coupling_alpha.data.fill_(0.05)
        elif self.think_structure == "trm_dual_z_coupled":
            nn.init.eye_(self.trm_l_to_h.weight)
            nn.init.eye_(self.trm_h_to_l.weight)
            self.trm_coupling_alpha.data.fill_(0.05)
        elif self.think_structure in {
            "trm_dual_z_coupled_delta_l_only",
            "trm_dual_z_coupled_mamba_h_only",
            "trm_dual_z_coupled_gated_proposal",
            "trm_dual_z_coupled_hybrid_router",
            "trm_dual_z_coupled_cross_attention",
            "trm_dual_z_coupled_step_conditioned_attention",
        }:
            nn.init.eye_(self.trm_l_to_h.weight)
            nn.init.eye_(self.trm_h_to_l.weight)
            self.trm_coupling_alpha.data.fill_(0.05)
            if hasattr(self, "trm_cross_alpha"):
                self.trm_cross_alpha.data.fill_(0.05)
            if hasattr(self, "trm_delta_alpha"):
                self.trm_delta_alpha.data.fill_(0.03)
            if hasattr(self, "trm_mamba_alpha"):
                self.trm_mamba_alpha.data.fill_(0.03)
            if hasattr(self, "trm_delta_gate_logit"):
                self.trm_delta_gate_logit.data.fill_(-4.0)
            if hasattr(self, "trm_mamba_gate_logit"):
                self.trm_mamba_gate_logit.data.fill_(-4.0)
            if hasattr(self, "trm_l_hybrid_router"):
                nn.init.zeros_(self.trm_l_hybrid_router.weight)
                nn.init.zeros_(self.trm_h_hybrid_router.weight)
                self.trm_l_hybrid_router.bias.data.copy_(
                    torch.tensor([2.0, -2.0, -2.0], dtype=self.trm_l_hybrid_router.bias.dtype)
                )
                self.trm_h_hybrid_router.bias.data.copy_(
                    torch.tensor([2.0, -2.0, -2.0], dtype=self.trm_h_hybrid_router.bias.dtype)
                )
        nn.init.zeros_(self.core_halt_head.weight)
        nn.init.constant_(self.core_halt_head.bias, -4.0)
        if self.tie_embeddings:
            nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        else:
            nn.init.xavier_uniform_(self.lm_head.weight)
        if hasattr(self, "value_code_proj"):
            nn.init.xavier_uniform_(self.value_code_proj.weight)

    def _value_features(self, device: torch.device) -> torch.Tensor:
        values = torch.arange(int(self.modulus), dtype=torch.float32, device=device)
        angle = (2.0 * math.pi * values) / float(max(1, int(self.modulus)))
        rows = []
        for frequency in (1.0, 2.0, 4.0):
            rows.append(torch.sin(angle * frequency))
            rows.append(torch.cos(angle * frequency))
        return torch.stack(rows, dim=-1)

    def _circular_value_embeddings(self, device: torch.device) -> torch.Tensor:
        if not hasattr(self, "value_code_proj"):
            raise RuntimeError("circular value embeddings requested for learned codec")
        return self.value_code_proj(self._value_features(device))

    def _op_order_embeddings(
        self,
        input_ids: torch.Tensor,
        *,
        op_order_off: bool = False,
    ) -> torch.Tensor | None:
        if self.op_order_embedding_mode == "none" or bool(op_order_off):
            return None
        if not hasattr(self, "op_order_embed"):
            return None
        op_mask = self._op_token_lookup.to(device=input_ids.device)[input_ids]
        if not bool(op_mask.any()):
            return None
        op_index = op_mask.long().cumsum(dim=1) - 1
        op_index = op_index.clamp(0, int(self.op_order_max_positions) - 1)
        order_embeddings = self.op_order_embed(op_index)
        return order_embeddings * op_mask.unsqueeze(-1).to(dtype=order_embeddings.dtype)

    def _token_embeddings(
        self,
        input_ids: torch.Tensor,
        *,
        op_order_off: bool = False,
    ) -> torch.Tensor:
        embeddings = self.token_embed(input_ids)
        if self.value_codec == "circular":
            value_ids_for_tokens = self._value_id_lookup.to(device=input_ids.device)[input_ids]
            value_mask = value_ids_for_tokens >= 0
            if bool(value_mask.any()):
                value_ids = value_ids_for_tokens[value_mask].clamp(0, int(self.modulus) - 1)
                value_embeddings = self._circular_value_embeddings(input_ids.device)
                embeddings = embeddings.clone()
                embeddings[value_mask] = value_embeddings[value_ids]
        op_order_embeddings = self._op_order_embeddings(
            input_ids,
            op_order_off=bool(op_order_off),
        )
        if op_order_embeddings is not None:
            embeddings = embeddings + op_order_embeddings
        return embeddings

    def _lm_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden)
        if self.value_codec != "circular":
            return logits
        value_embeddings = F.normalize(self._circular_value_embeddings(hidden.device), dim=-1)
        value_hidden = F.normalize(hidden, dim=-1)
        value_logits = torch.matmul(value_hidden, value_embeddings.t())
        value_logits = value_logits * self.value_logit_scale.exp().clamp(max=100.0)
        logits = logits.clone()
        logits.index_copy_(
            -1,
            self._value_token_ids.to(device=hidden.device),
            value_logits,
        )
        return logits

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def _position_ids(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.position_embedding_mode == "randomized" and self.training:
            if int(seq_len) < int(self.max_seq_len):
                sampled = torch.randperm(int(self.max_seq_len), device=device)[: int(seq_len)]
                sampled = sampled.sort().values
                return sampled.unsqueeze(0)
        return torch.arange(int(seq_len), device=device).unsqueeze(0)

    def _run_stage(
        self,
        stage: nn.Module,
        x: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(
            stage,
            (
                NativeETDBlock,
                NativeMamba3Block,
                NativeTRMOfficialBlock,
                NativeTRMOfficialStack,
                NativeTRMOfficialPreNormBlock,
                NativeTRMOfficialPreNormStack,
                NativeTRMGatedAttentionBlock,
                NativeTRMQwenAttentionBlock,
                NativeTRMMixerBlock,
            ),
        ):
            return stage(x, causal_mask=causal_mask)
        return stage(x, attention_mask=None)

    def _initial_trm_state(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = encoded.shape
        z_l = self.z_l_init.to(dtype=encoded.dtype).expand(batch, seq_len, -1)
        z_h = self.z_h_init.to(dtype=encoded.dtype).expand(batch, seq_len, -1)
        if self.think_structure in {
            "trm_dual_z_reversed_mha_etd",
            "trm_dual_z_nested_reversed_mha_etd",
            "trm_dual_z_nested_reversed_mha_etd_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router",
        }:
            return encoded + z_l, encoded + z_h
        if self.think_structure in {
            "trm_dual_z_diffusive",
            "trm_dual_z_diffusive_reversed_hybrid_3to1",
            "trm_dual_z_diffusive_reversed_hybrid_3to1_joint_readout",
            "trm_dual_z_reversed_hybrid_3to1",
            "trm_dual_z_reversed_hybrid_3to1_prenorm",
            "trm_dual_z_reversed_hybrid_3to1_joint_readout",
            "trm_dual_z_reversed_hybrid_3to1_core_gated_readout",
            "trm_dual_z_reversed_hybrid_3to1_semantic_carry",
            "trm_dual_z_reversed_hybrid_3to1_order_router",
            "trm_dual_z_reversed_hybrid_3to1_recent_order_router",
            "trm_dual_z_reversed_hybrid_3to1_state_gru_order_router",
            "trm_dual_z_reversed_hybrid_3to1_transition_state_order_router",
            "trm_dual_z_official_schedule_split_mixer_3to1",
            "trm_dual_z_reversed_mha_etd",
            "trm_dual_z_nested_reversed_mha_etd",
            "trm_dual_z_nested_reversed_mha_etd_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router",
            "trm_dual_z_nested_reversed_hybrid_3to1",
            "trm_dual_z_nested_official_schedule_split_mixer_3to1",
            "trm_dual_z_gated",
            "trm_dual_z_residual",
        }:
            z_l = z_l + self.trm_init_l_proj(encoded)
            z_h = z_h + self.trm_init_h_proj(encoded)
        return z_l, z_h

    def _initial_halt_state(self, encoded: torch.Tensor) -> torch.Tensor:
        pooled = encoded.mean(dim=1)
        if self.halt_pooling == "dedicated":
            return self.halt_state_norm(torch.tanh(self.halt_state_init(pooled)))
        return pooled

    def _update_halt_state(
        self,
        halt_state: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        pooled = state.mean(dim=1)
        if self.halt_pooling == "dedicated":
            return self.halt_state_norm(self.halt_state_update(pooled, halt_state))
        return pooled

    def _core_halt_logits(
        self,
        state: torch.Tensor,
        *,
        halt_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.halt_pooling == "dedicated":
            pooled = halt_state if halt_state is not None else self._initial_halt_state(state)
        elif self.halt_pooling == "mean":
            pooled = state.mean(dim=1)
        else:
            pooled = state[:, -1, :]
        return self.core_halt_head(pooled).squeeze(-1)

    def _trm_step_state(
        self,
        encoded: torch.Tensor,
        *,
        h_step: int,
        l_step: int,
    ) -> torch.Tensor:
        step_count = int(self.trm_step_embed.num_embeddings)
        step_id = (int(h_step) * (self.trm_l_cycles + 1) + int(l_step)) % step_count
        step = torch.full(
            (encoded.shape[1],),
            step_id,
            dtype=torch.long,
            device=encoded.device,
        )
        return self.trm_step_embed(step).unsqueeze(0).expand(encoded.shape[0], -1, -1).to(dtype=encoded.dtype)

    def _trm_diffusion_context(
        self,
        encoded: torch.Tensor,
        *,
        h_step: int,
        total_steps: int,
        l_step: int,
    ) -> torch.Tensor:
        total = max(1, int(total_steps))
        substeps = max(1, self.trm_l_cycles + 1)
        substep = min(substeps - 1, max(0, int(l_step)))
        progress = (float(h_step) + float(substep + 1) / float(substeps)) / float(total)
        progress = max(0.0, min(1.0, progress))
        noise_level = 1.0 - progress
        time_features = torch.tensor(
            [[progress, noise_level]],
            dtype=encoded.dtype,
            device=encoded.device,
        )
        time_state = self.trm_diffusion_time_mlp(time_features).view(1, 1, -1)
        step_state = self._trm_step_state(
            encoded,
            h_step=int(h_step),
            l_step=int(l_step),
        )
        return step_state + time_state.expand(encoded.shape[0], encoded.shape[1], -1)

    def _gated_trm_update(
        self,
        previous: torch.Tensor,
        candidate: torch.Tensor,
        context: torch.Tensor,
        *,
        gate: nn.Linear,
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        update_gate = torch.sigmoid(gate(torch.cat([previous, candidate, context], dim=-1)))
        return norm(previous + update_gate * (candidate - previous))

    def _apply_trm_recurrent_layerscale(
        self,
        previous: torch.Tensor,
        updated: torch.Tensor,
        *,
        state_name: str,
    ) -> torch.Tensor:
        if self.trm_recurrent_layerscale_mode == "none":
            return updated
        parameter_name = f"trm_nested_{state_name}_recurrent_layerscale"
        if not hasattr(self, parameter_name):
            return updated
        scale = getattr(self, parameter_name).to(dtype=updated.dtype)
        return previous + scale * (updated - previous)

    def _apply_semantic_carry(
        self,
        state: torch.Tensor,
        anchor: torch.Tensor,
        context: torch.Tensor,
        *,
        gate: nn.Linear,
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        carry_gate = torch.sigmoid(gate(torch.cat([state, anchor, context], dim=-1)))
        return norm(state + carry_gate * (anchor - state))

    def _apply_prefix_scratch(
        self,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        encoded: torch.Tensor,
        *,
        z_h_zero: bool = False,
    ) -> torch.Tensor:
        if bool(z_h_zero):
            return torch.zeros_like(z_h)
        scratch_input = torch.cat([z_l, z_h, encoded], dim=-1)
        scratch_delta = self.trm_scratch_proj(scratch_input)
        positions = (
            torch.arange(1, scratch_delta.shape[1] + 1, device=scratch_delta.device)
            .view(1, -1, 1)
            .to(dtype=scratch_delta.dtype)
        )
        causal_prefix = scratch_delta.cumsum(dim=1) / positions
        gate = torch.sigmoid(self.trm_scratch_gate_logit).to(dtype=z_h.dtype)
        return self.trm_scratch_norm(z_h + gate * causal_prefix)

    def _apply_core_carrier(
        self,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        encoded: torch.Tensor,
        *,
        z_h_zero: bool = False,
    ) -> torch.Tensor:
        if bool(z_h_zero):
            return torch.zeros_like(z_h)
        carrier_input = self.trm_carrier_in(torch.cat([z_l, z_h, encoded], dim=-1))
        carrier_state, _ = self.trm_carrier_rnn(torch.tanh(carrier_input))
        gate = torch.sigmoid(self.trm_carrier_gate_logit).to(dtype=z_h.dtype)
        return self.trm_carrier_norm(z_h + gate * carrier_state.to(dtype=z_h.dtype))

    def _apply_nested_core_carrier(
        self,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        encoded: torch.Tensor,
        *,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
        carrier_off: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(carrier_off):
            return z_l, z_h
        if self.carrier_state_mode == "gru":
            carrier_input = self.trm_nested_core_carrier_in(
                torch.cat([z_l, z_h, encoded], dim=-1)
            )
            carrier_state, _ = self.trm_nested_core_carrier_rnn(torch.tanh(carrier_input))
        elif self.carrier_state_mode == "encoded":
            carrier_state = encoded
        elif self.carrier_state_mode == "state_mean":
            carrier_state = 0.5 * (z_l + z_h)
        elif self.carrier_state_mode == "state_delta":
            carrier_state = z_l - z_h
        elif self.carrier_state_mode == "encoded_state_mean":
            carrier_state = (encoded + z_l + z_h) / 3.0
        else:
            raise ValueError(f"unknown carrier_state_mode: {self.carrier_state_mode}")
        carrier_state = carrier_state.to(dtype=z_l.dtype)
        gate = torch.sigmoid(self.trm_nested_core_carrier_gate_logit).to(dtype=z_l.dtype)
        if bool(z_l_zero):
            next_l = torch.zeros_like(z_l)
        else:
            l_delta = self.trm_nested_core_carrier_l_norm(carrier_state)
            next_l = z_l + gate * l_delta
        if bool(z_h_zero):
            next_h = torch.zeros_like(z_h)
        else:
            h_delta = self.trm_nested_core_carrier_h_norm(carrier_state)
            next_h = z_h + gate * h_delta
        return next_l, next_h

    def _nested_step_context(
        self,
        encoded: torch.Tensor,
        *,
        h_step: int,
    ) -> torch.Tensor:
        step_index = min(
            max(0, int(h_step)),
            int(self.trm_nested_step_embed.num_embeddings) - 1,
        )
        step_ids = torch.full(
            (int(encoded.shape[0]), int(encoded.shape[1])),
            step_index,
            dtype=torch.long,
            device=encoded.device,
        )
        return self.trm_nested_step_embed(step_ids).to(dtype=encoded.dtype)

    def _apply_single_core_carrier(
        self,
        state: torch.Tensor,
        encoded: torch.Tensor,
    ) -> torch.Tensor:
        carrier_input = self.single_carrier_in(torch.cat([state, encoded], dim=-1))
        carrier_state, _ = self.single_carrier_rnn(torch.tanh(carrier_input))
        gate = torch.sigmoid(self.single_carrier_gate_logit).to(dtype=state.dtype)
        return self.single_carrier_norm(state + gate * carrier_state.to(dtype=state.dtype))

    def _single_order_route1_context(self, encoded: torch.Tensor) -> torch.Tensor:
        """Causal suffix-biased context for reverse-order recurrent transitions.

        The GRU runs left-to-right over the token stream, so the answer logit at
        a position only sees previous input tokens. The route is still a normal
        LM path; it does not compute or inject the answer.
        """
        route_state, _ = self.single_order_route1_gru(encoded)
        alpha = 0.75
        running = encoded[:, 0, :]
        rows = [running]
        for index in range(1, int(encoded.shape[1])):
            running = alpha * running + (1.0 - alpha) * encoded[:, index, :]
            rows.append(running)
        recent = torch.stack(rows, dim=1)
        return self.single_order_route1_context_norm(
            encoded + route_state.to(dtype=encoded.dtype) + recent.to(dtype=encoded.dtype)
        )

    def _run_single_order_router_step(
        self,
        state: torch.Tensor,
        encoded: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        step_index: int = 0,
        total_steps: int = 1,
    ) -> torch.Tensor:
        route0 = self._run_stage(self.think, state, causal_mask=causal_mask)
        route1_context = self._single_order_route1_context(encoded)
        if hasattr(self, "single_order_state_stream_in"):
            shifted_state = torch.zeros_like(state)
            shifted_state[:, 1:, :] = state[:, :-1, :]
            state_stream = self.single_order_state_stream_norm(
                self.single_order_state_stream_in(
                    torch.cat([encoded, shifted_state], dim=-1)
                )
            )
            route1_context = route1_context + state_stream.to(
                dtype=route1_context.dtype
            )
        progress = None
        if hasattr(self, "single_order_time_condition") or hasattr(
            self,
            "single_order_time_gate",
        ):
            denom = max(1, int(total_steps))
            progress = encoded.new_tensor(
                [float(step_index + 1) / float(denom), 1.0 / float(denom)]
            )
        if hasattr(self, "single_order_time_condition"):
            assert progress is not None
            time_bias = self.single_order_time_condition(progress).view(1, 1, -1)
            route1_context = route1_context + time_bias.to(dtype=route1_context.dtype)
        route1_candidate = self._run_stage(
            self.think,
            self.single_order_route1_input_norm(state + route1_context),
            causal_mask=causal_mask,
        )
        route1_gate_logits = self.single_order_route1_update_gate(
            torch.cat([route0, route1_candidate, route1_context], dim=-1)
        )
        if hasattr(self, "single_order_time_gate"):
            assert progress is not None
            time_gate = self.single_order_time_gate(progress).view(1, 1, -1)
            route1_gate_logits = route1_gate_logits + time_gate.to(
                dtype=route1_gate_logits.dtype
            )
        route1_gate = torch.sigmoid(route1_gate_logits)
        route1 = route0 + route1_gate * (route1_candidate - route0)
        force_route = getattr(self, "trm_order_router_force_route", None)
        if force_route is None:
            route = torch.softmax(self.trm_order_router(encoded), dim=-1)
        else:
            route_id = int(force_route)
            if route_id not in {0, 1}:
                raise ValueError("trm_order_router_force_route must be 0, 1, or None")
            route = encoded.new_zeros((*encoded.shape[:-1], 2))
            route[..., route_id] = 1.0
        route0_weight = route[..., 0:1].to(dtype=route0.dtype)
        route1_weight = route[..., 1:2].to(dtype=route1.dtype)
        mixed = route0_weight * route0 + route1_weight * route1
        if hasattr(self, "single_order_recurrent_layerscale"):
            scale = self.single_order_recurrent_layerscale.to(dtype=mixed.dtype)
            return state + scale * (mixed - state)
        return mixed

    def _run_trm_h_cycle(
        self,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        encoded: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(z_l_zero):
            z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        for _ in range(self.trm_l_cycles):
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
            if bool(z_h_zero):
                z_h = torch.zeros_like(z_h)
            z_l = self._run_stage(
                self.think,
                z_l + z_h + encoded,
                causal_mask=causal_mask,
            )
            z_l = self.trm_l_post_norm(z_l)
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        z_h = self._run_stage(
            self.think,
            z_h + z_l,
            causal_mask=causal_mask,
        )
        z_h = self.trm_h_post_norm(z_h)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        return z_l, z_h

    def _run_trm_hrm_separate_h_cycle(
        self,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        encoded: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """HRM-style dual-state update with separate L and H recurrent modules."""
        if bool(z_l_zero):
            z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        for _ in range(self.trm_l_cycles):
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
            if bool(z_h_zero):
                z_h = torch.zeros_like(z_h)
            z_l = self._run_stage(
                self.think,
                z_l + z_h + encoded,
                causal_mask=causal_mask,
            )
            z_l = self.trm_l_post_norm(z_l)
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        z_h = self._run_stage(
            self.hrm_h_think,
            z_h + z_l,
            causal_mask=causal_mask,
        )
        z_h = self.trm_h_post_norm(z_h)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        return z_l, z_h

    def _run_trm_interactive_h_cycle(
        self,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        encoded: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(z_l_zero):
            z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        for _ in range(self.trm_l_cycles):
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
            if bool(z_h_zero):
                z_h = torch.zeros_like(z_h)
            z_l = self.trm_l_post_norm(
                self._run_stage(
                    self.think,
                    z_l + z_h + encoded,
                    causal_mask=causal_mask,
                )
            )
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        z_h = self.trm_h_post_norm(
            self._run_stage(
                self.think,
                z_h + z_l,
                causal_mask=causal_mask,
            )
        )
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        if self.think_structure == "trm_dual_z_interactive_prefix_scratch":
            z_h = self._apply_prefix_scratch(
                z_l,
                z_h,
                encoded,
                z_h_zero=bool(z_h_zero),
            )
        elif self.think_structure == "trm_dual_z_interactive_core_carrier":
            z_h = self._apply_core_carrier(
                z_l,
                z_h,
                encoded,
                z_h_zero=bool(z_h_zero),
            )
        return z_l, z_h

    def _run_trm_interactive_transition_gate_h_cycle(
        self,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        encoded: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(z_l_zero):
            z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        l_gate = torch.sigmoid(self.trm_l_transition_gate_logit).to(dtype=encoded.dtype)
        h_gate = torch.sigmoid(self.trm_h_transition_gate_logit).to(dtype=encoded.dtype)
        for _ in range(self.trm_l_cycles):
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
            if bool(z_h_zero):
                z_h = torch.zeros_like(z_h)
            candidate_l = self._run_stage(
                self.think,
                z_l + z_h + encoded,
                causal_mask=causal_mask,
            )
            z_l = self.trm_l_post_norm(z_l + l_gate * (candidate_l - z_l))
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        candidate_h = self._run_stage(
            self.think,
            z_h + z_l,
            causal_mask=causal_mask,
        )
        z_h = self.trm_h_post_norm(z_h + h_gate * (candidate_h - z_h))
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        return z_l, z_h

    def _run_trm_diffusive_h_cycle(
        self,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        encoded: torch.Tensor,
        *,
        h_step: int,
        total_steps: int,
        causal_mask: torch.Tensor,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(z_l_zero):
            z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        for l_step in range(self.trm_l_cycles):
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
            if bool(z_h_zero):
                z_h = torch.zeros_like(z_h)
            diffusion_context = self._trm_diffusion_context(
                encoded,
                h_step=int(h_step),
                total_steps=int(total_steps),
                l_step=int(l_step),
            )
            l_context = encoded + z_h + diffusion_context
            candidate_l = self._run_stage(
                self.think,
                self.trm_diffusion_input_norm(z_l + l_context),
                causal_mask=causal_mask,
            )
            z_l = self._gated_trm_update(
                z_l,
                candidate_l,
                l_context,
                gate=self.trm_diffusion_l_update_gate,
                norm=self.trm_l_post_norm,
            )
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        diffusion_context = self._trm_diffusion_context(
            encoded,
            h_step=int(h_step),
            total_steps=int(total_steps),
            l_step=self.trm_l_cycles,
        )
        h_context = encoded + z_l + diffusion_context
        candidate_h = self._run_stage(
            self.think,
            self.trm_diffusion_input_norm(z_h + h_context),
            causal_mask=causal_mask,
        )
        z_h = self._gated_trm_update(
            z_h,
            candidate_h,
            h_context,
            gate=self.trm_diffusion_h_update_gate,
            norm=self.trm_h_post_norm,
        )
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        return z_l, z_h

    def _run_trm_diffusive_reversed_hybrid_h_cycle(
        self,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        encoded: torch.Tensor,
        *,
        h_step: int,
        total_steps: int,
        causal_mask: torch.Tensor,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Diffusive dual-z update with reversed 3:1 mixers per TRM state.

        z_L carries the recurrent trajectory through Mamba3-heavy updates.
        z_H carries fast correction state through GatedDelta-heavy updates.
        Each block includes a final attention sync layer, keeping prompt
        grounding inside the canonical recurrent LM path.
        """
        if bool(z_l_zero):
            z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        for l_step in range(self.trm_l_cycles):
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
            if bool(z_h_zero):
                z_h = torch.zeros_like(z_h)
            diffusion_context = self._trm_diffusion_context(
                encoded,
                h_step=int(h_step),
                total_steps=int(total_steps),
                l_step=int(l_step),
            )
            l_context = encoded + z_h + diffusion_context
            candidate_l = self._run_stage(
                self.trm_l_mamba3_attention_hybrid,
                self.trm_diffusion_input_norm(z_l + l_context),
                causal_mask=causal_mask,
            )
            z_l = self._gated_trm_update(
                z_l,
                candidate_l,
                l_context,
                gate=self.trm_diffusion_l_update_gate,
                norm=self.trm_l_post_norm,
            )
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        diffusion_context = self._trm_diffusion_context(
            encoded,
            h_step=int(h_step),
            total_steps=int(total_steps),
            l_step=self.trm_l_cycles,
        )
        h_context = encoded + z_l + diffusion_context
        candidate_h = self._run_stage(
            self.trm_h_gated_delta_attention_hybrid,
            self.trm_diffusion_input_norm(z_h + h_context),
            causal_mask=causal_mask,
        )
        z_h = self._gated_trm_update(
            z_h,
            candidate_h,
            h_context,
            gate=self.trm_diffusion_h_update_gate,
            norm=self.trm_h_post_norm,
        )
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        return z_l, z_h

    def _run_trm_reversed_hybrid_order_router_h_cycle(
        self,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        encoded: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend normal L->H and reverse-primed H->L->H update orders."""

        def causal_recent_context(
            source: torch.Tensor,
            *,
            decay: float = 0.75,
        ) -> torch.Tensor:
            """Causal recency-biased context for suffix-heavy prompts."""
            alpha = float(decay)
            running = source[:, 0, :]
            rows = [running]
            for index in range(1, int(source.shape[1])):
                running = alpha * running + (1.0 - alpha) * source[:, index, :]
                rows.append(running)
            return torch.stack(rows, dim=1)

        def transition_state_active() -> bool:
            return (
                self.think_structure
                == "trm_dual_z_reversed_hybrid_3to1_transition_state_order_router"
            )

        def initial_transition_state(context: torch.Tensor) -> torch.Tensor | None:
            if not transition_state_active():
                return None
            seeded = torch.tanh(self.trm_order_transition_seed(context))
            return self.trm_order_transition_norm(seeded)

        def update_transition_state(
            cur_l: torch.Tensor,
            cur_h: torch.Tensor,
            context: torch.Tensor,
            transition_state: torch.Tensor | None,
        ) -> torch.Tensor | None:
            if transition_state is None:
                return None
            signal = torch.tanh(
                self.trm_order_transition_in(torch.cat([cur_l, cur_h, context], dim=-1))
            )
            next_state = self.trm_order_transition_cell(
                signal.reshape(-1, signal.shape[-1]),
                transition_state.reshape(-1, transition_state.shape[-1]),
            )
            return self.trm_order_transition_norm(next_state.reshape_as(transition_state))

        def transition_context(
            base_context: torch.Tensor,
            transition_state: torch.Tensor | None,
        ) -> torch.Tensor:
            if transition_state is None:
                return base_context
            gate = torch.sigmoid(self.trm_order_transition_gate_logit).to(
                dtype=base_context.dtype
            )
            return base_context + gate * transition_state.to(dtype=base_context.dtype)

        def l_then_h(
            start_l: torch.Tensor,
            start_h: torch.Tensor,
            *,
            context: torch.Tensor,
            transition_state: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
            cur_l = start_l
            cur_h = start_h
            for _ in range(self.trm_l_cycles):
                if bool(z_l_zero):
                    cur_l = torch.zeros_like(cur_l)
                if bool(z_h_zero):
                    cur_h = torch.zeros_like(cur_h)
                transition_state = update_transition_state(
                    cur_l,
                    cur_h,
                    context,
                    transition_state,
                )
                l_context = transition_context(context + cur_h, transition_state)
                candidate_l = self._run_stage(
                    self.trm_l_mamba3_attention_hybrid,
                    self.trm_reversed_hybrid_input_norm(cur_l + l_context),
                    causal_mask=causal_mask,
                )
                cur_l = self._gated_trm_update(
                    cur_l,
                    candidate_l,
                    l_context,
                    gate=self.trm_reversed_hybrid_l_update_gate,
                    norm=self.trm_l_post_norm,
                )
            if bool(z_l_zero):
                cur_l = torch.zeros_like(cur_l)
            if bool(z_h_zero):
                cur_h = torch.zeros_like(cur_h)
            transition_state = update_transition_state(
                cur_l,
                cur_h,
                context,
                transition_state,
            )
            h_context = transition_context(context + cur_l, transition_state)
            candidate_h = self._run_stage(
                self.trm_h_gated_delta_attention_hybrid,
                self.trm_reversed_hybrid_input_norm(cur_h + h_context),
                causal_mask=causal_mask,
            )
            cur_h = self._gated_trm_update(
                cur_h,
                candidate_h,
                h_context,
                gate=self.trm_reversed_hybrid_h_update_gate,
                norm=self.trm_h_post_norm,
            )
            return cur_l, cur_h, transition_state

        def h_then_l_then_h(
            start_l: torch.Tensor,
            start_h: torch.Tensor,
            *,
            context: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
            cur_l = start_l
            cur_h = start_h
            transition_state = initial_transition_state(context)
            if bool(z_l_zero):
                cur_l = torch.zeros_like(cur_l)
            if bool(z_h_zero):
                cur_h = torch.zeros_like(cur_h)
            transition_state = update_transition_state(
                cur_l,
                cur_h,
                context,
                transition_state,
            )
            h_context = transition_context(context + cur_l, transition_state)
            candidate_h = self._run_stage(
                self.trm_h_gated_delta_attention_hybrid,
                self.trm_reversed_hybrid_input_norm(cur_h + h_context),
                causal_mask=causal_mask,
            )
            cur_h = self._gated_trm_update(
                cur_h,
                candidate_h,
                h_context,
                gate=self.trm_reversed_hybrid_h_update_gate,
                norm=self.trm_h_post_norm,
            )
            return l_then_h(cur_l, cur_h, context=context, transition_state=transition_state)

        if bool(z_l_zero):
            z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        lh_l, lh_h, _ = l_then_h(z_l, z_h, context=encoded)
        if self.think_structure == "trm_dual_z_reversed_hybrid_3to1_recent_order_router":
            route1_context = causal_recent_context(encoded)
        elif self.think_structure == "trm_dual_z_reversed_hybrid_3to1_state_gru_order_router":
            order_state, _ = self.trm_order_state_gru(encoded)
            route1_context = self.trm_order_state_norm(encoded + order_state)
        else:
            route1_context = encoded
        hlh_l, hlh_h, _ = h_then_l_then_h(z_l, z_h, context=route1_context)
        force_route = getattr(self, "trm_order_router_force_route", None)
        if force_route is None:
            route = torch.softmax(self.trm_order_router(encoded), dim=-1)
        else:
            route_id = int(force_route)
            if route_id not in {0, 1}:
                raise ValueError("trm_order_router_force_route must be 0, 1, or None")
            route = encoded.new_zeros((*encoded.shape[:-1], 2))
            route[..., route_id] = 1.0
        l_weight = route[..., 0:1].to(dtype=lh_l.dtype)
        h_weight = route[..., 1:2].to(dtype=lh_h.dtype)
        out_l = self.trm_l_post_norm(l_weight * lh_l + h_weight * hlh_l)
        out_h = self.trm_h_post_norm(l_weight * lh_h + h_weight * hlh_h)
        if bool(z_l_zero):
            out_l = torch.zeros_like(out_l)
        if bool(z_h_zero):
            out_h = torch.zeros_like(out_h)
        return out_l, out_h

    def _run_trm_reversed_hybrid_h_cycle(
        self,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        encoded: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Non-diffusive L-then-H split-mixer dual-z update.

        Despite the historical `reversed_hybrid` names kept for compatibility,
        the default path here keeps the official TRM schedule: several z_L
        updates, then one z_H update. The split-mixer variants assign
        z_L=Mamba3+Attention and z_H=GatedDelta+Attention.
        """
        if self.think_structure in {
            "trm_dual_z_reversed_hybrid_3to1_order_router",
            "trm_dual_z_reversed_hybrid_3to1_recent_order_router",
            "trm_dual_z_reversed_hybrid_3to1_state_gru_order_router",
            "trm_dual_z_reversed_hybrid_3to1_transition_state_order_router",
        }:
            return self._run_trm_reversed_hybrid_order_router_h_cycle(
                z_l,
                z_h,
                encoded,
                causal_mask=causal_mask,
                z_l_zero=bool(z_l_zero),
                z_h_zero=bool(z_h_zero),
            )
        if bool(z_l_zero):
            z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        semantic_carry = self.think_structure == "trm_dual_z_reversed_hybrid_3to1_semantic_carry"
        if semantic_carry:
            anchor_l, anchor_h = self._initial_trm_state(encoded)
        for _ in range(self.trm_l_cycles):
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
            if bool(z_h_zero):
                z_h = torch.zeros_like(z_h)
            l_context = encoded + z_h
            prev_l = z_l
            candidate_l = self._run_stage(
                self.trm_l_mamba3_attention_hybrid,
                self.trm_reversed_hybrid_input_norm(z_l + l_context),
                causal_mask=causal_mask,
            )
            z_l = self._gated_trm_update(
                z_l,
                candidate_l,
                l_context,
                gate=self.trm_reversed_hybrid_l_update_gate,
                norm=self.trm_l_post_norm,
            )
            if self.think_structure in {
                "trm_dual_z_nested_reversed_hybrid_3to1",
                "trm_dual_z_nested_official_schedule_split_mixer_3to1",
            }:
                l_update_signal = torch.cat(
                    [
                        z_l,
                        prev_l,
                        candidate_l - prev_l,
                        l_context,
                    ],
                    dim=-1,
                )
                l_nested_delta = self.trm_nested_l_optimizer(
                    self.trm_nested_l_update_norm(l_update_signal)
                )
                l_nested_gate = torch.sigmoid(self.trm_nested_l_update_gate_logit).to(
                    dtype=z_l.dtype
                )
                z_l = self.trm_l_post_norm(z_l + l_nested_gate * l_nested_delta)
            if semantic_carry:
                z_l = self._apply_semantic_carry(
                    z_l,
                    anchor_l,
                    l_context,
                    gate=self.trm_semantic_l_carry_gate,
                    norm=self.trm_l_post_norm,
                )
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        h_context = encoded + z_l
        prev_h = z_h
        candidate_h = self._run_stage(
            self.trm_h_gated_delta_attention_hybrid,
            self.trm_reversed_hybrid_input_norm(z_h + h_context),
            causal_mask=causal_mask,
        )
        z_h = self._gated_trm_update(
            z_h,
            candidate_h,
            h_context,
            gate=self.trm_reversed_hybrid_h_update_gate,
            norm=self.trm_h_post_norm,
        )
        if self.think_structure in {
            "trm_dual_z_nested_reversed_hybrid_3to1",
            "trm_dual_z_nested_official_schedule_split_mixer_3to1",
        }:
            h_update_signal = torch.cat(
                [
                    z_h,
                    prev_h,
                    candidate_h - prev_h,
                    h_context,
                ],
                dim=-1,
            )
            h_nested_delta = self.trm_nested_h_optimizer(
                self.trm_nested_h_update_norm(h_update_signal)
            )
            h_nested_gate = torch.sigmoid(self.trm_nested_h_update_gate_logit).to(
                dtype=z_h.dtype
            )
            z_h = self.trm_h_post_norm(z_h + h_nested_gate * h_nested_delta)
        if semantic_carry:
            z_h = self._apply_semantic_carry(
                z_h,
                anchor_h,
                h_context,
                gate=self.trm_semantic_h_carry_gate,
                norm=self.trm_h_post_norm,
            )
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        return z_l, z_h

    def _run_trm_reversed_mha_h_cycle(
        self,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        encoded: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        h_step: int = 0,
        coupling_off: bool = False,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
        carrier_off: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Baseline-preserving dual reverse routing with the warm-startable MHA ETD block."""
        cross_exchange = (
            self.think_structure
            == "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange"
        )
        step_conditioned = (
            self.think_structure
            == "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned"
        )
        order_router = self.think_structure in {
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router",
        }
        sequence_order_router = (
            self.think_structure
            == "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router"
        )
        order_bound_router = (
            self.think_structure
            == "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router"
        )
        step_context = (
            self._nested_step_context(encoded, h_step=int(h_step))
            if step_conditioned
            else None
        )
        if bool(z_l_zero):
            z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        if order_router:

            def route1_order_context(
                cur_l: torch.Tensor,
                cur_h: torch.Tensor,
            ) -> torch.Tensor:
                if not bool(order_bound_router):
                    return torch.zeros_like(encoded)
                query = self.trm_nested_route1_order_query_norm(cur_h + cur_l)
                attended, _weights = self.trm_nested_route1_order_attn(
                    query,
                    encoded,
                    encoded,
                    attn_mask=causal_mask,
                    need_weights=False,
                )
                gate = torch.sigmoid(self.trm_nested_route1_order_gate_logit).to(
                    dtype=encoded.dtype
                )
                return gate * self.trm_nested_route1_order_norm(attended)

            def run_l_update(
                cur_l: torch.Tensor,
                cur_h: torch.Tensor,
                *,
                bind_order: bool = False,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if bool(z_l_zero):
                    cur_l = torch.zeros_like(cur_l)
                if bool(z_h_zero):
                    cur_h = torch.zeros_like(cur_h)
                bound_context = route1_order_context(cur_l, cur_h) if bind_order else 0.0
                l_context = encoded + cur_h
                if bind_order:
                    l_context = l_context + self.trm_nested_route1_order_to_l(
                        bound_context.to(dtype=cur_l.dtype)
                    )
                baseline_l = self._run_stage(
                    self.think,
                    cur_l,
                    causal_mask=causal_mask,
                )
                contextual_l = self._run_stage(
                    self.think,
                    self.trm_reversed_hybrid_input_norm(cur_l + l_context),
                    causal_mask=causal_mask,
                )
                l_gate = torch.sigmoid(
                    self.trm_reversed_hybrid_l_update_gate(
                        torch.cat([baseline_l, contextual_l, l_context], dim=-1)
                    )
                )
                next_l = baseline_l + l_gate * (contextual_l - baseline_l)
                l_update_signal = torch.cat(
                    [
                        next_l,
                        baseline_l,
                        contextual_l - baseline_l,
                        l_context,
                    ],
                    dim=-1,
                )
                l_nested_delta = self.trm_nested_l_optimizer(
                    self.trm_nested_l_update_norm(l_update_signal)
                )
                l_nested_gate = torch.sigmoid(self.trm_nested_l_update_gate_logit).to(
                    dtype=next_l.dtype
                )
                next_l = next_l + l_nested_gate * l_nested_delta
                if bool(z_l_zero):
                    next_l = torch.zeros_like(next_l)
                return next_l, cur_h

            def run_h_update(
                cur_l: torch.Tensor,
                cur_h: torch.Tensor,
                *,
                bind_order: bool = False,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if bool(z_l_zero):
                    cur_l = torch.zeros_like(cur_l)
                if bool(z_h_zero):
                    cur_h = torch.zeros_like(cur_h)
                bound_context = route1_order_context(cur_l, cur_h) if bind_order else 0.0
                h_context = encoded + cur_l
                if bind_order:
                    h_context = h_context + self.trm_nested_route1_order_to_h(
                        bound_context.to(dtype=cur_h.dtype)
                    )
                baseline_h = self._run_stage(
                    self.think,
                    cur_h,
                    causal_mask=causal_mask,
                )
                contextual_h = self._run_stage(
                    self.think,
                    self.trm_reversed_hybrid_input_norm(cur_h + h_context),
                    causal_mask=causal_mask,
                )
                h_gate = torch.sigmoid(
                    self.trm_reversed_hybrid_h_update_gate(
                        torch.cat([baseline_h, contextual_h, h_context], dim=-1)
                    )
                )
                next_h = baseline_h + h_gate * (contextual_h - baseline_h)
                h_update_signal = torch.cat(
                    [
                        next_h,
                        baseline_h,
                        contextual_h - baseline_h,
                        h_context,
                    ],
                    dim=-1,
                )
                h_nested_delta = self.trm_nested_h_optimizer(
                    self.trm_nested_h_update_norm(h_update_signal)
                )
                h_nested_gate = torch.sigmoid(self.trm_nested_h_update_gate_logit).to(
                    dtype=next_h.dtype
                )
                next_h = next_h + h_nested_gate * h_nested_delta
                if bool(z_h_zero):
                    next_h = torch.zeros_like(next_h)
                return cur_l, next_h

            def run_route(*, prime_h: bool) -> tuple[torch.Tensor, torch.Tensor]:
                route_l = z_l
                route_h = z_h
                bind_order = bool(prime_h and order_bound_router)
                if bool(prime_h):
                    route_l, route_h = run_h_update(
                        route_l,
                        route_h,
                        bind_order=bind_order,
                    )
                for _ in range(self.trm_l_cycles):
                    route_l, route_h = run_l_update(
                        route_l,
                        route_h,
                        bind_order=bind_order,
                    )
                route_l, route_h = run_h_update(
                    route_l,
                    route_h,
                    bind_order=bind_order,
                )
                route_l, route_h = self._apply_nested_core_carrier(
                    route_l,
                    route_h,
                    encoded,
                    z_l_zero=bool(z_l_zero),
                    z_h_zero=bool(z_h_zero),
                    carrier_off=bool(carrier_off),
                )
                return route_l, route_h

            route0_l, route0_h = run_route(prime_h=False)
            route1_l, route1_h = run_route(prime_h=True)
            force_route = getattr(self, "trm_nested_order_router_force_route", None)
            if bool(coupling_off):
                force_route = 0
            if force_route is None:
                route_logits = self.trm_nested_order_router(encoded)
                if bool(sequence_order_router):
                    route_logits = route_logits[:, -1:, :].expand_as(route_logits)
                route = torch.softmax(route_logits, dim=-1)
            else:
                route_id = int(force_route)
                if route_id not in {0, 1}:
                    raise ValueError(
                        "trm_nested_order_router_force_route must be 0, 1, or None"
                    )
                route = encoded.new_zeros((*encoded.shape[:-1], 2))
                route[..., route_id] = 1.0
            route0_weight = route[..., 0:1].to(dtype=route0_l.dtype)
            route1_weight = route[..., 1:2].to(dtype=route1_l.dtype)
            out_l = route0_weight * route0_l + route1_weight * route1_l
            out_h = route0_weight * route0_h + route1_weight * route1_h
            if bool(z_l_zero):
                out_l = torch.zeros_like(out_l)
            if bool(z_h_zero):
                out_h = torch.zeros_like(out_h)
            return out_l, out_h
        for _ in range(self.trm_l_cycles):
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
            if bool(z_h_zero):
                z_h = torch.zeros_like(z_h)
            previous_l = z_l
            l_context = encoded + z_h
            baseline_l = self._run_stage(
                self.think,
                z_l,
                causal_mask=causal_mask,
            )
            contextual_l = self._run_stage(
                self.think,
                self.trm_reversed_hybrid_input_norm(z_l + l_context),
                causal_mask=causal_mask,
            )
            l_gate = torch.sigmoid(
                self.trm_reversed_hybrid_l_update_gate(
                    torch.cat([baseline_l, contextual_l, l_context], dim=-1)
                )
            )
            z_l = baseline_l + l_gate * (contextual_l - baseline_l)
            if self.think_structure in {
                "trm_dual_z_nested_reversed_mha_etd",
                "trm_dual_z_nested_reversed_mha_etd_joint_readout",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
            }:
                l_update_signal = torch.cat(
                    [
                        z_l,
                        baseline_l,
                        contextual_l - baseline_l,
                        l_context,
                    ],
                    dim=-1,
                )
                l_nested_delta = self.trm_nested_l_optimizer(
                    self.trm_nested_l_update_norm(l_update_signal)
                )
                l_nested_gate = torch.sigmoid(self.trm_nested_l_update_gate_logit).to(
                    dtype=z_l.dtype
                )
                z_l = z_l + l_nested_gate * l_nested_delta
                if cross_exchange and not bool(coupling_off) and not bool(z_h_zero):
                    cross_gate = torch.sigmoid(
                        self.trm_nested_cross_exchange_gate_logit
                    ).to(dtype=z_h.dtype)
                    z_h = z_h + cross_gate * self.trm_nested_cross_to_h_norm(
                        l_nested_delta.to(dtype=z_h.dtype)
                    )
                if step_conditioned and not bool(coupling_off):
                    assert step_context is not None
                    step_l_signal = torch.cat(
                        [
                            z_l,
                            z_h,
                            encoded,
                            step_context,
                        ],
                        dim=-1,
                    )
                    step_l_delta = self.trm_nested_step_l_optimizer(
                        self.trm_nested_step_l_update_norm(step_l_signal)
                    )
                    step_gate = torch.sigmoid(
                        self.trm_nested_step_update_gate_logit
                    ).to(dtype=z_l.dtype)
                    z_l = z_l + step_gate * step_l_delta
            z_l = self._apply_trm_recurrent_layerscale(
                previous_l,
                z_l,
                state_name="l",
            )
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        previous_h = z_h
        h_context = encoded + z_l
        baseline_h = self._run_stage(
            self.think,
            z_h,
            causal_mask=causal_mask,
        )
        contextual_h = self._run_stage(
            self.think,
            self.trm_reversed_hybrid_input_norm(z_h + h_context),
            causal_mask=causal_mask,
        )
        h_gate = torch.sigmoid(
            self.trm_reversed_hybrid_h_update_gate(
                torch.cat([baseline_h, contextual_h, h_context], dim=-1)
            )
        )
        z_h = baseline_h + h_gate * (contextual_h - baseline_h)
        if self.think_structure in {
            "trm_dual_z_nested_reversed_mha_etd",
            "trm_dual_z_nested_reversed_mha_etd_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
        }:
            h_update_signal = torch.cat(
                [
                    z_h,
                    baseline_h,
                    contextual_h - baseline_h,
                    h_context,
                ],
                dim=-1,
            )
            h_nested_delta = self.trm_nested_h_optimizer(
                self.trm_nested_h_update_norm(h_update_signal)
            )
            h_nested_gate = torch.sigmoid(self.trm_nested_h_update_gate_logit).to(
                dtype=z_h.dtype
            )
            z_h = z_h + h_nested_gate * h_nested_delta
            if cross_exchange and not bool(coupling_off) and not bool(z_l_zero):
                cross_gate = torch.sigmoid(
                    self.trm_nested_cross_exchange_gate_logit
                ).to(dtype=z_l.dtype)
                z_l = z_l + cross_gate * self.trm_nested_cross_to_l_norm(
                    h_nested_delta.to(dtype=z_l.dtype)
                )
            if step_conditioned and not bool(coupling_off):
                assert step_context is not None
                step_h_signal = torch.cat(
                    [
                        z_h,
                        z_l,
                        encoded,
                        step_context,
                    ],
                    dim=-1,
                )
                step_h_delta = self.trm_nested_step_h_optimizer(
                    self.trm_nested_step_h_update_norm(step_h_signal)
                )
                step_gate = torch.sigmoid(
                    self.trm_nested_step_update_gate_logit
                ).to(dtype=z_h.dtype)
                z_h = z_h + step_gate * step_h_delta
        z_h = self._apply_trm_recurrent_layerscale(
            previous_h,
            z_h,
            state_name="h",
        )
        if (
            self.think_structure
            in {
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
            }
        ):
            z_l, z_h = self._apply_nested_core_carrier(
                z_l,
                z_h,
                encoded,
                z_l_zero=bool(z_l_zero),
                z_h_zero=bool(z_h_zero),
                carrier_off=bool(carrier_off),
            )
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        return z_l, z_h

    def _run_trm_gated_h_cycle(
        self,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        encoded: torch.Tensor,
        *,
        h_step: int,
        causal_mask: torch.Tensor,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(z_l_zero):
            z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        for l_step in range(self.trm_l_cycles):
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
            if bool(z_h_zero):
                z_h = torch.zeros_like(z_h)
            step_state = self._trm_step_state(encoded, h_step=int(h_step), l_step=int(l_step))
            l_context = encoded + z_h + step_state
            candidate_l = self._run_stage(
                self.think,
                self.trm_l_input_norm(z_l + l_context),
                causal_mask=causal_mask,
            )
            z_l = self._gated_trm_update(
                z_l,
                candidate_l,
                l_context,
                gate=self.trm_l_update_gate,
                norm=self.trm_l_post_norm,
            )
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        step_state = self._trm_step_state(
            encoded,
            h_step=int(h_step),
            l_step=self.trm_l_cycles,
        )
        h_context = encoded + z_l + step_state
        candidate_h = self._run_stage(
            self.think,
            self.trm_h_input_norm(z_h + h_context),
            causal_mask=causal_mask,
        )
        z_h = self._gated_trm_update(
            z_h,
            candidate_h,
            h_context,
            gate=self.trm_h_update_gate,
            norm=self.trm_h_post_norm,
        )
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        return z_l, z_h

    def _run_trm_coupled_residual_h_cycle(
        self,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        encoded: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        coupling_off: bool = False,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta_alpha = self.trm_delta_alpha.to(dtype=encoded.dtype)
        mamba_alpha = self.trm_mamba_alpha.to(dtype=encoded.dtype)
        coupling_alpha = (
            torch.zeros((), device=encoded.device, dtype=encoded.dtype)
            if bool(coupling_off)
            else self.trm_coupling_alpha.to(dtype=encoded.dtype)
        )
        if bool(z_l_zero):
            z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        for _ in range(self.trm_l_cycles):
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
            if bool(z_h_zero):
                z_h = torch.zeros_like(z_h)
            official_l = self._run_stage(
                self.think,
                z_l + z_h + encoded,
                causal_mask=causal_mask,
            )
            delta_input = z_l + encoded
            delta_proposal = self._run_stage(
                self.trm_l_delta_proposal,
                delta_input,
                causal_mask=causal_mask,
            ) - delta_input
            top_down = self.trm_h_to_l(z_h)
            z_l = self.trm_l_post_norm(
                official_l + delta_alpha * delta_proposal + coupling_alpha * top_down
            )
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        official_h = self._run_stage(
            self.think,
            z_h + z_l,
            causal_mask=causal_mask,
        )
        mamba_input = z_h + encoded
        mamba_proposal = self._run_stage(
            self.trm_h_mamba_proposal,
            mamba_input,
            causal_mask=causal_mask,
        ) - mamba_input
        bottom_up = self.trm_l_to_h(z_l)
        z_h = self.trm_h_post_norm(
            official_h + mamba_alpha * mamba_proposal + coupling_alpha * bottom_up
        )
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        return z_l, z_h

    def _run_trm_coupled_h_cycle(
        self,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        encoded: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        coupling_off: bool = False,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coupling_alpha = (
            torch.zeros((), device=encoded.device, dtype=encoded.dtype)
            if bool(coupling_off)
            else self.trm_coupling_alpha.to(dtype=encoded.dtype)
        )
        if bool(z_l_zero):
            z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        for _ in range(self.trm_l_cycles):
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
            if bool(z_h_zero):
                z_h = torch.zeros_like(z_h)
            official_l = self._run_stage(
                self.think,
                z_l + z_h + encoded,
                causal_mask=causal_mask,
            )
            local_proposal = torch.zeros_like(official_l)
            if hasattr(self, "trm_l_delta_proposal"):
                delta_input = z_l + encoded
                delta_scale = (
                    torch.sigmoid(self.trm_delta_gate_logit).to(dtype=encoded.dtype)
                    if hasattr(self, "trm_delta_gate_logit")
                    else self.trm_delta_alpha.to(dtype=encoded.dtype)
                )
                local_proposal = delta_scale * (
                    self._run_stage(
                        self.trm_l_delta_proposal,
                        delta_input,
                        causal_mask=causal_mask,
                    )
                    - delta_input
                )
            z_l = self.trm_l_post_norm(
                official_l + coupling_alpha * self.trm_h_to_l(z_h) + local_proposal
            )
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        official_h = self._run_stage(
            self.think,
            z_h + z_l,
            causal_mask=causal_mask,
        )
        slow_proposal = torch.zeros_like(official_h)
        if hasattr(self, "trm_h_mamba_proposal"):
            mamba_input = z_h + encoded
            mamba_scale = (
                torch.sigmoid(self.trm_mamba_gate_logit).to(dtype=encoded.dtype)
                if hasattr(self, "trm_mamba_gate_logit")
                else self.trm_mamba_alpha.to(dtype=encoded.dtype)
            )
            slow_proposal = mamba_scale * (
                self._run_stage(
                    self.trm_h_mamba_proposal,
                    mamba_input,
                    causal_mask=causal_mask,
                )
                - mamba_input
            )
        z_h = self.trm_h_post_norm(
            official_h + coupling_alpha * self.trm_l_to_h(z_l) + slow_proposal
        )
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        return z_l, z_h

    def _run_trm_coupled_hybrid_router_h_cycle(
        self,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        encoded: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        coupling_off: bool = False,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dual-z TRM with routed official attention, delta, and Mamba updates.

        The official TRM attention update remains the dominant initialized path.
        GatedDeltaNet is treated as the fast/local low-state proposal, Mamba3 as
        the slow high-state proposal, and L/H coupling is an ablatable recurrent
        communication path rather than a hidden answer channel.
        """
        coupling_scale = (
            torch.zeros((), device=encoded.device, dtype=encoded.dtype)
            if bool(coupling_off)
            else torch.ones((), device=encoded.device, dtype=encoded.dtype)
        )
        if bool(z_l_zero):
            z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        for _ in range(self.trm_l_cycles):
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
            if bool(z_h_zero):
                z_h = torch.zeros_like(z_h)
            official_l = self._run_stage(
                self.think,
                z_l + z_h + encoded,
                causal_mask=causal_mask,
            )
            delta_input = z_l + encoded
            delta_update = self._run_stage(
                self.trm_l_delta_proposal,
                delta_input,
                causal_mask=causal_mask,
            ) - delta_input
            top_down_update = coupling_scale * self.trm_h_to_l(z_h)
            l_weights = torch.softmax(
                self.trm_l_hybrid_router(torch.cat([z_l, z_h, encoded], dim=-1)),
                dim=-1,
            ).to(dtype=encoded.dtype)
            z_l = self.trm_l_post_norm(
                z_l
                + l_weights[..., 0:1] * (official_l - z_l)
                + l_weights[..., 1:2] * delta_update
                + l_weights[..., 2:3] * top_down_update
            )
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        official_h = self._run_stage(
            self.think,
            z_h + z_l,
            causal_mask=causal_mask,
        )
        mamba_input = z_h + z_l + encoded
        mamba_update = self._run_stage(
            self.trm_h_mamba_proposal,
            mamba_input,
            causal_mask=causal_mask,
        ) - mamba_input
        bottom_up_update = coupling_scale * self.trm_l_to_h(z_l)
        h_weights = torch.softmax(
            self.trm_h_hybrid_router(torch.cat([z_l, z_h, encoded], dim=-1)),
            dim=-1,
        ).to(dtype=encoded.dtype)
        z_h = self.trm_h_post_norm(
            z_h
            + h_weights[..., 0:1] * (official_h - z_h)
            + h_weights[..., 1:2] * mamba_update
            + h_weights[..., 2:3] * bottom_up_update
        )
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        return z_l, z_h

    def _cross_state_attention(
        self,
        query: torch.Tensor,
        source: torch.Tensor,
        *,
        attn: nn.MultiheadAttention,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        mixed, _ = attn(query, source, source, attn_mask=causal_mask, need_weights=False)
        return mixed

    def _run_trm_coupled_cross_attention_h_cycle(
        self,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        encoded: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        coupling_off: bool = False,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coupling_alpha = (
            torch.zeros((), device=encoded.device, dtype=encoded.dtype)
            if bool(coupling_off)
            else self.trm_coupling_alpha.to(dtype=encoded.dtype)
        )
        cross_alpha = (
            torch.zeros((), device=encoded.device, dtype=encoded.dtype)
            if bool(coupling_off)
            else self.trm_cross_alpha.to(dtype=encoded.dtype)
        )
        if bool(z_l_zero):
            z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        for _ in range(self.trm_l_cycles):
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
            if bool(z_h_zero):
                z_h = torch.zeros_like(z_h)
            official_l = self._run_stage(
                self.think,
                z_l + z_h + encoded,
                causal_mask=causal_mask,
            )
            cross_l = self._cross_state_attention(
                z_l + encoded,
                z_h + encoded,
                attn=self.trm_l_cross_attn,
                causal_mask=causal_mask,
            )
            z_l = self.trm_l_post_norm(
                official_l + coupling_alpha * self.trm_h_to_l(z_h) + cross_alpha * cross_l
            )
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        official_h = self._run_stage(
            self.think,
            z_h + z_l,
            causal_mask=causal_mask,
        )
        cross_h = self._cross_state_attention(
            z_h + encoded,
            z_l + encoded,
            attn=self.trm_h_cross_attn,
            causal_mask=causal_mask,
        )
        z_h = self.trm_h_post_norm(
            official_h + coupling_alpha * self.trm_l_to_h(z_l) + cross_alpha * cross_h
        )
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        return z_l, z_h

    def _run_trm_coupled_step_conditioned_h_cycle(
        self,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        encoded: torch.Tensor,
        *,
        h_step: int,
        causal_mask: torch.Tensor,
        coupling_off: bool = False,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coupling_alpha = (
            torch.zeros((), device=encoded.device, dtype=encoded.dtype)
            if bool(coupling_off)
            else self.trm_coupling_alpha.to(dtype=encoded.dtype)
        )
        if bool(z_l_zero):
            z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        for l_step in range(self.trm_l_cycles):
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
            if bool(z_h_zero):
                z_h = torch.zeros_like(z_h)
            step_state = self._trm_step_state(encoded, h_step=int(h_step), l_step=int(l_step))
            official_l = self._run_stage(
                self.think,
                z_l + z_h + encoded + step_state,
                causal_mask=causal_mask,
            )
            z_l = self.trm_l_post_norm(
                official_l + coupling_alpha * self.trm_h_to_l(z_h)
            )
            if bool(z_l_zero):
                z_l = torch.zeros_like(z_l)
        step_state = self._trm_step_state(
            encoded,
            h_step=int(h_step),
            l_step=self.trm_l_cycles,
        )
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        official_h = self._run_stage(
            self.think,
            z_h + z_l + step_state,
            causal_mask=causal_mask,
        )
        z_h = self.trm_h_post_norm(
            official_h + coupling_alpha * self.trm_l_to_h(z_l)
        )
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        return z_l, z_h

    def _run_thinking(
        self,
        encoded: torch.Tensor,
        *,
        think_steps: int,
        state_reset_each_step: bool,
        thinking_block_off: bool,
        causal_mask: torch.Tensor,
        coupling_off: bool = False,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
        carrier_off: bool = False,
        adaptive_halt: bool = False,
        halt_threshold: float = 0.5,
        halt_min_steps: int = 1,
        return_runtime: bool = False,
        return_state_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = int(encoded.shape[0])
        collect_halt = bool(adaptive_halt) or bool(return_runtime)
        collect_state_trace = bool(return_runtime) and bool(return_state_trace)
        halt_logits_steps: list[torch.Tensor] = []
        state_trace_l_steps: list[torch.Tensor] = []
        state_trace_h_steps: list[torch.Tensor] = []
        core_halted = torch.zeros(batch, dtype=torch.bool, device=encoded.device)
        halt_steps = torch.zeros(batch, dtype=torch.long, device=encoded.device)
        executed_steps = 0
        halt_state = (
            self._initial_halt_state(encoded)
            if self.halt_pooling == "dedicated"
            else None
        )

        def record_halt(state: torch.Tensor, step_index: int) -> bool:
            nonlocal core_halted, halt_steps, executed_steps, halt_state
            executed_steps = int(step_index) + 1
            if not collect_halt:
                return False
            if self.halt_pooling == "dedicated":
                assert halt_state is not None
                halt_state = self._update_halt_state(halt_state, state)
            halt_logits = self._core_halt_logits(state, halt_state=halt_state)
            halt_logits_steps.append(halt_logits)
            if executed_steps < max(1, int(halt_min_steps)):
                return False
            halt_now = torch.sigmoid(halt_logits) >= float(halt_threshold)
            newly_halted = halt_now & ~core_halted
            step_value = torch.full_like(halt_steps, executed_steps)
            halt_steps = torch.where(newly_halted, step_value, halt_steps)
            core_halted = core_halted | halt_now
            return bool(adaptive_halt and bool(core_halted.all().item()))

        def finish(state: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
            if not return_runtime:
                return state
            if halt_logits_steps:
                halt_logits = torch.stack(halt_logits_steps, dim=1)
            else:
                halt_logits = encoded.new_empty((batch, 0))
            fallback_steps = torch.full_like(halt_steps, executed_steps)
            runtime = {
                "core_q_halt_logits": halt_logits,
                "core_halted": core_halted,
                "halt_steps": torch.where(core_halted, halt_steps, fallback_steps),
                "executed_think_steps": torch.tensor(
                    executed_steps,
                    dtype=torch.long,
                    device=encoded.device,
                ),
            }
            if collect_state_trace:
                if state_trace_h_steps:
                    runtime["core_state_trace_h"] = torch.stack(state_trace_h_steps, dim=1)
                else:
                    runtime["core_state_trace_h"] = encoded.new_empty(
                        (batch, 0, *encoded.shape[1:])
                    )
                if state_trace_l_steps:
                    runtime["core_state_trace_l"] = torch.stack(state_trace_l_steps, dim=1)
            return state, runtime

        if bool(thinking_block_off) or int(think_steps) <= 0:
            return finish(encoded)
        if self.think_structure in {
            "single",
            "single_core_carrier",
            *SINGLE_ORDER_ROUTER_THINK_STRUCTURES,
        }:
            h = encoded
            for step_index in range(max(0, int(think_steps))):
                base = encoded if bool(state_reset_each_step) else h
                if self.think_structure in SINGLE_ORDER_ROUTER_THINK_STRUCTURES:
                    h = self._run_single_order_router_step(
                        base,
                        encoded,
                        causal_mask=causal_mask,
                        step_index=step_index,
                        total_steps=int(think_steps),
                    )
                else:
                    h = self._run_stage(self.think, base, causal_mask=causal_mask)
                if self.think_structure == "single_core_carrier":
                    h = self._apply_single_core_carrier(h, encoded)
                if collect_state_trace:
                    state_trace_h_steps.append(h)
                if record_halt(h, step_index):
                    break
            return finish(h)

        z_l, z_h = self._initial_trm_state(encoded)
        if bool(z_l_zero):
            z_l = torch.zeros_like(z_l)
        if bool(z_h_zero):
            z_h = torch.zeros_like(z_h)
        for h_step in range(max(0, int(think_steps))):
            if bool(state_reset_each_step):
                z_l, z_h = self._initial_trm_state(encoded)
                if bool(z_l_zero):
                    z_l = torch.zeros_like(z_l)
                if bool(z_h_zero):
                    z_h = torch.zeros_like(z_h)
            no_grad_cycle = (
                self.training
                and self.trm_no_grad_inner_cycles
                and h_step < (int(think_steps) - 1)
            )
            if no_grad_cycle:
                with torch.no_grad():
                    if self.think_structure == "trm_dual_z_gated":
                        z_l, z_h = self._run_trm_gated_h_cycle(
                            z_l,
                            z_h,
                            encoded,
                            h_step=h_step,
                            causal_mask=causal_mask,
                            z_l_zero=bool(z_l_zero),
                            z_h_zero=bool(z_h_zero),
                        )
                    elif self.think_structure in {
                        "trm_dual_z_interactive",
                        "trm_dual_z_interactive_residual_readout",
                        "trm_dual_z_interactive_prefix_scratch",
                        "trm_dual_z_interactive_core_carrier",
                    }:
                        z_l, z_h = self._run_trm_interactive_h_cycle(
                            z_l,
                            z_h,
                            encoded,
                            causal_mask=causal_mask,
                            z_l_zero=bool(z_l_zero),
                            z_h_zero=bool(z_h_zero),
                        )
                    elif self.think_structure == "trm_dual_z_interactive_transition_gate":
                        z_l, z_h = self._run_trm_interactive_transition_gate_h_cycle(
                            z_l,
                            z_h,
                            encoded,
                            causal_mask=causal_mask,
                            z_l_zero=bool(z_l_zero),
                            z_h_zero=bool(z_h_zero),
                        )
                    elif self.think_structure == "trm_dual_z_diffusive":
                        z_l, z_h = self._run_trm_diffusive_h_cycle(
                            z_l,
                            z_h,
                            encoded,
                            h_step=h_step,
                            total_steps=int(think_steps),
                            causal_mask=causal_mask,
                            z_l_zero=bool(z_l_zero),
                            z_h_zero=bool(z_h_zero),
                        )
                    elif self.think_structure == "trm_dual_z_diffusive_reversed_hybrid_3to1":
                        z_l, z_h = self._run_trm_diffusive_reversed_hybrid_h_cycle(
                            z_l,
                            z_h,
                            encoded,
                            h_step=h_step,
                            total_steps=int(think_steps),
                            causal_mask=causal_mask,
                            z_l_zero=bool(z_l_zero),
                            z_h_zero=bool(z_h_zero),
                        )
                    elif self.think_structure == "trm_dual_z_diffusive_reversed_hybrid_3to1_joint_readout":
                        z_l, z_h = self._run_trm_diffusive_reversed_hybrid_h_cycle(
                            z_l,
                            z_h,
                            encoded,
                            h_step=h_step,
                            total_steps=int(think_steps),
                            causal_mask=causal_mask,
                            z_l_zero=bool(z_l_zero),
                            z_h_zero=bool(z_h_zero),
                        )
                    elif self.think_structure == "trm_dual_z_hrm_separate":
                        z_l, z_h = self._run_trm_hrm_separate_h_cycle(
                            z_l,
                            z_h,
                            encoded,
                            causal_mask=causal_mask,
                            z_l_zero=bool(z_l_zero),
                            z_h_zero=bool(z_h_zero),
                        )
                    elif self.think_structure in {
                        "trm_dual_z_reversed_mha_etd",
                        "trm_dual_z_nested_reversed_mha_etd",
                        "trm_dual_z_nested_reversed_mha_etd_joint_readout",
                        "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
                        "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
                        "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
                        "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
                        "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router",
                        "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router",
                        "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router",
                    }:
                        z_l, z_h = self._run_trm_reversed_mha_h_cycle(
                            z_l,
                            z_h,
                            encoded,
                            causal_mask=causal_mask,
                            h_step=h_step,
                            coupling_off=bool(coupling_off),
                            z_l_zero=bool(z_l_zero),
                            z_h_zero=bool(z_h_zero),
                            carrier_off=bool(carrier_off),
                        )
                    elif self.think_structure in {
                        "trm_dual_z_reversed_hybrid_3to1",
                        "trm_dual_z_reversed_hybrid_3to1_prenorm",
                        "trm_dual_z_reversed_hybrid_3to1_joint_readout",
                        "trm_dual_z_reversed_hybrid_3to1_core_gated_readout",
                        "trm_dual_z_reversed_hybrid_3to1_semantic_carry",
                        "trm_dual_z_reversed_hybrid_3to1_order_router",
                        "trm_dual_z_reversed_hybrid_3to1_recent_order_router",
                        "trm_dual_z_reversed_hybrid_3to1_state_gru_order_router",
                        "trm_dual_z_reversed_hybrid_3to1_transition_state_order_router",
                        "trm_dual_z_official_schedule_split_mixer_3to1",
                        "trm_dual_z_nested_reversed_hybrid_3to1",
                        "trm_dual_z_nested_official_schedule_split_mixer_3to1",
                    }:
                        z_l, z_h = self._run_trm_reversed_hybrid_h_cycle(
                            z_l,
                            z_h,
                            encoded,
                            causal_mask=causal_mask,
                            z_l_zero=bool(z_l_zero),
                            z_h_zero=bool(z_h_zero),
                        )
                    elif self.think_structure == "trm_dual_z_coupled_residual":
                        z_l, z_h = self._run_trm_coupled_residual_h_cycle(
                            z_l,
                            z_h,
                            encoded,
                            causal_mask=causal_mask,
                            coupling_off=bool(coupling_off),
                            z_l_zero=bool(z_l_zero),
                            z_h_zero=bool(z_h_zero),
                        )
                    elif self.think_structure in {
                        "trm_dual_z_coupled",
                        "trm_dual_z_coupled_delta_l_only",
                        "trm_dual_z_coupled_mamba_h_only",
                        "trm_dual_z_coupled_gated_proposal",
                    }:
                        z_l, z_h = self._run_trm_coupled_h_cycle(
                            z_l,
                            z_h,
                            encoded,
                            causal_mask=causal_mask,
                            coupling_off=bool(coupling_off),
                            z_l_zero=bool(z_l_zero),
                            z_h_zero=bool(z_h_zero),
                        )
                    elif self.think_structure == "trm_dual_z_coupled_hybrid_router":
                        z_l, z_h = self._run_trm_coupled_hybrid_router_h_cycle(
                            z_l,
                            z_h,
                            encoded,
                            causal_mask=causal_mask,
                            coupling_off=bool(coupling_off),
                            z_l_zero=bool(z_l_zero),
                            z_h_zero=bool(z_h_zero),
                        )
                    elif self.think_structure == "trm_dual_z_coupled_cross_attention":
                        z_l, z_h = self._run_trm_coupled_cross_attention_h_cycle(
                            z_l,
                            z_h,
                            encoded,
                            causal_mask=causal_mask,
                            coupling_off=bool(coupling_off),
                            z_l_zero=bool(z_l_zero),
                            z_h_zero=bool(z_h_zero),
                        )
                    elif self.think_structure == "trm_dual_z_coupled_step_conditioned_attention":
                        z_l, z_h = self._run_trm_coupled_step_conditioned_h_cycle(
                            z_l,
                            z_h,
                            encoded,
                            h_step=h_step,
                            causal_mask=causal_mask,
                            coupling_off=bool(coupling_off),
                            z_l_zero=bool(z_l_zero),
                            z_h_zero=bool(z_h_zero),
                        )
                    else:
                        z_l, z_h = self._run_trm_h_cycle(
                            z_l,
                            z_h,
                            encoded,
                            causal_mask=causal_mask,
                            z_l_zero=bool(z_l_zero),
                            z_h_zero=bool(z_h_zero),
                        )
            else:
                if self.think_structure == "trm_dual_z_gated":
                    z_l, z_h = self._run_trm_gated_h_cycle(
                        z_l,
                        z_h,
                        encoded,
                        h_step=h_step,
                        causal_mask=causal_mask,
                        z_l_zero=bool(z_l_zero),
                        z_h_zero=bool(z_h_zero),
                    )
                elif self.think_structure in {
                    "trm_dual_z_interactive",
                    "trm_dual_z_interactive_residual_readout",
                    "trm_dual_z_interactive_prefix_scratch",
                    "trm_dual_z_interactive_core_carrier",
                }:
                    z_l, z_h = self._run_trm_interactive_h_cycle(
                        z_l,
                        z_h,
                        encoded,
                        causal_mask=causal_mask,
                        z_l_zero=bool(z_l_zero),
                        z_h_zero=bool(z_h_zero),
                    )
                elif self.think_structure == "trm_dual_z_interactive_transition_gate":
                    z_l, z_h = self._run_trm_interactive_transition_gate_h_cycle(
                        z_l,
                        z_h,
                        encoded,
                        causal_mask=causal_mask,
                        z_l_zero=bool(z_l_zero),
                        z_h_zero=bool(z_h_zero),
                    )
                elif self.think_structure == "trm_dual_z_diffusive":
                    z_l, z_h = self._run_trm_diffusive_h_cycle(
                        z_l,
                        z_h,
                        encoded,
                        h_step=h_step,
                        total_steps=int(think_steps),
                        causal_mask=causal_mask,
                        z_l_zero=bool(z_l_zero),
                        z_h_zero=bool(z_h_zero),
                    )
                elif self.think_structure == "trm_dual_z_diffusive_reversed_hybrid_3to1":
                    z_l, z_h = self._run_trm_diffusive_reversed_hybrid_h_cycle(
                        z_l,
                        z_h,
                        encoded,
                        h_step=h_step,
                        total_steps=int(think_steps),
                        causal_mask=causal_mask,
                        z_l_zero=bool(z_l_zero),
                        z_h_zero=bool(z_h_zero),
                    )
                elif self.think_structure == "trm_dual_z_diffusive_reversed_hybrid_3to1_joint_readout":
                    z_l, z_h = self._run_trm_diffusive_reversed_hybrid_h_cycle(
                        z_l,
                        z_h,
                        encoded,
                        h_step=h_step,
                        total_steps=int(think_steps),
                        causal_mask=causal_mask,
                        z_l_zero=bool(z_l_zero),
                        z_h_zero=bool(z_h_zero),
                    )
                elif self.think_structure == "trm_dual_z_hrm_separate":
                    z_l, z_h = self._run_trm_hrm_separate_h_cycle(
                        z_l,
                        z_h,
                        encoded,
                        causal_mask=causal_mask,
                        z_l_zero=bool(z_l_zero),
                        z_h_zero=bool(z_h_zero),
                    )
                elif self.think_structure in {
                    "trm_dual_z_reversed_mha_etd",
                    "trm_dual_z_nested_reversed_mha_etd",
                    "trm_dual_z_nested_reversed_mha_etd_joint_readout",
                    "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
                    "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
                    "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
                    "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
                    "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router",
                    "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router",
                    "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router",
                }:
                    z_l, z_h = self._run_trm_reversed_mha_h_cycle(
                        z_l,
                        z_h,
                        encoded,
                        causal_mask=causal_mask,
                        h_step=h_step,
                        coupling_off=bool(coupling_off),
                        z_l_zero=bool(z_l_zero),
                        z_h_zero=bool(z_h_zero),
                        carrier_off=bool(carrier_off),
                    )
                elif self.think_structure in {
                    "trm_dual_z_reversed_hybrid_3to1",
                    "trm_dual_z_reversed_hybrid_3to1_prenorm",
                    "trm_dual_z_reversed_hybrid_3to1_joint_readout",
                    "trm_dual_z_reversed_hybrid_3to1_core_gated_readout",
                    "trm_dual_z_reversed_hybrid_3to1_semantic_carry",
                    "trm_dual_z_reversed_hybrid_3to1_order_router",
                    "trm_dual_z_reversed_hybrid_3to1_recent_order_router",
                    "trm_dual_z_reversed_hybrid_3to1_state_gru_order_router",
                    "trm_dual_z_reversed_hybrid_3to1_transition_state_order_router",
                    "trm_dual_z_official_schedule_split_mixer_3to1",
                    "trm_dual_z_nested_reversed_hybrid_3to1",
                    "trm_dual_z_nested_official_schedule_split_mixer_3to1",
                }:
                    z_l, z_h = self._run_trm_reversed_hybrid_h_cycle(
                        z_l,
                        z_h,
                        encoded,
                        causal_mask=causal_mask,
                        z_l_zero=bool(z_l_zero),
                        z_h_zero=bool(z_h_zero),
                    )
                elif self.think_structure == "trm_dual_z_coupled_residual":
                    z_l, z_h = self._run_trm_coupled_residual_h_cycle(
                        z_l,
                        z_h,
                        encoded,
                        causal_mask=causal_mask,
                        coupling_off=bool(coupling_off),
                        z_l_zero=bool(z_l_zero),
                        z_h_zero=bool(z_h_zero),
                    )
                elif self.think_structure in {
                    "trm_dual_z_coupled",
                    "trm_dual_z_coupled_delta_l_only",
                    "trm_dual_z_coupled_mamba_h_only",
                    "trm_dual_z_coupled_gated_proposal",
                }:
                    z_l, z_h = self._run_trm_coupled_h_cycle(
                        z_l,
                        z_h,
                        encoded,
                        causal_mask=causal_mask,
                        coupling_off=bool(coupling_off),
                        z_l_zero=bool(z_l_zero),
                        z_h_zero=bool(z_h_zero),
                    )
                elif self.think_structure == "trm_dual_z_coupled_hybrid_router":
                    z_l, z_h = self._run_trm_coupled_hybrid_router_h_cycle(
                        z_l,
                        z_h,
                        encoded,
                        causal_mask=causal_mask,
                        coupling_off=bool(coupling_off),
                        z_l_zero=bool(z_l_zero),
                        z_h_zero=bool(z_h_zero),
                    )
                elif self.think_structure == "trm_dual_z_coupled_cross_attention":
                    z_l, z_h = self._run_trm_coupled_cross_attention_h_cycle(
                        z_l,
                        z_h,
                        encoded,
                        causal_mask=causal_mask,
                        coupling_off=bool(coupling_off),
                        z_l_zero=bool(z_l_zero),
                        z_h_zero=bool(z_h_zero),
                    )
                elif self.think_structure == "trm_dual_z_coupled_step_conditioned_attention":
                    z_l, z_h = self._run_trm_coupled_step_conditioned_h_cycle(
                        z_l,
                        z_h,
                        encoded,
                        h_step=h_step,
                        causal_mask=causal_mask,
                        coupling_off=bool(coupling_off),
                        z_l_zero=bool(z_l_zero),
                        z_h_zero=bool(z_h_zero),
                    )
                else:
                    z_l, z_h = self._run_trm_h_cycle(
                        z_l,
                        z_h,
                        encoded,
                        causal_mask=causal_mask,
                        z_l_zero=bool(z_l_zero),
                        z_h_zero=bool(z_h_zero),
                    )
            if collect_state_trace:
                state_trace_l_steps.append(z_l)
                state_trace_h_steps.append(z_h)
            if record_halt(z_h, h_step):
                break
        if self.think_structure in {
            "trm_dual_z_gated",
            "trm_dual_z_residual",
            "trm_dual_z_interactive_residual_readout",
            "trm_dual_z_coupled_residual",
        }:
            return finish(self.trm_readout_norm(encoded + self.trm_readout_alpha * (z_l + z_h)))
        if self.think_structure in {
            "trm_dual_z_diffusive_reversed_hybrid_3to1_joint_readout",
            "trm_dual_z_reversed_hybrid_3to1_joint_readout",
        }:
            joint = torch.cat([encoded, z_l, z_h], dim=-1)
            return finish(self.trm_joint_readout_norm(z_h + self.trm_joint_readout_proj(joint)))
        if self.think_structure == "trm_dual_z_reversed_hybrid_3to1_core_gated_readout":
            joint = torch.cat([encoded, z_l, z_h], dim=-1)
            bridge = self.trm_joint_readout_proj(joint)
            core_gate = torch.tanh(z_h)
            alpha = self.trm_core_readout_alpha.to(dtype=z_h.dtype)
            return finish(self.trm_joint_readout_norm(z_h + alpha * bridge * core_gate))
        if self.think_structure == "trm_dual_z_nested_reversed_mha_etd_joint_readout":
            joint = torch.cat([encoded, z_l, z_h], dim=-1)
            return finish(self.trm_joint_readout_norm(z_h + self.trm_joint_readout_proj(joint)))
        if self.think_structure in {
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router",
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router",
        }:
            alpha = torch.tanh(self.trm_reversed_mha_readout_alpha).to(dtype=z_h.dtype)
            base = z_h + alpha * (z_l - z_h)
            joint = torch.cat([encoded, z_l, z_h], dim=-1)
            joint_residual = self.trm_joint_readout_norm(self.trm_joint_readout_proj(joint))
            residual_alpha = torch.tanh(self.trm_nested_mha_joint_readout_alpha).to(
                dtype=z_h.dtype
            )
            return finish(base + residual_alpha * joint_residual)
        if self.think_structure in {
            "trm_dual_z_reversed_mha_etd",
            "trm_dual_z_nested_reversed_mha_etd",
        }:
            alpha = torch.tanh(self.trm_reversed_mha_readout_alpha).to(dtype=z_h.dtype)
            return finish(z_h + alpha * (z_l - z_h))
        return finish(z_h)

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        *,
        think_steps: int,
        state_reset_each_step: bool = False,
        thinking_block_off: bool = False,
        coupling_off: bool = False,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
        carrier_off: bool = False,
        adaptive_halt: bool = False,
        halt_threshold: float = 0.5,
        halt_min_steps: int = 1,
        op_order_off: bool = False,
        return_runtime: bool = False,
        return_state_trace: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        seq_len = int(input_ids.shape[1])
        if seq_len > self.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")
        x = self._token_embeddings(input_ids, op_order_off=bool(op_order_off))
        if self.position_embedding_mode in {"learned", "randomized"}:
            x = x + self.pos_embed(self._position_ids(seq_len, input_ids.device))
        return self._forward_embedded_impl(
            x,
            think_steps=int(think_steps),
            state_reset_each_step=bool(state_reset_each_step),
            thinking_block_off=bool(thinking_block_off),
            coupling_off=bool(coupling_off),
            z_l_zero=bool(z_l_zero),
            z_h_zero=bool(z_h_zero),
            carrier_off=bool(carrier_off),
            adaptive_halt=bool(adaptive_halt),
            halt_threshold=float(halt_threshold),
            halt_min_steps=int(halt_min_steps),
            return_runtime=bool(return_runtime),
            return_state_trace=bool(return_state_trace),
        )

    def _forward_embedded_impl(
        self,
        x: torch.Tensor,
        *,
        think_steps: int,
        state_reset_each_step: bool = False,
        thinking_block_off: bool = False,
        coupling_off: bool = False,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
        carrier_off: bool = False,
        adaptive_halt: bool = False,
        halt_threshold: float = 0.5,
        halt_min_steps: int = 1,
        return_runtime: bool = False,
        return_state_trace: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("embedded inputs must have shape [batch, seq, dim]")
        seq_len = int(x.shape[1])
        if seq_len > self.max_seq_len:
            raise ValueError("embedded input sequence exceeds max_seq_len")
        mask = self._causal_mask(seq_len, x.device)
        encoded = self._run_stage(self.encode, x, causal_mask=mask)
        h = self._run_thinking(
            encoded,
            think_steps=int(think_steps),
            state_reset_each_step=bool(state_reset_each_step),
            thinking_block_off=bool(thinking_block_off),
            causal_mask=mask,
            coupling_off=bool(coupling_off),
            z_l_zero=bool(z_l_zero),
            z_h_zero=bool(z_h_zero),
            carrier_off=bool(carrier_off),
            adaptive_halt=bool(adaptive_halt),
            halt_threshold=float(halt_threshold),
            halt_min_steps=int(halt_min_steps),
            return_runtime=bool(return_runtime),
            return_state_trace=bool(return_state_trace),
        )
        runtime: dict[str, torch.Tensor] = {}
        if bool(return_runtime):
            h, runtime = h
        h = self._run_stage(self.decode, h, causal_mask=mask)
        logits = self._lm_logits(self.norm(h))
        if bool(return_runtime):
            return {"logits": logits, **runtime}
        return logits

    def forward_embeddings(
        self,
        embeddings: torch.Tensor,
        *,
        think_steps: int,
        state_reset_each_step: bool = False,
        thinking_block_off: bool = False,
        coupling_off: bool = False,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
        carrier_off: bool = False,
        adaptive_halt: bool = False,
        halt_threshold: float = 0.5,
        halt_min_steps: int = 1,
        op_order_off: bool = False,
    ) -> torch.Tensor:
        """Run the native LM path from precomputed token/bag embeddings."""
        x = embeddings
        if x.ndim != 3:
            raise ValueError("embeddings must have shape [batch, seq, dim]")
        if self.position_embedding_mode in {"learned", "randomized"}:
            seq_len = int(x.shape[1])
            if seq_len > self.max_seq_len:
                raise ValueError("embedded input sequence exceeds max_seq_len")
            x = x + self.pos_embed(self._position_ids(seq_len, x.device))
        return self._forward_embedded_impl(
            x,
            think_steps=int(think_steps),
            state_reset_each_step=bool(state_reset_each_step),
            thinking_block_off=bool(thinking_block_off),
            coupling_off=bool(coupling_off),
            z_l_zero=bool(z_l_zero),
            z_h_zero=bool(z_h_zero),
            carrier_off=bool(carrier_off),
            adaptive_halt=bool(adaptive_halt),
            halt_threshold=float(halt_threshold),
            halt_min_steps=int(halt_min_steps),
            return_runtime=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        think_steps: int,
        state_reset_each_step: bool = False,
        thinking_block_off: bool = False,
        coupling_off: bool = False,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
        carrier_off: bool = False,
        adaptive_halt: bool = False,
        halt_threshold: float = 0.5,
        halt_min_steps: int = 1,
        op_order_off: bool = False,
    ) -> torch.Tensor:
        return self._forward_impl(
            input_ids,
            think_steps=int(think_steps),
            state_reset_each_step=bool(state_reset_each_step),
            thinking_block_off=bool(thinking_block_off),
            coupling_off=bool(coupling_off),
            z_l_zero=bool(z_l_zero),
            z_h_zero=bool(z_h_zero),
            carrier_off=bool(carrier_off),
            adaptive_halt=bool(adaptive_halt),
            halt_threshold=float(halt_threshold),
            halt_min_steps=int(halt_min_steps),
            op_order_off=bool(op_order_off),
            return_runtime=False,
        )

    def forward_with_runtime(
        self,
        input_ids: torch.Tensor,
        *,
        think_steps: int,
        state_reset_each_step: bool = False,
        thinking_block_off: bool = False,
        coupling_off: bool = False,
        z_l_zero: bool = False,
        z_h_zero: bool = False,
        carrier_off: bool = False,
        adaptive_halt: bool = False,
        halt_threshold: float = 0.5,
        halt_min_steps: int = 1,
        op_order_off: bool = False,
        return_state_trace: bool = False,
    ) -> dict[str, torch.Tensor]:
        out = self._forward_impl(
            input_ids,
            think_steps=int(think_steps),
            state_reset_each_step=bool(state_reset_each_step),
            thinking_block_off=bool(thinking_block_off),
            coupling_off=bool(coupling_off),
            z_l_zero=bool(z_l_zero),
            z_h_zero=bool(z_h_zero),
            carrier_off=bool(carrier_off),
            adaptive_halt=bool(adaptive_halt),
            halt_threshold=float(halt_threshold),
            halt_min_steps=int(halt_min_steps),
            op_order_off=bool(op_order_off),
            return_runtime=True,
            return_state_trace=bool(return_state_trace),
        )
        return out


def answer_loss(
    logits: torch.Tensor,
    full_tokens: torch.Tensor,
    *,
    prompt_len: int,
    answer_loss_weight: float = 1.0,
    eos_loss_weight: float = 1.0,
    answer_eos_margin_weight: float = 0.0,
    answer_eos_margin: float = 0.0,
) -> torch.Tensor:
    # Logit at prompt_len - 1 predicts answer; prompt_len predicts EOS.
    answer_pos = int(prompt_len) - 1
    eos_pos = int(prompt_len)
    answer_targets = full_tokens[:, answer_pos + 1]
    eos_targets = full_tokens[:, eos_pos + 1]
    answer_logits = logits[:, answer_pos, :]
    eos_logits = logits[:, eos_pos, :]
    answer_ce = F.cross_entropy(answer_logits, answer_targets)
    eos_ce = F.cross_entropy(eos_logits, eos_targets)
    answer_weight = max(0.0, float(answer_loss_weight))
    eos_weight = max(0.0, float(eos_loss_weight))
    denom = max(answer_weight + eos_weight, 1e-8)
    loss = (answer_weight * answer_ce + eos_weight * eos_ce) / denom
    margin_weight = max(0.0, float(answer_eos_margin_weight))
    if margin_weight > 0.0:
        answer_target_logits = answer_logits.gather(
            1,
            answer_targets.view(-1, 1),
        ).squeeze(1)
        eos_as_first_logits = answer_logits[:, EOS]
        margin_loss = torch.relu(
            float(answer_eos_margin)
            - (answer_target_logits - eos_as_first_logits)
        ).mean()
        loss = loss + margin_weight * margin_loss
    return loss


def intermediate_depth_loss(
    model: NativeQTRMETDLM,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    prompt_len: int,
    max_depth: int,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    answer_pos = int(prompt_len) - 1
    for depth in range(1, max(1, int(max_depth)) + 1):
        depth_index = min(depth - 1, int(targets.shape[1]) - 1)
        logits = model(input_ids, think_steps=depth)
        losses.append(
            F.cross_entropy(
                logits[:, answer_pos, :],
                targets[:, depth_index],
            )
        )
    return torch.stack(losses).mean()


@torch.no_grad()
def generate_one(
    model: NativeQTRMETDLM,
    prompt: torch.Tensor,
    *,
    think_steps: int,
    max_new_tokens: int,
    state_reset_each_step: bool = False,
    thinking_block_off: bool = False,
    coupling_off: bool = False,
    z_l_zero: bool = False,
    z_h_zero: bool = False,
) -> list[int]:
    out = prompt.clone()
    for _ in range(int(max_new_tokens)):
        logits = model(
            out,
            think_steps=think_steps,
            state_reset_each_step=state_reset_each_step,
            thinking_block_off=thinking_block_off,
            coupling_off=coupling_off,
            z_l_zero=z_l_zero,
            z_h_zero=z_h_zero,
        )
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_id], dim=1)
        if int(next_id.item()) == EOS:
            break
    return out[0, int(prompt.shape[1]) :].detach().cpu().tolist()


def decode_answer(token_ids: Iterable[int]) -> str:
    pieces: list[str] = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id == EOS:
            break
        value = token_value(token_id)
        if value is None:
            pieces.append(f"<{token_id}>")
        else:
            pieces.append(str(value))
    return " ".join(pieces)


@torch.no_grad()
def evaluate(
    model: NativeQTRMETDLM,
    cases: list[NativeCase],
    args: argparse.Namespace,
    *,
    think_steps: int,
    ablation: str = "none",
) -> dict[str, object]:
    model.eval()
    device = torch.device(args.device)
    prompt_tokens, full_tokens = cases_to_batch(
        cases,
        device=device,
        ablation="op_zero" if ablation == "op_zero" else "op_shuffle" if ablation == "op_shuffle" else "none",
    )
    state_reset = ablation == "state_reset"
    thinking_off = ablation == "thinking_block_off"
    coupling_off = ablation == "coupling_off"
    z_l_zero = ablation == "z_l_zero"
    z_h_zero = ablation == "z_h_zero"
    generated: list[dict[str, object]] = []
    correct = 0
    answer_correct = 0
    eos_correct = 0
    first_token_eos = 0
    for row_idx, case in enumerate(cases):
        pred_tokens = generate_one(
            model,
            prompt_tokens[row_idx : row_idx + 1],
            think_steps=int(think_steps),
            max_new_tokens=2,
            state_reset_each_step=state_reset,
            thinking_block_off=thinking_off,
            coupling_off=coupling_off,
            z_l_zero=z_l_zero,
            z_h_zero=z_h_zero,
        )
        pred_answer = token_value(pred_tokens[0]) if pred_tokens else None
        is_correct = pred_answer == int(case.answer)
        has_eos = len(pred_tokens) > 1 and int(pred_tokens[1]) == EOS
        starts_with_eos = bool(pred_tokens) and int(pred_tokens[0]) == EOS
        correct += int(is_correct and has_eos)
        answer_correct += int(is_correct)
        eos_correct += int(has_eos)
        first_token_eos += int(starts_with_eos)
        if len(generated) < int(args.max_examples):
            generated.append(
                {
                    "case_id": case.case_id,
                    "start": case.start,
                    "op_ids": list(case.op_ids),
                    "gold_answer": case.answer,
                    "pred_tokens": pred_tokens,
                    "prediction": decode_answer(pred_tokens),
                    "exact": bool(is_correct and has_eos),
                }
            )
    logits = model(
        full_tokens[:, :-1],
        think_steps=int(think_steps),
        state_reset_each_step=state_reset,
        thinking_block_off=thinking_off,
        coupling_off=coupling_off,
        z_l_zero=z_l_zero,
        z_h_zero=z_h_zero,
    )
    prompt_len = len(case_prompt_tokens(cases[0]))
    loss = answer_loss(
        logits,
        full_tokens,
        prompt_len=prompt_len,
        answer_loss_weight=float(args.answer_loss_weight),
        eos_loss_weight=float(args.eos_loss_weight),
        answer_eos_margin_weight=float(args.answer_eos_margin_weight),
        answer_eos_margin=float(args.answer_eos_margin),
    )
    return {
        "ablation": ablation,
        "think_steps": int(think_steps),
        "cases": len(cases),
        "generation_exact": float(correct / max(1, len(cases))),
        "answer_token_accuracy": float(answer_correct / max(1, len(cases))),
        "eos_accuracy": float(eos_correct / max(1, len(cases))),
        "first_token_eos_rate": float(first_token_eos / max(1, len(cases))),
        "teacher_forced_answer_loss": float(loss.detach().cpu()),
        "examples": generated,
    }


def make_decision(eval_metrics: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    full = eval_metrics[f"think{args.eval_think_steps}"]
    think0 = eval_metrics["think0"]
    ablation_names = applicable_ablation_names(str(args.think_structure))
    ablation_exact = {
        name: float(eval_metrics[name]["generation_exact"])
        for name in ablation_names
        if name in eval_metrics
    }
    full_exact = float(full["generation_exact"])
    think0_exact = float(think0["generation_exact"])
    worst_ablation = max(ablation_exact.values()) if ablation_exact else float("-inf")
    reject_reasons: list[str] = []
    if full_exact < float(args.accept_min_exact):
        reject_reasons.append("full_exact_below_threshold")
    if (full_exact - think0_exact) < float(args.accept_min_depth_gain):
        reject_reasons.append("depth_gain_below_threshold")
    if (full_exact - worst_ablation) < float(args.accept_min_ablation_drop):
        reject_reasons.append("ablation_drop_below_threshold")
    decisive_metrics = {
        "full_generation_exact": full_exact,
        "think0_generation_exact": think0_exact,
        "full_minus_think0": full_exact - think0_exact,
        "full_minus_worst_ablation": full_exact - worst_ablation,
    }
    decisive_metrics.update(
        {f"{name}_generation_exact": value for name, value in ablation_exact.items()}
    )
    return {
        "accepted": not reject_reasons,
        "decision": str(args.accepted_decision) if not reject_reasons else "rejected",
        "reject_reasons": reject_reasons,
        "decisive_metrics": decisive_metrics,
        "thresholds": {
            "accept_min_exact": float(args.accept_min_exact),
            "accept_min_depth_gain": float(args.accept_min_depth_gain),
            "accept_min_ablation_drop": float(args.accept_min_ablation_drop),
        },
    }


def train_probe(args: argparse.Namespace) -> dict[str, object]:
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    train_cases = build_cases(
        count=int(args.train_cases),
        seed=int(args.seed),
        program_len=int(args.program_len),
        modulus=int(args.modulus),
    )
    eval_cases = build_cases(
        count=int(args.eval_cases),
        seed=int(args.eval_seed),
        program_len=int(args.program_len),
        modulus=int(args.modulus),
    )
    max_seq_len = int(args.program_len) + 7
    model = NativeQTRMETDLM(
        vocab=vocab_size(int(args.modulus)),
        max_seq_len=max_seq_len,
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        d_ff=int(args.d_ff),
        dropout=float(args.dropout),
        backbone=str(args.backbone),
        encode_backbone=str(args.encode_backbone or args.backbone),
        think_backbone=str(args.think_backbone or args.backbone),
        decode_backbone=str(args.decode_backbone or args.backbone),
        think_structure=str(args.think_structure),
        trm_l_cycles=int(args.trm_l_cycles),
        trm_no_grad_inner_cycles=not bool(args.trm_full_grad_cycles),
        n_kv_heads=int(args.n_kv_heads),
        hybrid_layers=int(args.hybrid_layers),
        attn_every=int(args.attn_every),
        delta_backend=str(args.delta_backend),
        delta_head_dim=int(args.delta_head_dim) if int(args.delta_head_dim) > 0 else None,
        delta_num_v_heads=int(args.delta_num_v_heads) if int(args.delta_num_v_heads) > 0 else None,
        delta_expand_v=float(args.delta_expand_v),
        delta_mode=str(args.delta_mode),
        delta_use_short_conv=not bool(args.delta_no_short_conv),
        delta_conv_size=int(args.delta_conv_size),
        delta_norm_eps=float(args.delta_norm_eps),
        attention_backend=str(args.attention_backend),
        strict_backends=bool(args.strict_backends),
        rope_theta=float(args.rope_theta),
        position_embedding_mode=str(args.position_embedding_mode),
        op_order_embedding_mode=str(args.op_order_embedding_mode),
        op_order_max_positions=int(args.op_order_max_positions),
        op_token_ids=range(OP_BASE, OP_BASE + len(OP_SPECS)),
        value_codec=str(args.value_codec),
        halt_pooling=str(args.halt_pooling),
        carrier_gate_init=float(args.carrier_gate_init),
        carrier_state_mode=str(args.carrier_state_mode),
        trm_recurrent_layerscale_mode=str(args.trm_recurrent_layerscale_mode),
        trm_recurrent_layerscale_init=float(args.trm_recurrent_layerscale_init),
    ).to(device)
    optimizer, optimizer_report = build_memory_efficient_optimizer(
        model,
        optimizer_name=str(getattr(args, "optimizer", "adamw")),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=device,
        galore_rank=int(getattr(args, "galore_rank", 128)),
        galore_update_proj_gap=int(getattr(args, "galore_update_proj_gap", 200)),
        galore_scale=float(getattr(args, "galore_scale", 0.25)),
        galore_proj_type=str(getattr(args, "galore_proj_type", "std")),
        galore_min_dim=int(getattr(args, "galore_min_dim", 128)),
        galore_include_embeddings=bool(getattr(args, "galore_include_embeddings", False)),
    )
    prompt_len = len(case_prompt_tokens(train_cases[0]))
    last_loss = 0.0
    for step in range(1, int(args.steps) + 1):
        model.train()
        batch = random.sample(train_cases, k=min(int(args.batch_size), len(train_cases)))
        if bool(args.active_len_curriculum):
            active_len = active_program_len_for_step(
                step=step,
                total_steps=int(args.steps),
                program_len=int(args.program_len),
                min_active_len=int(args.active_len_curriculum_min),
                warmup_fraction=float(args.active_len_curriculum_warmup_frac),
            )
            batch = [
                case_with_active_program_len(
                    case,
                    active_len=active_len,
                    modulus=int(args.modulus),
                )
                for case in batch
            ]
        _, full_tokens = cases_to_batch(batch, device=device)
        input_ids = full_tokens[:, :-1]
        logits = model(input_ids, think_steps=int(args.train_think_steps))
        loss = answer_loss(
            logits,
            full_tokens,
            prompt_len=prompt_len,
            answer_loss_weight=float(args.answer_loss_weight),
            eos_loss_weight=float(args.eos_loss_weight),
            answer_eos_margin_weight=float(args.answer_eos_margin_weight),
            answer_eos_margin=float(args.answer_eos_margin),
        )
        if float(args.depth_intermediate_loss_weight) > 0.0:
            depth_targets = depth_target_tokens(
                batch,
                max_depth=int(args.train_think_steps),
                modulus=int(args.modulus),
                device=device,
            )
            loss = loss + float(args.depth_intermediate_loss_weight) * (
                intermediate_depth_loss(
                    model,
                    input_ids,
                    depth_targets,
                    prompt_len=prompt_len,
                    max_depth=int(args.train_think_steps),
                )
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if int(args.log_every) > 0 and (
            step == 1 or step % int(args.log_every) == 0 or step == int(args.steps)
        ):
            payload: dict[str, object] = {"step": step, "loss": last_loss}
            if step == 1:
                payload["optimizer"] = optimizer_report
            print(json.dumps(payload, ensure_ascii=False))

    eval_metrics = {
        "think0": evaluate(model, eval_cases, args, think_steps=0),
        "think1": evaluate(model, eval_cases, args, think_steps=1),
        f"think{args.eval_think_steps}": evaluate(
            model,
            eval_cases,
            args,
            think_steps=int(args.eval_think_steps),
        ),
        "state_reset": evaluate(
            model,
            eval_cases,
            args,
            think_steps=int(args.eval_think_steps),
            ablation="state_reset",
        ),
        "op_zero": evaluate(
            model,
            eval_cases,
            args,
            think_steps=int(args.eval_think_steps),
            ablation="op_zero",
        ),
        "thinking_block_off": evaluate(
            model,
            eval_cases,
            args,
            think_steps=int(args.eval_think_steps),
            ablation="thinking_block_off",
        ),
    }
    for ablation_name in applicable_ablation_names(str(args.think_structure)):
        if ablation_name in eval_metrics:
            continue
        eval_metrics[ablation_name] = evaluate(
            model,
            eval_cases,
            args,
            think_steps=int(args.eval_think_steps),
            ablation=ablation_name,
        )
    decision = make_decision(eval_metrics, args)
    report: dict[str, object] = {
        "status": "complete",
        "target_level": "L1 QTRM-native ETD/TRM scaffold",
        "major_bottleneck": "donorless native repeated thinking block to LM logits",
        "method_class": "official-prior-inspired minimal reproduction",
        "closest_prior": {
            "paper": "Encode, Think, Decode: Scaling test-time reasoning with recursive latent thoughts",
            "arxiv": "2510.07358",
        },
        "train": vars(args),
        "last_loss": last_loss,
        "optimizer": optimizer_report,
        "eval_metrics": eval_metrics,
        **decision,
    }
    out_dir = Path(args.out_dir)
    if str(out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "args": vars(args),
                "report": report,
                "vocab": {
                    "pad": PAD,
                    "bos": BOS,
                    "eos": EOS,
                    "start": START,
                    "answer": ANS,
                    "op_base": OP_BASE,
                    "value_base": value_base(),
                    "op_specs": OP_SPECS,
                },
            },
            out_dir / "last.pt",
        )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a donorless QTRM-native Encode-Think-Decode probe."
    )
    parser.add_argument("--out-dir", default="local_eval/qtrm_native_etd_probe")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--train-cases", type=int, default=4096)
    parser.add_argument("--eval-cases", type=int, default=256)
    parser.add_argument("--program-len", type=int, default=4)
    parser.add_argument("--modulus", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--backbone",
        choices=SUPPORTED_BACKBONES,
        default="mha_etd",
    )
    parser.add_argument("--encode-backbone", choices=("", *SUPPORTED_BACKBONES), default="")
    parser.add_argument("--think-backbone", choices=("", *SUPPORTED_BACKBONES), default="")
    parser.add_argument("--decode-backbone", choices=("", *SUPPORTED_BACKBONES), default="")
    parser.add_argument("--think-structure", choices=SUPPORTED_THINK_STRUCTURES, default="single")
    parser.add_argument("--trm-l-cycles", type=int, default=1)
    parser.add_argument("--trm-full-grad-cycles", action="store_true")
    parser.add_argument("--hybrid-layers", type=int, default=4)
    parser.add_argument("--attn-every", type=int, default=4)
    parser.add_argument("--delta-backend", default="torch_gated_delta")
    parser.add_argument("--delta-head-dim", type=int, default=0)
    parser.add_argument("--delta-num-v-heads", type=int, default=0)
    parser.add_argument("--delta-expand-v", type=float, default=1.0)
    parser.add_argument("--delta-mode", default="chunk")
    parser.add_argument("--delta-no-short-conv", action="store_true")
    parser.add_argument("--delta-conv-size", type=int, default=4)
    parser.add_argument("--delta-norm-eps", type=float, default=1e-6)
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--strict-backends", action="store_true")
    parser.add_argument("--rope-theta", type=float, default=100000.0)
    parser.add_argument(
        "--position-embedding-mode",
        choices=("learned", "none", "randomized"),
        default="learned",
        help=(
            "Use learned absolute input position embeddings, disable them, "
            "or sample ordered positions from the model context during "
            "training for length-generalization experiments."
        ),
    )
    parser.add_argument(
        "--op-order-embedding-mode",
        choices=("none", "learned"),
        default="none",
        help="Add learned operation-index embeddings to operation tokens.",
    )
    parser.add_argument("--op-order-max-positions", type=int, default=32)
    parser.add_argument(
        "--value-codec",
        choices=("learned", "circular"),
        default="learned",
        help=(
            "Use learned value-token embeddings/readout, or a circular latent "
            "value codec for modular synthetic tasks."
        ),
    )
    parser.add_argument("--halt-pooling", choices=("last", "mean", "dedicated"), default="last")
    parser.add_argument(
        "--carrier-gate-init",
        type=float,
        default=-1.0,
        help=(
            "Initial logit for internal carrier residual gates. Lower values "
            "make carrier insertion closer to identity at startup."
        ),
    )
    parser.add_argument(
        "--carrier-state-mode",
        choices=SUPPORTED_CARRIER_STATE_MODES,
        default="gru",
        help=(
            "Carrier state source. Non-GRU modes are deterministic probes for "
            "reducing random-init dependence in additive carrier experiments."
        ),
    )
    parser.add_argument(
        "--trm-recurrent-layerscale-mode",
        choices=("none", "scalar", "channel"),
        default="none",
        help=(
            "Scale full nested TRM recurrent updates as previous + scale * "
            "(updated - previous). Use init=1.0 to preserve an existing route "
            "or a small init for identity-biased recurrence experiments."
        ),
    )
    parser.add_argument(
        "--trm-recurrent-layerscale-init",
        type=float,
        default=1.0,
        help="Initial scale for --trm-recurrent-layerscale-mode.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--optimizer", choices=MEMORY_EFFICIENT_OPTIMIZERS, default="adamw")
    parser.add_argument("--galore-rank", type=int, default=128)
    parser.add_argument("--galore-update-proj-gap", type=int, default=200)
    parser.add_argument("--galore-scale", type=float, default=0.25)
    parser.add_argument(
        "--galore-proj-type",
        choices=("std", "reverse_std", "right", "left", "full"),
        default="std",
    )
    parser.add_argument("--galore-min-dim", type=int, default=128)
    parser.add_argument("--galore-include-embeddings", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--answer-loss-weight", type=float, default=1.0)
    parser.add_argument("--eos-loss-weight", type=float, default=0.25)
    parser.add_argument("--answer-eos-margin-weight", type=float, default=0.25)
    parser.add_argument("--answer-eos-margin", type=float, default=0.5)
    parser.add_argument("--depth-intermediate-loss-weight", type=float, default=0.0)
    parser.add_argument("--active-len-curriculum", action="store_true")
    parser.add_argument("--active-len-curriculum-min", type=int, default=1)
    parser.add_argument("--active-len-curriculum-warmup-frac", type=float, default=0.5)
    parser.add_argument("--train-think-steps", type=int, default=4)
    parser.add_argument("--eval-think-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=335)
    parser.add_argument("--eval-seed", type=int, default=9335)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-examples", type=int, default=4)
    parser.add_argument("--accept-min-exact", type=float, default=0.80)
    parser.add_argument("--accept-min-depth-gain", type=float, default=0.10)
    parser.add_argument("--accept-min-ablation-drop", type=float, default=0.10)
    parser.add_argument("--accepted-decision", default="accepted_l1_native_etd")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = train_probe(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if bool(report["accepted"]) else 1)


if __name__ == "__main__":
    main()
