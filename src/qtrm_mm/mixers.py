from __future__ import annotations
from typing import Optional
import importlib
import torch
from torch import nn
import torch.nn.functional as F


class TorchGatedDeltaMixer(nn.Module):
    """Trainable PyTorch reference backend for gated recurrent mixing.

    This is not official KDA. It is a bounded recurrent mixer that allows
    architecture/debug training without external kernels. Production runs should
    use FLA KDA/GatedDelta backends through strict mode.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, d = x.shape
        u, v, decay = self.in_proj(x).chunk(3, dim=-1)
        u = torch.tanh(u)
        v = torch.tanh(v)
        decay = torch.sigmoid(decay)
        gate = torch.sigmoid(self.gate_proj(x))

        # sequential reference implementation; intentionally simple.
        state = torch.zeros(b, d, dtype=x.dtype, device=x.device)
        outs = []
        mask = attention_mask.to(x.dtype) if attention_mask is not None else None
        for i in range(t):
            if mask is not None:
                m = mask[:, i : i + 1]
            else:
                m = 1.0
            state = decay[:, i] * state + (1.0 - decay[:, i]) * u[:, i] * m
            y = gate[:, i] * v[:, i] + (1.0 - gate[:, i]) * state
            outs.append(y)
        y = torch.stack(outs, dim=1)
        return self.out_proj(self.dropout(y))


class FLADeltaMixer(nn.Module):
    """Adapter placeholder for FLA KDA/GatedDelta backends.

    The adapter performs lazy imports and attempts several common class names.
    If the installed FLA API differs, edit _build_impl only; the QTRM block
    interface remains unchanged.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        backend: str,
        strict: bool = False,
        fallback_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.backend = backend
        self.strict = strict
        self.impl = self._build_impl(d_model, n_heads, backend, strict, **kwargs)
        self.is_official_backend = self.impl is not None
        if self.impl is None:
            if strict:
                raise RuntimeError(f"Requested {backend}, but no compatible FLA implementation was found")
            self.impl = TorchGatedDeltaMixer(d_model=d_model, n_heads=n_heads, dropout=fallback_dropout)

    def _build_impl(self, d_model: int, n_heads: int, backend: str, strict: bool, **kwargs):
        def load_symbol(candidates: list[tuple[str, str]]):
            errors = []
            for module_name, symbol_name in candidates:
                try:
                    module = importlib.import_module(module_name)
                    return getattr(module, symbol_name)
                except Exception as exc:
                    errors.append(f"{module_name}.{symbol_name}: {type(exc).__name__}: {exc}")
            if strict:
                detail = "\n".join(f"  - {item}" for item in errors)
                raise RuntimeError(f"Could not import FLA backend {backend}\n{detail}")
            return None

        try:
            if backend == "fla_kda":
                # FLA APIs have changed across versions; keep this adapter explicit.
                cls = load_symbol([
                    ("fla.layers", "KimiDeltaAttention"),
                    ("fla.layers.kda", "KimiDeltaAttention"),
                    ("fla.layers.kimi_delta_attn", "KimiDeltaAttention"),
                ])
                if cls is None:
                    return None
                return cls(hidden_size=d_model, num_heads=n_heads, **kwargs)
            if backend == "fla_gated_delta":
                # Official GatedDeltaNet/FLA README path is:
                #     from fla.layers import GatedDeltaNet
                cls = load_symbol([
                    ("fla.layers", "GatedDeltaNet"),
                    ("fla.layers.gated_deltanet", "GatedDeltaNet"),
                    ("fla.layers.gated_delta_net", "GatedDeltaNet"),
                ])
                if cls is None:
                    return None
                return cls(hidden_size=d_model, num_heads=n_heads, **kwargs)
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Could not import FLA backend {backend}") from exc
        return None

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        try:
            out = self.impl(x, attention_mask=attention_mask)
            if isinstance(out, tuple):
                out = out[0]
            return out
        except TypeError:
            out = self.impl(x)
            if isinstance(out, tuple):
                out = out[0]
            return out


def build_delta_mixer(d_model: int, n_heads: int, backend: str, strict: bool, dropout: float = 0.0, **kwargs):
    if backend == "torch_gated_delta":
        return TorchGatedDeltaMixer(d_model, n_heads, dropout=dropout)
    if backend in {"fla_kda", "fla_gated_delta"}:
        return FLADeltaMixer(
            d_model,
            n_heads,
            backend=backend,
            strict=strict,
            fallback_dropout=dropout,
            **kwargs,
        )
    raise ValueError(f"Unknown delta backend: {backend}")
