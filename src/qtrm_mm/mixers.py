from __future__ import annotations
from typing import Optional
import importlib
import os
from pathlib import Path
import sys
import types
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
        def add_local_fla_reference_path() -> None:
            if os.environ.get("QTRM_DISABLE_LOCAL_FLA_REFERENCE") == "1":
                return
            repo_root = Path(__file__).resolve().parents[2]
            fla_root = repo_root / "references" / "official" / "flash-linear-attention"
            if fla_root.exists() and str(fla_root) not in sys.path:
                sys.path.insert(0, str(fla_root))

        def load_symbol(candidates: list[tuple[str, str]]):
            errors = []
            for attempt in range(2):
                for module_name, symbol_name in candidates:
                    try:
                        module = importlib.import_module(module_name)
                        return getattr(module, symbol_name)
                    except Exception as exc:
                        errors.append(f"{module_name}.{symbol_name}: {type(exc).__name__}: {exc}")
                if attempt == 0:
                    add_local_fla_reference_path()
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


class OfficialMamba3Mixer(nn.Module):
    """Adapter for the official state-spaces Mamba-3 module.

    The upstream package imports legacy Mamba CUDA extensions from
    ``mamba_ssm.__init__`` before reaching ``modules.mamba3``. For local
    reference experiments we load the package path directly so the Mamba-3
    Triton module can be tested independently.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        strict: bool = False,
        fallback_dropout: float = 0.0,
        d_state: int = 128,
        expand: int = 2,
        headdim: int | None = None,
        ngroups: int = 1,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.strict = bool(strict)
        fallback = TorchGatedDeltaMixer(
            d_model=d_model,
            n_heads=n_heads,
            dropout=fallback_dropout,
        )
        self.runtime_fallback = fallback
        self.impl = self._build_impl(
            d_model=int(d_model),
            n_heads=int(n_heads),
            d_state=int(d_state),
            expand=int(expand),
            headdim=int(headdim) if headdim else None,
            ngroups=int(ngroups),
            chunk_size=int(chunk_size),
        )
        self.is_official_backend = self.impl is not None
        if self.impl is None:
            if self.strict:
                raise RuntimeError("Requested official Mamba-3, but no compatible implementation was found")
            self.impl = self.runtime_fallback
            object.__setattr__(self, "_cpu_fallback", None)
        else:
            object.__setattr__(self, "_cpu_fallback", self.runtime_fallback)

    def _build_impl(
        self,
        *,
        d_model: int,
        n_heads: int,
        d_state: int,
        expand: int,
        headdim: int | None,
        ngroups: int,
        chunk_size: int,
    ):
        def patch_triton_descriptor_alias() -> None:
            try:
                import triton.language as tl
            except Exception:
                return
            if not hasattr(tl, "make_tensor_descriptor") and hasattr(
                tl,
                "_experimental_make_tensor_descriptor",
            ):
                tl.make_tensor_descriptor = tl._experimental_make_tensor_descriptor

        def add_local_mamba_reference_path() -> Path | None:
            if os.environ.get("QTRM_DISABLE_LOCAL_MAMBA_REFERENCE") == "1":
                return None
            repo_root = Path(__file__).resolve().parents[2]
            mamba_root = repo_root / "references" / "official" / "mamba"
            mamba_pkg = mamba_root / "mamba_ssm"
            if mamba_root.exists() and str(mamba_root) not in sys.path:
                sys.path.insert(0, str(mamba_root))
            return mamba_pkg if mamba_pkg.exists() else None

        def install_lightweight_mamba_package(mamba_pkg: Path) -> None:
            sys.modules.pop("mamba_ssm", None)
            package = types.ModuleType("mamba_ssm")
            package.__path__ = [str(mamba_pkg)]
            package.__version__ = "2.3.2.post1"
            sys.modules["mamba_ssm"] = package

        patch_triton_descriptor_alias()
        errors: list[str] = []
        try:
            module = importlib.import_module("mamba_ssm.modules.mamba3")
            cls = getattr(module, "Mamba3")
        except Exception as exc:
            errors.append(f"normal import: {type(exc).__name__}: {exc}")
            mamba_pkg = add_local_mamba_reference_path()
            if mamba_pkg is None:
                if self.strict:
                    raise RuntimeError("Could not find local official Mamba reference") from exc
                return None
            try:
                install_lightweight_mamba_package(mamba_pkg)
                module = importlib.import_module("mamba_ssm.modules.mamba3")
                cls = getattr(module, "Mamba3")
            except Exception as local_exc:
                errors.append(f"local reference import: {type(local_exc).__name__}: {local_exc}")
                if self.strict:
                    detail = "\n".join(f"  - {item}" for item in errors)
                    raise RuntimeError(f"Could not import official Mamba-3 backend\n{detail}") from local_exc
                return None

        inner = int(expand) * int(d_model)
        if headdim is None:
            head_dim = max(16, int(d_model) // max(1, int(n_heads)))
        else:
            head_dim = int(headdim)
        if inner % head_dim != 0:
            head_dim = max(1, inner // max(1, int(n_heads)))
        return cls(
            d_model=int(d_model),
            d_state=int(d_state),
            expand=int(expand),
            headdim=int(head_dim),
            ngroups=int(ngroups),
            chunk_size=int(chunk_size),
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        del attention_mask
        if x.device.type == "cpu" and self.is_official_backend and not self.strict:
            fallback = getattr(self, "_cpu_fallback", None)
            if fallback is not None:
                return fallback(x)
        try:
            return self.impl(x)
        except Exception:
            if self.strict:
                raise
            fallback = self.runtime_fallback.to(device=x.device, dtype=x.dtype)
            self.impl = fallback
            object.__setattr__(self, "_runtime_fallback_active", True)
            return self.impl(x)


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
