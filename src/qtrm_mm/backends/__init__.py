"""Backend registry for QTRM attention and delta backends.

Available backends:
  attention: sdpa, flash_attn
  delta:     torch_gated_delta, fla_kda, fla_gated_delta

FlashAttention is optional; if not installed, falls back to SDPA.
"""
from __future__ import annotations
import importlib

_HAS_FLASH_ATTN = False
try:
    from flash_attn import flash_attn_func
    _HAS_FLASH_ATTN = True
except ImportError:
    pass

def _can_import_symbol(module_name: str, symbol_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
        getattr(module, symbol_name)
        return True
    except Exception:
        return False


_HAS_FLA_GATED_DELTA = any(
    _can_import_symbol(module_name, "GatedDeltaNet")
    for module_name in (
        "fla.layers",
        "fla.layers.gated_deltanet",
        "fla.layers.gated_delta_net",
    )
)
_HAS_FLA_KDA = any(
    _can_import_symbol(module_name, "KimiDeltaAttention")
    for module_name in (
        "fla.layers",
        "fla.layers.kda",
        "fla.layers.kimi_delta_attn",
    )
)
_HAS_FLA = _HAS_FLA_GATED_DELTA or _HAS_FLA_KDA


def get_attention_backend(name: str):
    """Return the attention function for the given backend name."""
    if name == "sdpa":
        import torch.nn.functional as F
        return F.scaled_dot_product_attention
    if name == "flash_attn":
        if not _HAS_FLASH_ATTN:
            import warnings
            warnings.warn(
                "flash-attn not installed, falling back to SDPA. "
                "Install with: pip install flash-attn --no-build-isolation",
                UserWarning,
            )
            import torch.nn.functional as F
            return F.scaled_dot_product_attention
        return _flash_attn_wrapper
    raise ValueError(f"Unknown attention backend: {name}")


def _flash_attn_wrapper(q, k, v, attn_mask=None, dropout_p=0.0):
    """Wrapper to make flash_attn_func compatible with SDPA signature.

    q, k, v: (B, H, S, D)
    Returns: (B, H, S, D)
    """
    from flash_attn import flash_attn_func
    q2 = q.transpose(1, 2).contiguous()
    k2 = k.transpose(1, 2).contiguous()
    v2 = v.transpose(1, 2).contiguous()
    out, _ = flash_attn_func(
        q2, k2, v2,
        dropout_p=dropout_p if dropout_p > 0 else 0.0,
        causal=False,
    )
    return out.transpose(1, 2)


def get_delta_backend(name: str):
    """Return the delta mixer factory for the given backend name."""
    if name == "torch_gated_delta":
        from ..mixers import TorchGatedDeltaMixer
        return TorchGatedDeltaMixer
    if name in {"fla_kda", "fla_gated_delta"}:
        if name == "fla_kda" and not _HAS_FLA_KDA:
            import warnings
            warnings.warn(
                "FLA KDA backend is not installed, falling back to torch_gated_delta. "
                "Install with: pip install flash-linear-attention",
                UserWarning,
            )
            from ..mixers import TorchGatedDeltaMixer
            return TorchGatedDeltaMixer
        if name == "fla_gated_delta" and not _HAS_FLA_GATED_DELTA:
            import warnings
            warnings.warn(
                "FLA GatedDeltaNet backend is not installed, falling back to torch_gated_delta. "
                "Install with: pip install flash-linear-attention",
                UserWarning,
            )
            from ..mixers import TorchGatedDeltaMixer
            return TorchGatedDeltaMixer
        from ..mixers import FLADeltaMixer
        return FLADeltaMixer
    raise ValueError(f"Unknown delta backend: {name}")


def check_strict_backends(cfg):
    """Validate backend requirements in strict mode."""
    errors = []
    attn = getattr(cfg, "attention_backend", "sdpa")
    delta = getattr(cfg, "delta_backend", "torch_gated_delta")
    strict = getattr(cfg, "strict_backends", False)

    if strict:
        if attn == "flash_attn" and not _HAS_FLASH_ATTN:
            errors.append("flash-attn required but not installed")
        if delta == "fla_kda" and not _HAS_FLA_KDA:
            errors.append("FLA KDA backend required but not installed")
        if delta == "fla_gated_delta" and not _HAS_FLA_GATED_DELTA:
            errors.append("FLA GatedDeltaNet backend required but not installed")

    if errors:
        raise RuntimeError(
            "Strict backend check failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )


HAS_FLASH_ATTN = _HAS_FLASH_ATTN
HAS_FLA = _HAS_FLA
HAS_FLA_GATED_DELTA = _HAS_FLA_GATED_DELTA
HAS_FLA_KDA = _HAS_FLA_KDA

__all__ = [
    "get_attention_backend",
    "get_delta_backend",
    "check_strict_backends",
    "HAS_FLASH_ATTN",
    "HAS_FLA",
    "HAS_FLA_GATED_DELTA",
    "HAS_FLA_KDA",
]
