from __future__ import annotations
from typing import Optional
import importlib
import inspect
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


class OfficialGatedDeltaNet2Mixer(nn.Module):
    """Adapter for NVlabs/GatedDeltaNet-2 official implementation.

    The official repository is a LiTGPT training tree, not a small installed
    package.  Load ``lit_gpt/gdn2.py`` directly so importing the mixer does not
    pull optional training dependencies such as Lightning.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        strict: bool = False,
        fallback_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.strict = bool(strict)
        self.impl = self._build_impl(d_model=d_model, n_heads=n_heads, strict=bool(strict), **kwargs)
        self.is_official_backend = self.impl is not None
        if self.impl is None:
            raise RuntimeError(
                "Requested official GatedDeltaNet-2, but no compatible implementation was found; "
                "fallback is disabled for official_gated_delta2."
            )
        object.__setattr__(self, "_runtime_fallback_active", False)

    def _build_impl(self, *, d_model: int, n_heads: int, strict: bool, **kwargs):
        repo_root = Path(__file__).resolve().parents[2]
        fla_root = repo_root / "references" / "official" / "flash-linear-attention"
        gdn2_fla_root = repo_root / "references" / "official" / "flash-linear-attention-gdn2"
        gdn2_root = repo_root / "references" / "official" / "gated-deltanet-2"
        lit_gpt_root = gdn2_root / "lit_gpt"
        errors: list[str] = []

        fla_roots = [root for root in (gdn2_fla_root, fla_root) if root.exists()]
        for root in reversed(fla_roots):
            root_str = str(root)
            sys.path[:] = [path for path in sys.path if path != root_str]
            sys.path.insert(0, root_str)

        preferred_fla_root = gdn2_fla_root if gdn2_fla_root.exists() else fla_root
        loaded_fla = sys.modules.get("fla")
        loaded_fla_file = str(getattr(loaded_fla, "__file__", "")) if loaded_fla is not None else ""
        if loaded_fla is not None and preferred_fla_root.exists() and not loaded_fla_file.startswith(str(preferred_fla_root)):
            for name in list(sys.modules):
                if name == "fla" or name.startswith("fla."):
                    sys.modules.pop(name, None)
        if str(gdn2_root) not in sys.path and gdn2_root.exists():
            sys.path.insert(0, str(gdn2_root))
        if not lit_gpt_root.exists():
            if strict:
                raise RuntimeError(f"Official GatedDeltaNet-2 reference not found at {gdn2_root}")
            return None
        try:
            package = sys.modules.get("lit_gpt")
            if package is None or not hasattr(package, "__path__"):
                package = types.ModuleType("lit_gpt")
                package.__path__ = [str(lit_gpt_root)]
                sys.modules["lit_gpt"] = package
            spec = importlib.util.spec_from_file_location("lit_gpt.gdn2", lit_gpt_root / "gdn2.py")
            if spec is None or spec.loader is None:
                raise ImportError(f"could not create import spec for {lit_gpt_root / 'gdn2.py'}")
            module = importlib.util.module_from_spec(spec)
            sys.modules["lit_gpt.gdn2"] = module
            spec.loader.exec_module(module)
            self._patch_official_gdn2_fla_compat()
            cls = getattr(module, "GatedDeltaNet2")
        except Exception as exc:
            errors.append(f"load official GatedDeltaNet2: {type(exc).__name__}: {exc}")
            if strict:
                detail = "\n".join(f"  - {item}" for item in errors)
                raise RuntimeError(f"Could not import official GatedDeltaNet-2\n{detail}") from exc
            return None

        head_dim = int(kwargs.pop("head_dim", None) or max(1, int(d_model) // max(1, int(n_heads))))
        num_v_heads = int(kwargs.pop("num_v_heads", None) or int(n_heads))
        expand_v = float(kwargs.pop("expand_v", 1.0))
        mode = str(kwargs.pop("mode", "chunk"))
        use_short_conv = bool(kwargs.pop("use_short_conv", True))
        conv_size = int(kwargs.pop("conv_size", 4))
        norm_eps = float(kwargs.pop("norm_eps", 1e-5))
        return cls(
            hidden_size=int(d_model),
            num_heads=int(n_heads),
            num_v_heads=num_v_heads,
            head_dim=head_dim,
            expand_v=expand_v,
            mode=mode,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            norm_eps=norm_eps,
            **kwargs,
        )

    @staticmethod
    def _patch_official_gdn2_fla_compat() -> None:
        """Bridge minor public GDN2/FLA API skew without changing GDN2 math."""

        try:
            chunk_module = importlib.import_module("lit_gpt.gdn2_ops.chunk_gdn2")
            target = getattr(chunk_module, "chunk_gla_fwd_o_gk")
            if getattr(target, "_qtrm_gdn2_kwargs_compat", False):
                return
            signature = inspect.signature(target)
            accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
            if accepts_kwargs:
                return
            supported = set(signature.parameters)
            passthrough_target = target

            def wrapped_chunk_gla_fwd_o_gk(*args, **kwargs):
                kwargs = {key: value for key, value in kwargs.items() if key in supported}
                return passthrough_target(*args, **kwargs)

            wrapped_chunk_gla_fwd_o_gk.__name__ = getattr(target, "__name__", "chunk_gla_fwd_o_gk")
            wrapped_chunk_gla_fwd_o_gk.__doc__ = getattr(target, "__doc__", None)
            wrapped_chunk_gla_fwd_o_gk._qtrm_gdn2_kwargs_compat = True
            setattr(chunk_module, "chunk_gla_fwd_o_gk", wrapped_chunk_gla_fwd_o_gk)
        except Exception:
            return

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.impl(x, attention_mask=attention_mask)
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
    if backend in {"official_gated_delta2", "official_gdn2"}:
        return OfficialGatedDeltaNet2Mixer(
            d_model=d_model,
            n_heads=n_heads,
            strict=True,
            fallback_dropout=dropout,
            **kwargs,
        )
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

# ============================================================
# Gating v2 (2026-05-30) - Torch Reference Implementation
# Based on: ReGLA (arXiv:2502.01578), RWKV-7 (arXiv:2503.14456),
#           Gated DeltaNet improvements (2025-2026)
#
# This is a drop-in improved version of TorchGatedDeltaMixer
# with:
#   - Vector-valued gating (per dimension)
#   - Refined delta rule + in-context learning rate
#   - Additional normalization for long recurrence stability
#
# Fully One-Body compatible. No side organs.
# ============================================================

class TorchGatedDeltaNet2MixerV2(nn.Module):
    """
    Improved PyTorch reference for Gated DeltaNet-2 style recurrence (v2).

    Key upgrades over original TorchGatedDeltaMixer:
    - Vector-valued gating (per-channel)
    - Explicit in-context learning rate in the delta update
    - Better normalization for training stability at depth

    This version is intended for architecture exploration and debugging.
    Production use should eventually move to optimized FLA / official kernels.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Expanded projection: u, v, decay, in_context_lr
        self.in_proj = nn.Linear(d_model, 4 * d_model, bias=False)

        # Vector-valued gate (per dimension)
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)

        # Optional: separate forget gate (RWKV-7 style vector forgetting)
        # Can be enabled later via config
        self.use_vector_forget = False
        if self.use_vector_forget:
            self.forget_proj = nn.Linear(d_model, d_model, bias=True)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Extra normalization for long recurrence stability (ReGLA-inspired)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # RI-4 A-Mode robustness for hybrid recurrent engine:
        # answer_state_loop does .unsqueeze(1) → (B, 1, D) for the recurrent proposal.
        # Support 2D/3D inputs without breaking the unpack.
        if x.dim() == 2:
            x = x.unsqueeze(1)
        while x.dim() > 3:
            x = x.squeeze(1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        b, t, d = x.shape

        proj = self.in_proj(x)
        u, v, decay, in_context_lr = proj.chunk(4, dim=-1)

        u = torch.tanh(u)
        v = torch.tanh(v)
        decay = torch.sigmoid(decay)
        in_context_lr = torch.sigmoid(in_context_lr)

        # Vector gate (per dimension)
        gate = torch.sigmoid(self.gate_proj(x))

        # Optional vector forget gate
        forget = None
        if self.use_vector_forget and hasattr(self, 'forget_proj'):
            forget = torch.sigmoid(self.forget_proj(x))

        # Sequential reference (for correctness & debugging)
        state = torch.zeros(b, d, dtype=x.dtype, device=x.device)
        outs = []
        mask = attention_mask.to(x.dtype) if attention_mask is not None else None

        for i in range(t):
            m = mask[:, i : i + 1] if mask is not None else 1.0

            # Refined delta update with in-context learning rate
            update = in_context_lr[:, i] * u[:, i] * m

            if forget is not None:
                state = forget[:, i] * state + (1.0 - forget[:, i]) * update
            else:
                state = decay[:, i] * state + (1.0 - decay[:, i]) * update

            y = gate[:, i] * v[:, i] + (1.0 - gate[:, i]) * state
            outs.append(y)

        y = torch.stack(outs, dim=1)
        y = self.norm(y)                    # Stability normalization
        return self.out_proj(self.dropout(y))


# =============================================================================
# Optional: Sparse Slot Router integration (for RI-4 PoC)
# =============================================================================
# This allows TorchGatedDeltaNet2MixerV2 to use the new Raven/MSA-style
# sparse persistent slots. The router is intentionally optional and has
# perfect ablation hooks so it can be turned on/off cleanly during experiments.
#
# When enabled:
#   - A small number of persistent slots live alongside the dense state.
#   - The router (from the new sparse_slot_router module) decides which slots
#     participate in the current update.
#   - Non-selected slots get near-perfect persistence (key for long-horizon
#     raw intelligence stability).
#   - Stochastic breadth noise can be injected into the router for exploration.
#
# This is the concrete first implementation step for RI-4.
# =============================================================================

try:
    from .memory.sparse_slot_router import SparseSlotRouter
except Exception:
    SparseSlotRouter = None  # graceful fallback if module not present


def _add_sparse_slot_router_support(mixer_cls):
    """
    Monkey-patch style extension to add optional sparse slot router
    to any DeltaNet-style mixer that maintains a recurrent state.

    In practice we call this on TorchGatedDeltaNet2MixerV2 after definition.
    """
    original_forward = mixer_cls.forward

    def forward_with_sparse_router(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        stochastic_breadth_noise: Optional[torch.Tensor] = None,
        use_sparse_slots: bool = False,
        slot_router: Optional["SparseSlotRouter"] = None,
    ) -> torch.Tensor:
        # RI-4 A-Mode shape normalization (same as base mixer)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        while x.dim() > 3:
            x = x.squeeze(1)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if not use_sparse_slots or slot_router is None or SparseSlotRouter is None:
            return original_forward(self, x, attention_mask=attention_mask)

        # Run normal dense recurrence first (keeps backward compatibility)
        y_dense = original_forward(self, x, attention_mask=attention_mask)

        # Ask the router for a read signal + selection mask
        # We use the last timestep of the input as query context
        read_signal, slot_mask, _ = slot_router(
            x,
            stochastic_noise=stochastic_breadth_noise,
        )

        # Simple gated fusion of the sparse memory read into the dense output.
        # This keeps everything inside the One-Body path.
        # The caller (hybrid block) can later make this fusion more sophisticated
        # using the existing vector-valued gate.
        gate = torch.sigmoid(self.gate_proj(x) if hasattr(self, 'gate_proj') else 0.1)
        if gate.dim() == 2:
            gate = gate.unsqueeze(1)
        y = y_dense + gate * read_signal.unsqueeze(1) if y_dense.dim() == 3 else y_dense + gate * read_signal

        # NOTE: True selective *write* to slots (Raven persistence) should be
        # performed in the rehearsal logic or a dedicated memory manager step
        # that also receives the slot_mask. This forward only demonstrates the
        # read + causal injection path for the first RI-4 smoke.

        return y

    mixer_cls.forward = forward_with_sparse_router
    mixer_cls._supports_sparse_slots = True
    return mixer_cls


# Apply the extension to the active V2 mixer
if TorchGatedDeltaNet2MixerV2 is not None:
    TorchGatedDeltaNet2MixerV2 = _add_sparse_slot_router_support(TorchGatedDeltaNet2MixerV2)
