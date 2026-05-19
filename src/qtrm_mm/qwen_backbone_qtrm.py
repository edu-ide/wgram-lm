from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import torch
from torch import nn

from .config import QTRMConfig
from .core import QTRMRecursiveCore
from .norm import RMSNorm


@dataclass(frozen=True)
class QwenBackboneQTRMReport:
    model_id: str
    vocab_size: int
    hidden_size: int
    qwen_parameters: int
    qwen_trainable_parameters: int
    qtrm_parameters: int
    qtrm_trainable_parameters: int
    runtime_donor: bool = False
    integrated_qwen_backbone: bool = True
    standalone_graph: bool = True
    mandatory_core: bool = False
    core_impl: str = ""
    normal_core_gate: float = 0.0
    qwen_core_layers_cloned: bool = False
    core_insertion_mode: str = "final_residual"
    core_insert_after_layer: int = -1
    core_residual_gate_mode: str = "constant"


def _text_config(config: Any) -> Any:
    return getattr(config, "text_config", config)


def _config_int(config: Any, name: str, default: int) -> int:
    value = getattr(_text_config(config), name, None)
    if value is None:
        value = getattr(config, name, None)
    return int(default if value is None else value)


def _count_parameters(
    module: nn.Module,
    *,
    trainable_only: bool = False,
    exclude_ids: Optional[set[int]] = None,
) -> int:
    exclude_ids = exclude_ids or set()
    return sum(
        int(p.numel())
        for p in module.parameters()
        if id(p) not in exclude_ids and (p.requires_grad or not trainable_only)
    )


def _get_attr_path(root: Any, path: str) -> Any:
    current = root
    if path == "":
        return current
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _find_qwen_text_model(qwen_model: nn.Module) -> nn.Module:
    for path in (
        "model.language_model",
        "language_model",
        "model",
        "",
    ):
        candidate = _get_attr_path(qwen_model, path)
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise ValueError("could not find Qwen text model layers")


def _find_ouro_text_model(ouro_model: nn.Module) -> nn.Module:
    for path in (
        "model",
        "",
    ):
        candidate = _get_attr_path(ouro_model, path)
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise ValueError("could not find Ouro text model layers")


def _first_parameter_dtype(module: nn.Module, default: torch.dtype) -> torch.dtype:
    for parameter in module.parameters(recurse=True):
        return parameter.dtype
    return default


def _default_qwen_core_layer_indices(text_model: nn.Module) -> tuple[int, ...]:
    layers = getattr(text_model, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("Qwen text model has no decoder layers")
    layer_types = list(getattr(getattr(text_model, "config", None), "layer_types", []) or [])
    full_attention = [
        idx
        for idx, layer_type in enumerate(layer_types[: len(layers)])
        if str(layer_type) == "full_attention"
    ]
    if full_attention:
        return (int(full_attention[0]),)
    return (len(layers) // 2,)


def _default_ouro_core_layer_indices(text_model: nn.Module) -> tuple[int, ...]:
    layers = getattr(text_model, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("Ouro text model has no decoder layers")
    return (len(layers) // 2,)


def _normalise_layer_indices(
    indices: Optional[Sequence[int]],
    *,
    num_layers: int,
    default: Sequence[int],
) -> tuple[int, ...]:
    values = tuple(int(i) for i in (default if indices is None else indices))
    if not values:
        raise ValueError("at least one Qwen core layer index is required")
    for value in values:
        if value < 0 or value >= int(num_layers):
            raise ValueError(f"Qwen core layer index out of range: {value}")
    return values


def _fallback_causal_mask(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    b, t, _ = hidden_states.shape
    dtype = hidden_states.dtype
    device = hidden_states.device
    causal = torch.zeros((t, t), dtype=dtype, device=device)
    causal = causal.masked_fill(
        torch.ones((t, t), dtype=torch.bool, device=device).triu(1),
        torch.finfo(dtype).min,
    )
    mask = causal[None, None, :, :]
    if attention_mask is not None:
        key_mask = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
        mask = mask + key_mask
    return mask.expand(b, 1, t, t)


def _relative_state_delta(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    diff = (current.float() - previous.float()).pow(2).mean(dim=(1, 2)).sqrt()
    denom = previous.float().pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-6)
    return diff / denom


class QwenLayerWrappedStack(nn.Module):
    """Reuse pretrained Qwen decoder layers as recurrent transition blocks."""

    def __init__(
        self,
        text_model: nn.Module,
        layer_indices: Sequence[int],
        *,
        force_causal: bool = True,
        clone_layers: bool = False,
        trainable_clones: bool = True,
    ) -> None:
        super().__init__()
        layers = getattr(text_model, "layers", None)
        if layers is None or len(layers) == 0:
            raise ValueError("Qwen text model has no decoder layers")
        self.text_model = text_model
        self.layer_indices = tuple(int(i) for i in layer_indices)
        self.clone_layers = bool(clone_layers)
        if self.clone_layers:
            self.layers = nn.ModuleList([copy.deepcopy(layers[i]) for i in self.layer_indices])
            for parameter in self.layers.parameters():
                parameter.requires_grad_(bool(trainable_clones))
        else:
            self.layers = nn.ModuleList([layers[i] for i in self.layer_indices])
        self.force_causal = bool(force_causal)
        layer_types = list(getattr(getattr(text_model, "config", None), "layer_types", []) or [])
        self.layer_types = tuple(
            str(layer_types[i]) if i < len(layer_types) else str(getattr(layers[i], "layer_type", "full_attention"))
            for i in self.layer_indices
        )

    def _position_context(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        b, t, _ = hidden_states.shape
        position_ids = torch.arange(t, device=hidden_states.device, dtype=torch.long)
        text_position_ids = position_ids.view(1, -1).expand(b, -1)
        rotary_ids = position_ids.view(1, 1, -1).expand(3, b, -1)
        rotary = getattr(self.text_model, "rotary_emb", None)
        if rotary is None:
            return (hidden_states, hidden_states), text_position_ids
        return rotary(hidden_states, rotary_ids), text_position_ids

    def _causal_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        text_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask

            config = getattr(self.text_model, "config", None)
            if config is not None:
                return create_causal_mask(
                    config=config,
                    inputs_embeds=hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    position_ids=text_position_ids,
                )
        except Exception:
            pass
        return _fallback_causal_mask(hidden_states, attention_mask)

    def _linear_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        cache_position: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        updater = getattr(self.text_model, "_update_linear_attn_mask", None)
        if updater is None:
            return attention_mask
        try:
            return updater(attention_mask, cache_position)
        except (AttributeError, TypeError):
            try:
                return updater(attention_mask, None)
            except TypeError:
                return updater(attention_mask)
        except IndexError:
            return updater(attention_mask)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        layer_dtype = _first_parameter_dtype(self.layers[0], input_dtype)
        hidden_states = x.to(dtype=layer_dtype)
        position_embeddings, text_position_ids = self._position_context(hidden_states)
        causal_mask = self._causal_mask(hidden_states, attention_mask, text_position_ids)
        linear_mask = self._linear_attention_mask(attention_mask, text_position_ids[0])
        for layer, layer_type in zip(self.layers, self.layer_types):
            if str(layer_type) == "linear_attention" and not self.force_causal:
                layer_mask = linear_mask
            elif str(layer_type) == "linear_attention" and linear_mask is not None:
                layer_mask = linear_mask
            else:
                layer_mask = causal_mask
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        return hidden_states.to(dtype=input_dtype)


class OuroLayerWrappedStack(nn.Module):
    """Reuse pretrained Ouro decoder layers as recurrent transition blocks."""

    def __init__(
        self,
        text_model: nn.Module,
        layer_indices: Sequence[int],
    ) -> None:
        super().__init__()
        layers = getattr(text_model, "layers", None)
        if layers is None or len(layers) == 0:
            raise ValueError("Ouro text model has no decoder layers")
        self.text_model = text_model
        self.layer_indices = tuple(int(i) for i in layer_indices)
        self.layers = nn.ModuleList([layers[i] for i in self.layer_indices])
        layer_types = list(getattr(getattr(text_model, "config", None), "layer_types", []) or [])
        self.layer_types = tuple(
            str(layer_types[i])
            if i < len(layer_types)
            else str(getattr(layers[i], "attention_type", "full_attention"))
            for i in self.layer_indices
        )

    def _position_context(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        b, t, _ = hidden_states.shape
        cache_position = torch.arange(t, device=hidden_states.device, dtype=torch.long)
        position_ids = cache_position.view(1, -1).expand(b, -1)
        rotary = getattr(self.text_model, "rotary_emb", None)
        if rotary is None:
            return (hidden_states, hidden_states), position_ids, cache_position
        return rotary(hidden_states, position_ids), position_ids, cache_position

    def _causal_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        try:
            from transformers.masking_utils import create_causal_mask

            config = getattr(self.text_model, "config", None)
            if config is not None:
                return create_causal_mask(
                    config=config,
                    inputs_embeds=hidden_states,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=None,
                    position_ids=position_ids,
                )
        except Exception:
            pass
        return _fallback_causal_mask(hidden_states, attention_mask)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        layer_dtype = _first_parameter_dtype(self.layers[0], input_dtype)
        hidden_states = x.to(dtype=layer_dtype)
        position_embeddings, position_ids, cache_position = self._position_context(hidden_states)
        causal_mask = self._causal_mask(hidden_states, attention_mask, position_ids, cache_position)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                current_ut=0,
            )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        return hidden_states.to(dtype=input_dtype)


class QwenLayerWrappedRecursiveCore(nn.Module):
    """Dual z_L/z_H recurrent core using pretrained Qwen layers as transitions."""

    def __init__(
        self,
        cfg: QTRMConfig,
        qwen_model: nn.Module,
        *,
        layer_indices: Optional[Sequence[int]] = None,
        shared_stack: bool = False,
        clone_layers: bool = False,
        trainable_clones: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        text_model = _find_qwen_text_model(qwen_model)
        layers = getattr(text_model, "layers")
        selected = _normalise_layer_indices(
            layer_indices,
            num_layers=len(layers),
            default=_default_qwen_core_layer_indices(text_model),
        )
        self.layer_indices = selected
        self.shared_stack = bool(shared_stack)
        self.z_l_init = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.z_h_init = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.norm_l = RMSNorm(cfg.d_model)
        self.norm_h = RMSNorm(cfg.d_model)
        self.step_conditioning = (
            nn.Embedding(max(1, int(cfg.core_step_conditioning_max_steps)), cfg.d_model)
            if bool(getattr(cfg, "core_step_conditioning_enabled", False))
            else None
        )
        if self.step_conditioning is not None:
            nn.init.normal_(self.step_conditioning.weight, mean=0.0, std=0.02)
        self.fast_stack = QwenLayerWrappedStack(
            text_model,
            selected,
            force_causal=bool(getattr(cfg, "core_causal", True)),
            clone_layers=bool(clone_layers),
            trainable_clones=bool(trainable_clones),
        )
        if self.shared_stack:
            self.slow_stack = self.fast_stack
        else:
            self.slow_stack = QwenLayerWrappedStack(
                text_model,
                selected,
                force_causal=bool(getattr(cfg, "core_causal", True)),
                clone_layers=bool(clone_layers),
                trainable_clones=bool(trainable_clones),
            )

    def forward(
        self,
        workspace: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        del kwargs
        b = workspace.shape[0]
        z_l = workspace + self.z_l_init.to(device=workspace.device, dtype=workspace.dtype)
        z_h = workspace + self.z_h_init.to(device=workspace.device, dtype=workspace.dtype)
        trajectory = []
        steps = 0
        convergence_deltas = []
        converged = torch.zeros(b, device=workspace.device, dtype=torch.bool)
        loop_id = 0
        for outer in range(int(self.cfg.outer_steps)):
            previous_z_h = z_h
            for _h in range(int(self.cfg.h_cycles)):
                for _l in range(int(self.cfg.l_cycles)):
                    source_l = z_h + workspace
                    if self.step_conditioning is not None:
                        step_idx = min(int(loop_id), self.step_conditioning.num_embeddings - 1)
                        step_id = torch.tensor(step_idx, device=workspace.device, dtype=torch.long)
                        step = self.step_conditioning(step_id).view(1, 1, -1)
                        source_l = source_l + step * float(self.cfg.core_step_conditioning_scale)
                    z_l = self.norm_l(z_l + source_l)
                    z_l = self.fast_stack(z_l, attention_mask=attention_mask)
                    loop_id += 1
                    steps += 1
                source_h = z_l
                if self.step_conditioning is not None:
                    step_idx = min(int(loop_id), self.step_conditioning.num_embeddings - 1)
                    step_id = torch.tensor(step_idx, device=workspace.device, dtype=torch.long)
                    step = self.step_conditioning(step_id).view(1, 1, -1)
                    source_h = source_h + step * float(self.cfg.core_step_conditioning_scale)
                z_h = self.norm_h(z_h + source_h)
                z_h = self.slow_stack(z_h, attention_mask=attention_mask)
                loop_id += 1
                steps += 1
            trajectory.append(z_h)
            relative_delta = _relative_state_delta(z_h, previous_z_h)
            convergence_deltas.append(relative_delta)
            if (
                bool(getattr(self.cfg, "core_convergence_halt_enabled", False))
                and (outer + 1) >= int(getattr(self.cfg, "core_convergence_halt_min_outer", 1))
            ):
                converged = relative_delta <= float(
                    getattr(self.cfg, "core_convergence_halt_threshold", 1e-3)
                )
                if bool(converged.all().detach().cpu().item()):
                    break
        convergence_delta = (
            torch.stack(convergence_deltas, dim=1)
            if convergence_deltas
            else workspace.new_empty((b, 0), dtype=torch.float32)
        )
        info = {
            "steps": torch.full((b,), int(steps), device=workspace.device, dtype=torch.long),
            "outer_iterations": torch.full(
                (b,),
                int(len(trajectory)),
                device=workspace.device,
                dtype=torch.long,
            ),
            "converged": converged,
            "convergence_delta": convergence_delta,
        }
        return z_l, z_h, trajectory, info


class OuroWeightWrappedRecursiveCore(nn.Module):
    """Dual z_L/z_H recurrent core using pretrained Ouro layers as transitions."""

    def __init__(
        self,
        cfg: QTRMConfig,
        ouro_model: nn.Module,
        *,
        layer_indices: Optional[Sequence[int]] = None,
        shared_stack: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        text_model = _find_ouro_text_model(ouro_model)
        hidden_size = int(getattr(getattr(text_model, "config", None), "hidden_size", cfg.d_model))
        if hidden_size != int(cfg.d_model):
            raise ValueError(
                "Ouro hidden_size must match QTRM/Qwen hidden size for direct wrapping: "
                f"{hidden_size} != {cfg.d_model}"
            )
        layers = getattr(text_model, "layers")
        selected = _normalise_layer_indices(
            layer_indices,
            num_layers=len(layers),
            default=_default_ouro_core_layer_indices(text_model),
        )
        self.layer_indices = selected
        self.shared_stack = bool(shared_stack)
        self.z_l_init = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.z_h_init = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.norm_l = RMSNorm(cfg.d_model)
        self.norm_h = RMSNorm(cfg.d_model)
        self.step_conditioning = (
            nn.Embedding(max(1, int(cfg.core_step_conditioning_max_steps)), cfg.d_model)
            if bool(getattr(cfg, "core_step_conditioning_enabled", False))
            else None
        )
        if self.step_conditioning is not None:
            nn.init.normal_(self.step_conditioning.weight, mean=0.0, std=0.02)
        self.fast_stack = OuroLayerWrappedStack(text_model, selected)
        if self.shared_stack:
            self.slow_stack = self.fast_stack
        else:
            self.slow_stack = OuroLayerWrappedStack(text_model, selected)

    def forward(
        self,
        workspace: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        del kwargs
        b = workspace.shape[0]
        z_l = workspace + self.z_l_init.to(device=workspace.device, dtype=workspace.dtype)
        z_h = workspace + self.z_h_init.to(device=workspace.device, dtype=workspace.dtype)
        trajectory = []
        steps = 0
        convergence_deltas = []
        converged = torch.zeros(b, device=workspace.device, dtype=torch.bool)
        loop_id = 0
        for outer in range(int(self.cfg.outer_steps)):
            previous_z_h = z_h
            for _h in range(int(self.cfg.h_cycles)):
                for _l in range(int(self.cfg.l_cycles)):
                    source_l = z_h + workspace
                    if self.step_conditioning is not None:
                        step_idx = min(int(loop_id), self.step_conditioning.num_embeddings - 1)
                        step_id = torch.tensor(step_idx, device=workspace.device, dtype=torch.long)
                        step = self.step_conditioning(step_id).view(1, 1, -1)
                        source_l = source_l + step * float(self.cfg.core_step_conditioning_scale)
                    z_l = self.norm_l(z_l + source_l)
                    z_l = self.fast_stack(z_l, attention_mask=attention_mask)
                    loop_id += 1
                    steps += 1
                source_h = z_l
                if self.step_conditioning is not None:
                    step_idx = min(int(loop_id), self.step_conditioning.num_embeddings - 1)
                    step_id = torch.tensor(step_idx, device=workspace.device, dtype=torch.long)
                    step = self.step_conditioning(step_id).view(1, 1, -1)
                    source_h = source_h + step * float(self.cfg.core_step_conditioning_scale)
                z_h = self.norm_h(z_h + source_h)
                z_h = self.slow_stack(z_h, attention_mask=attention_mask)
                loop_id += 1
                steps += 1
            trajectory.append(z_h)
            relative_delta = _relative_state_delta(z_h, previous_z_h)
            convergence_deltas.append(relative_delta)
            if (
                bool(getattr(self.cfg, "core_convergence_halt_enabled", False))
                and (outer + 1) >= int(getattr(self.cfg, "core_convergence_halt_min_outer", 1))
            ):
                converged = relative_delta <= float(
                    getattr(self.cfg, "core_convergence_halt_threshold", 1e-3)
                )
                if bool(converged.all().detach().cpu().item()):
                    break
        convergence_delta = (
            torch.stack(convergence_deltas, dim=1)
            if convergence_deltas
            else workspace.new_empty((b, 0), dtype=torch.float32)
        )
        info = {
            "steps": torch.full((b,), int(steps), device=workspace.device, dtype=torch.long),
            "outer_iterations": torch.full(
                (b,),
                int(len(trajectory)),
                device=workspace.device,
                dtype=torch.long,
            ),
            "converged": converged,
            "convergence_delta": convergence_delta,
        }
        return z_l, z_h, trajectory, info


def build_qtrm_core_config_from_qwen(
    qwen_config: Any,
    *,
    max_seq_len: int,
    n_core_layers: int = 1,
    h_cycles: int = 1,
    l_cycles: int = 1,
    outer_steps: int = 1,
    dropout: float = 0.0,
    delta_backend: str = "fla_gated_delta",
    strict_backends: bool = True,
    core_causal: bool = True,
    core_convergence_halt_enabled: bool = False,
    core_convergence_halt_threshold: float = 1e-3,
    core_convergence_halt_min_outer: int = 1,
    core_step_conditioning_enabled: bool = False,
    core_step_conditioning_max_steps: int = 64,
    core_step_conditioning_scale: float = 1.0,
) -> QTRMConfig:
    """Build a small QTRM recurrent core that preserves Qwen text dimensions."""
    text_cfg = _text_config(qwen_config)
    hidden_size = _config_int(qwen_config, "hidden_size", 2048)
    num_attention_heads = _config_int(qwen_config, "num_attention_heads", 8)
    num_key_value_heads = _config_int(qwen_config, "num_key_value_heads", 2)
    intermediate_size = _config_int(qwen_config, "intermediate_size", hidden_size * 3)
    head_dim = _config_int(qwen_config, "linear_key_head_dim", hidden_size // num_attention_heads)
    num_value_heads = _config_int(qwen_config, "linear_num_value_heads", num_attention_heads)
    conv_size = _config_int(qwen_config, "linear_conv_kernel_dim", 4)
    full_attention_interval = _config_int(qwen_config, "full_attention_interval", 4)
    rope_theta = getattr(getattr(text_cfg, "rope_parameters", None), "rope_theta", None)
    if rope_theta is None and isinstance(getattr(text_cfg, "rope_parameters", None), dict):
        rope_theta = text_cfg.rope_parameters.get("rope_theta")
    rope_theta = float(rope_theta if rope_theta is not None else 10000000.0)
    return QTRMConfig(
        vocab_size=_config_int(qwen_config, "vocab_size", 248320),
        d_model=hidden_size,
        n_heads=num_attention_heads,
        n_kv_heads=num_key_value_heads,
        d_ff=intermediate_size,
        max_seq_len=int(max_seq_len),
        core_causal=bool(core_causal),
        n_core_layers=int(n_core_layers),
        attn_every=int(full_attention_interval),
        h_cycles=int(h_cycles),
        l_cycles=int(l_cycles),
        outer_steps=int(outer_steps),
        dropout=float(dropout),
        rope_theta=rope_theta,
        delta_backend=str(delta_backend),
        delta_head_dim=int(head_dim),
        delta_num_v_heads=int(num_value_heads),
        delta_conv_size=int(conv_size),
        delta_norm_eps=float(getattr(text_cfg, "rms_norm_eps", 1e-6)),
        attention_backend="sdpa",
        strict_backends=bool(strict_backends),
        use_stable_inject=True,
        core_context_enabled=False,
        core_halt_enabled=False,
        core_convergence_halt_enabled=bool(core_convergence_halt_enabled),
        core_convergence_halt_threshold=float(core_convergence_halt_threshold),
        core_convergence_halt_min_outer=int(core_convergence_halt_min_outer),
        core_step_conditioning_enabled=bool(core_step_conditioning_enabled),
        core_step_conditioning_max_steps=max(
            int(core_step_conditioning_max_steps),
            int(outer_steps) * int(h_cycles) * (int(l_cycles) + 1) + 1,
        ),
        core_step_conditioning_scale=float(core_step_conditioning_scale),
    )


class QwenBackboneQTRM(nn.Module):
    """Qwen-integrated QTRM-native model with a recurrent core in the LM path.

    This is not a donor sidecar. The Qwen model is the token embedding,
    decoder/backbone, and LM-head path. QTRM is inserted into that same causal
    logits path by perturbing Qwen's final hidden states before the tied LM
    head. ``force_core_off=True`` and ``core_gate_override=0`` are diagnostic
    ablations; canonical integrated runs should use ``mandatory_core=True`` so
    the normal forward path cannot learn or default to a closed core gate.
    """

    def __init__(
        self,
        qwen_model: nn.Module,
        *,
        model_id: str = "",
        core_config: Optional[QTRMConfig] = None,
        max_seq_len: int = 512,
        freeze_qwen: bool = True,
        core_gate_init: float = -8.0,
        residual_scale: float = 1.0,
        core_impl: str = "qtrm_block_stack",
        qwen_core_layer_indices: Optional[Sequence[int]] = None,
        ouro_model: Optional[nn.Module] = None,
        ouro_core_layer_indices: Optional[Sequence[int]] = None,
        core_adapter_dim: int = 0,
        core_delta_adapter_mode: str = "add",
        clone_qwen_core_layers: bool = False,
        trainable_qwen_core_clones: bool = True,
        mandatory_core: bool = False,
        core_insertion_mode: str = "final_residual",
        core_insert_after_layer: int = -1,
        core_residual_gate_mode: str = "constant",
        core_residual_gate_dim: int = 128,
        core_residual_gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.qwen = qwen_model
        self.model_id = str(model_id)
        self.core_impl = str(core_impl)
        self.mandatory_core = bool(mandatory_core)
        self.clone_qwen_core_layers = bool(clone_qwen_core_layers)
        self.core_insertion_mode = str(core_insertion_mode)
        if self.core_insertion_mode not in {"final_residual", "mid_layer_suffix"}:
            raise ValueError(f"unknown core_insertion_mode: {self.core_insertion_mode}")
        self.core_insert_after_layer = int(core_insert_after_layer)
        self.core_residual_gate_mode = str(core_residual_gate_mode)
        if self.core_residual_gate_mode not in {"constant", "token_mlp"}:
            raise ValueError(f"unknown core_residual_gate_mode: {self.core_residual_gate_mode}")
        hidden_size = _config_int(self.qwen.config, "hidden_size", 2048)
        self.core_cfg = core_config or build_qtrm_core_config_from_qwen(
            self.qwen.config,
            max_seq_len=max_seq_len,
        )
        if int(self.core_cfg.d_model) != int(hidden_size):
            raise ValueError(
                "QTRM core d_model must match Qwen hidden_size: "
                f"{self.core_cfg.d_model} != {hidden_size}"
            )
        self.core_in_norm = RMSNorm(hidden_size)
        if self.core_impl == "qtrm_block_stack":
            self.core = QTRMRecursiveCore(self.core_cfg)
        elif self.core_impl in {
            "qwen_layer_wrapped",
            "qwen_shared_layer_wrapped",
            "ouro_shared_qwen_layer",
        }:
            self.core = QwenLayerWrappedRecursiveCore(
                self.core_cfg,
                self.qwen,
                layer_indices=qwen_core_layer_indices,
                shared_stack=self.core_impl in {
                    "qwen_shared_layer_wrapped",
                    "ouro_shared_qwen_layer",
                },
                clone_layers=bool(clone_qwen_core_layers),
                trainable_clones=bool(trainable_qwen_core_clones),
            )
        elif self.core_impl == "ouro_weight_wrapped":
            if ouro_model is None:
                raise ValueError("core_impl=ouro_weight_wrapped requires ouro_model")
            self.ouro_model = ouro_model
            for parameter in self.ouro_model.parameters():
                parameter.requires_grad_(False)
            self.ouro_model.eval()
            self.core = OuroWeightWrappedRecursiveCore(
                self.core_cfg,
                self.ouro_model,
                layer_indices=ouro_core_layer_indices,
                shared_stack=True,
            )
        else:
            raise ValueError(f"unknown QwenBackboneQTRM core_impl: {self.core_impl}")
        self.core_out_norm = RMSNorm(hidden_size)
        self.core_suffix_stack: Optional[QwenLayerWrappedStack] = None
        if self.core_insertion_mode == "mid_layer_suffix":
            text_model = _find_qwen_text_model(self.qwen)
            layers = getattr(text_model, "layers", None)
            if layers is None or len(layers) == 0:
                raise ValueError("mid_layer_suffix insertion requires Qwen text layers")
            if self.core_insert_after_layer < 0:
                self.core_insert_after_layer = max(0, (len(layers) // 2) - 1)
            if self.core_insert_after_layer >= len(layers) - 1:
                raise ValueError(
                    "core_insert_after_layer must be before the final layer: "
                    f"{self.core_insert_after_layer} >= {len(layers) - 1}"
                )
            suffix_indices = tuple(range(self.core_insert_after_layer + 1, len(layers)))
            self.core_suffix_stack = QwenLayerWrappedStack(
                text_model,
                suffix_indices,
                force_causal=bool(getattr(self.core_cfg, "core_causal", True)),
                clone_layers=False,
                trainable_clones=False,
            )
        adapter_dim = int(core_adapter_dim)
        adapter_mode = str(core_delta_adapter_mode)
        if adapter_mode not in {"add", "adapter_only"}:
            raise ValueError(f"unknown core_delta_adapter_mode: {adapter_mode}")
        self.core_delta_adapter_mode = adapter_mode
        self.core_delta_adapter = (
            nn.Sequential(
                RMSNorm(hidden_size),
                nn.Linear(hidden_size, adapter_dim, bias=False),
                nn.GELU(),
                nn.Linear(adapter_dim, hidden_size, bias=False),
            )
            if adapter_dim > 0
            else None
        )
        if self.core_delta_adapter is not None:
            final = self.core_delta_adapter[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
        gate_dim = int(core_residual_gate_dim)
        self.core_residual_gate = (
            nn.Sequential(
                RMSNorm(hidden_size),
                nn.Linear(hidden_size, gate_dim, bias=False),
                nn.GELU(),
                nn.Linear(gate_dim, 1, bias=True),
            )
            if self.core_residual_gate_mode == "token_mlp" and gate_dim > 0
            else None
        )
        if self.core_residual_gate is not None:
            final_gate = self.core_residual_gate[-1]
            if isinstance(final_gate, nn.Linear):
                nn.init.zeros_(final_gate.weight)
                nn.init.constant_(final_gate.bias, float(core_residual_gate_init))
        self.core_gate_logit = nn.Parameter(torch.tensor(float(core_gate_init)))
        if self.mandatory_core:
            self.core_gate_logit.requires_grad_(False)
        self.residual_scale = float(residual_scale)
        if bool(freeze_qwen):
            for parameter in self.qwen.parameters():
                parameter.requires_grad_(False)
            self.qwen.eval()

    def freeze_qwen_parameters(self) -> None:
        for parameter in self.qwen.parameters():
            parameter.requires_grad_(False)
        self.qwen.eval()

    def set_qwen_partial_trainable(
        self,
        *,
        layer_indices: Optional[Sequence[int]] = None,
        train_embeddings: bool = False,
        train_lm_head: bool = False,
        train_final_norm: bool = False,
    ) -> dict[str, object]:
        """Freeze Qwen then unfreeze only explicit original-backbone pieces."""
        self.freeze_qwen_parameters()
        text_model = _find_qwen_text_model(self.qwen)
        layers = getattr(text_model, "layers", None)
        selected = _normalise_layer_indices(
            layer_indices,
            num_layers=len(layers),
            default=(),
        ) if layer_indices else ()
        for index in selected:
            for parameter in layers[int(index)].parameters():
                parameter.requires_grad_(True)
        if bool(train_embeddings):
            for parameter in self.qwen.get_input_embeddings().parameters():
                parameter.requires_grad_(True)
        if bool(train_lm_head):
            for parameter in self._lm_head().parameters():
                parameter.requires_grad_(True)
        final_norm_names = ("norm", "final_norm", "ln_f")
        final_norm_modules = []
        if bool(train_final_norm):
            for name in final_norm_names:
                module = getattr(text_model, name, None)
                if isinstance(module, nn.Module):
                    final_norm_modules.append(name)
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
        if any(parameter.requires_grad for parameter in self.qwen.parameters()):
            self.qwen.train()
        return {
            "layer_indices": list(selected),
            "train_embeddings": bool(train_embeddings),
            "train_lm_head": bool(train_lm_head),
            "train_final_norm": bool(train_final_norm),
            "final_norm_modules": final_norm_modules,
            "qwen_trainable_parameters": _count_parameters(self.qwen, trainable_only=True),
        }

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device | str] = None,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> "QwenBackboneQTRM":
        try:
            from transformers import AutoModelForImageTextToText
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers is required for QwenBackboneQTRM") from exc
        qwen = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        )
        if device is not None:
            qwen = qwen.to(device)
        core_config = kwargs.pop("core_config", None)
        if core_config is None:
            core_config = build_qtrm_core_config_from_qwen(
                qwen.config,
                max_seq_len=int(kwargs.pop("max_seq_len", 512)),
                n_core_layers=int(kwargs.pop("n_core_layers", 1)),
                h_cycles=int(kwargs.pop("h_cycles", 1)),
                l_cycles=int(kwargs.pop("l_cycles", 1)),
                outer_steps=int(kwargs.pop("outer_steps", 1)),
                dropout=float(kwargs.pop("dropout", 0.0)),
                delta_backend=str(kwargs.pop("delta_backend", "fla_gated_delta")),
                strict_backends=bool(kwargs.pop("strict_backends", True)),
                core_causal=bool(kwargs.pop("core_causal", True)),
                core_convergence_halt_enabled=bool(
                    kwargs.pop("core_convergence_halt_enabled", False)
                ),
                core_convergence_halt_threshold=float(
                    kwargs.pop("core_convergence_halt_threshold", 1e-3)
                ),
                core_convergence_halt_min_outer=int(
                    kwargs.pop("core_convergence_halt_min_outer", 1)
                ),
                core_step_conditioning_enabled=bool(
                    kwargs.pop("core_step_conditioning_enabled", False)
                ),
                core_step_conditioning_max_steps=int(
                    kwargs.pop("core_step_conditioning_max_steps", 64)
                ),
                core_step_conditioning_scale=float(
                    kwargs.pop("core_step_conditioning_scale", 1.0)
                ),
            )
        else:
            kwargs.pop("max_seq_len", None)
        return cls(qwen, model_id=model_id, core_config=core_config, **kwargs)

    def report(self) -> QwenBackboneQTRMReport:
        input_embeddings = self.qwen.get_input_embeddings()
        vocab_size = int(input_embeddings.num_embeddings)
        hidden_size = int(input_embeddings.embedding_dim)
        qwen_parameter_ids = {id(parameter) for parameter in self.qwen.parameters()}
        external_parameter_ids = set(qwen_parameter_ids)
        ouro_model = getattr(self, "ouro_model", None)
        if ouro_model is not None:
            external_parameter_ids.update(id(parameter) for parameter in ouro_model.parameters())
        return QwenBackboneQTRMReport(
            model_id=self.model_id,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            qwen_parameters=_count_parameters(self.qwen),
            qwen_trainable_parameters=_count_parameters(self.qwen, trainable_only=True),
            qtrm_parameters=(
                _count_parameters(self.core, exclude_ids=external_parameter_ids)
                + _count_parameters(self.core_in_norm)
                + _count_parameters(self.core_out_norm)
                + (
                    _count_parameters(self.core_delta_adapter)
                    if self.core_delta_adapter is not None
                    else 0
                )
                + (
                    _count_parameters(self.core_residual_gate)
                    if self.core_residual_gate is not None
                    else 0
                )
                + int(self.core_gate_logit.numel())
            ),
            qtrm_trainable_parameters=(
                _count_parameters(
                    self.core,
                    trainable_only=True,
                    exclude_ids=external_parameter_ids,
                )
                + _count_parameters(self.core_in_norm, trainable_only=True)
                + _count_parameters(self.core_out_norm, trainable_only=True)
                + (
                    _count_parameters(self.core_delta_adapter, trainable_only=True)
                    if self.core_delta_adapter is not None
                    else 0
                )
                + (
                    _count_parameters(self.core_residual_gate, trainable_only=True)
                    if self.core_residual_gate is not None
                    else 0
                )
                + int(self.core_gate_logit.numel() if self.core_gate_logit.requires_grad else 0)
            ),
            runtime_donor=False,
            integrated_qwen_backbone=True,
            standalone_graph=True,
            mandatory_core=bool(self.mandatory_core),
            core_impl=str(self.core_impl),
            normal_core_gate=float(self.normal_core_gate_value()),
            qwen_core_layers_cloned=bool(self.clone_qwen_core_layers),
            core_insertion_mode=str(self.core_insertion_mode),
            core_insert_after_layer=int(self.core_insert_after_layer),
            core_residual_gate_mode=str(self.core_residual_gate_mode),
        )

    def normal_core_gate_value(self) -> float:
        if self.mandatory_core:
            return 1.0
        return float(torch.sigmoid(self.core_gate_logit).detach().cpu())

    def _normal_core_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.mandatory_core:
            return hidden_states.new_tensor(1.0)
        return torch.sigmoid(self.core_gate_logit).to(dtype=hidden_states.dtype)

    def _lm_head(self) -> nn.Module:
        head = self.qwen.get_output_embeddings()
        if head is None:
            head = getattr(self.qwen, "lm_head", None)
        if head is None:
            raise ValueError("Qwen model has no output embedding / lm_head")
        return head

    def _text_final_norm(self) -> Optional[nn.Module]:
        text_model = _find_qwen_text_model(self.qwen)
        for name in ("norm", "final_norm", "ln_f"):
            module = getattr(text_model, name, None)
            if isinstance(module, nn.Module):
                return module
        return None

    def _apply_text_final_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm = self._text_final_norm()
        if norm is None:
            return hidden_states
        return norm(hidden_states)

    def _core_delta(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, core_hidden, _, core_info = self.core(
            self.core_in_norm(hidden_states),
            attention_mask=attention_mask,
        )
        core_delta = self.core_out_norm(core_hidden).to(
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        core_info["core_hidden"] = core_hidden
        core_info["core_delta"] = core_delta
        if self.core_delta_adapter is not None:
            raw_core_delta = core_delta
            adapter_delta = self.core_delta_adapter(raw_core_delta).to(
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            if self.core_delta_adapter_mode == "adapter_only":
                core_delta = adapter_delta
            else:
                core_delta = raw_core_delta + adapter_delta
        return core_delta, core_info

    def _adaptive_residual_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.core_residual_gate is None:
            return hidden_states.new_tensor(1.0)
        gate = torch.sigmoid(self.core_residual_gate(hidden_states.float())).to(
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return gate

    def _mid_layer_suffix_logits(
        self,
        outputs: Any,
        attention_mask: Optional[torch.Tensor],
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.core_suffix_stack is None:
            raise RuntimeError("mid_layer_suffix mode requires core_suffix_stack")
        hidden_states_tuple = getattr(outputs, "hidden_states", None)
        if hidden_states_tuple is None:
            raise RuntimeError("mid_layer_suffix mode requires output_hidden_states=True")
        hidden_index = int(self.core_insert_after_layer) + 1
        if hidden_index >= len(hidden_states_tuple):
            raise RuntimeError(
                "Qwen hidden_states does not contain the requested insertion point: "
                f"index={hidden_index}, available={len(hidden_states_tuple)}"
            )
        insert_hidden = hidden_states_tuple[hidden_index]
        core_delta, core_info = self._core_delta(insert_hidden, attention_mask)
        residual_gate = self._adaptive_residual_gate(insert_hidden)
        core_info["residual_gate_mean"] = residual_gate.detach().float().mean()
        core_info["residual_gate"] = residual_gate
        patched_hidden = insert_hidden + (
            gate.to(dtype=insert_hidden.dtype)
            * float(self.residual_scale)
            * residual_gate
            * core_delta
        )
        suffix_hidden = self.core_suffix_stack(
            patched_hidden.to(dtype=insert_hidden.dtype),
            attention_mask=attention_mask,
        )
        suffix_hidden = self._apply_text_final_norm(suffix_hidden)
        logits = self._lm_head()(suffix_hidden.to(dtype=insert_hidden.dtype))
        return logits, core_info

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        force_core_off: bool = False,
        core_gate_override: Optional[float] = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        if bool(force_core_off):
            return self.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
                **kwargs,
            )

        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.hidden_states[-1]
        gate = (
            hidden_states.new_tensor(float(core_gate_override))
            if core_gate_override is not None
            else self._normal_core_gate(hidden_states)
        )
        if float(gate.detach().cpu()) == 0.0:
            logits = outputs.logits
        else:
            if self.core_insertion_mode == "mid_layer_suffix":
                logits, core_info = self._mid_layer_suffix_logits(
                    outputs,
                    attention_mask,
                    gate,
                )
            else:
                core_delta, core_info = self._core_delta(hidden_states, attention_mask)
                residual_gate = self._adaptive_residual_gate(hidden_states)
                core_info["residual_gate_mean"] = residual_gate.detach().float().mean()
                core_info["residual_gate"] = residual_gate
                residual_hidden = hidden_states + (
                    gate.to(dtype=hidden_states.dtype)
                    * float(self.residual_scale)
                    * residual_gate
                    * core_delta
                )
                residual_hidden = residual_hidden.to(dtype=hidden_states.dtype)
                logits = self._lm_head()(residual_hidden)
        if "logits" not in locals():
            logits = outputs.logits
        if not return_dict:
            return (logits,)
        result = {
            key: value
            for key, value in outputs.items()
            if key not in {"logits", "hidden_states", "past_key_values"}
        }
        result["logits"] = logits
        result["qtrm_core_gate"] = gate.detach()
        if "core_info" in locals():
            result["qtrm_core_steps"] = core_info.get("steps")
            result["qtrm_core_outer_iterations"] = core_info.get("outer_iterations")
            result["qtrm_core_converged"] = core_info.get("converged")
            result["qtrm_core_convergence_delta"] = core_info.get("convergence_delta")
            result["qtrm_core_residual_gate_mean"] = core_info.get("residual_gate_mean")
            result["qtrm_core_residual_gate"] = core_info.get("residual_gate")
            result["qtrm_core_hidden"] = core_info.get("core_hidden")
            result["qtrm_core_delta"] = core_info.get("core_delta")
        return SimpleNamespace(**result)
