from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5MLP,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)


@dataclass
class Qwen35FullMsaTextOutput:
    last_hidden_state: torch.Tensor
    selected_doc_ids_by_layer: list[list[list[int]]]


@dataclass
class Qwen35FullMsaCausalLMOutput:
    logits: torch.Tensor
    selected_doc_ids_by_layer: list[list[list[int]]]
    loss: Optional[torch.Tensor] = None


class Qwen35FullMsaAttention(Qwen3_5Attention):
    """Qwen3.5-native MSA attention prototype.

    This preserves Qwen3.5's gated q projection and q/k RMSNorm contract, then
    routes over document ids using MSA-style chunk-pooled sparse document
    selection. It is a small training/smoke implementation, not the full
    Memory Parallel runtime from the MSA repository.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        msa_config = _as_dict(getattr(config, "msa_config", None))
        self.top_k_docs = int(msa_config.get("top_k_docs", 8))
        self.pooling_kernel_size = int(msa_config.get("pooling_kernel_size", 64))
        self.head_reduce_method = str(msa_config.get("head_reduce_method", "mean"))
        self.query_reduce_method = str(msa_config.get("query_reduce_method", "max"))
        self.chunk_reduce_method = str(msa_config.get("chunk_reduce_method", "max"))
        self.decouple_router = bool(msa_config.get("decouple_router", True))
        self.aux_loss_method = str(msa_config.get("aux_loss_method", "INFONCE"))
        self.last_selected_doc_ids: list[list[int]] = []
        if self.decouple_router:
            self.router_q_proj = nn.Linear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim,
                bias=False,
            )
            self.router_k_proj = nn.Linear(
                config.hidden_size,
                config.num_key_value_heads * self.head_dim,
                bias=False,
            )
        else:
            self.router_q_proj = None
            self.router_k_proj = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.LongTensor] = None,
        **_: Any,
    ) -> tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        valid_mask = _valid_token_mask(attention_mask, hidden_states)
        if doc_ids is None:
            doc_ids = hidden_states.new_zeros(input_shape, dtype=torch.long)
        doc_ids = doc_ids.to(device=hidden_states.device)
        if doc_ids.shape != valid_mask.shape:
            raise ValueError(
                f"doc_ids must have shape {tuple(valid_mask.shape)}, got {tuple(doc_ids.shape)}"
            )

        if self.decouple_router:
            if self.router_q_proj is None or self.router_k_proj is None:
                raise RuntimeError("decouple_router=True but router projections are missing")
            routing_q = self.router_q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            routing_k = self.router_k_proj(hidden_states).view(
                *input_shape,
                -1,
                self.head_dim,
            ).transpose(1, 2)
            routing_k = repeat_kv(routing_k, self.num_key_value_groups)
        else:
            routing_q = query_states
            routing_k = repeat_kv(key_states, self.num_key_value_groups)

        if "INFONCE" in self.aux_loss_method.upper():
            routing_q = F.normalize(routing_q, p=2, dim=-1)
            routing_k = F.normalize(routing_k, p=2, dim=-1)

        selected_key_mask, selected_doc_ids = self._select_docs(
            doc_ids=doc_ids,
            valid_mask=valid_mask,
            routing_q=routing_q,
            routing_k=routing_k,
        )
        self.last_selected_doc_ids = selected_doc_ids

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        seq_len = hidden_states.shape[1]
        causal = torch.ones(
            (seq_len, seq_len),
            device=hidden_states.device,
            dtype=torch.bool,
        ).tril()
        keep = (
            causal[None, None, :, :]
            & selected_key_mask[:, None, None, :]
            & valid_mask[:, None, :, None]
        )
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=keep,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), None

    def _select_docs(
        self,
        *,
        doc_ids: torch.LongTensor,
        valid_mask: torch.Tensor,
        routing_q: torch.Tensor,
        routing_k: torch.Tensor,
    ) -> tuple[torch.Tensor, list[list[int]]]:
        batch, seq_len = doc_ids.shape
        selected = (doc_ids == 0) & valid_mask
        selected_doc_ids: list[list[int]] = []
        for batch_idx in range(batch):
            sample_doc_ids = doc_ids[batch_idx]
            query_positions = torch.nonzero(
                (sample_doc_ids == 0) & valid_mask[batch_idx],
                as_tuple=False,
            ).flatten()
            doc_values = torch.unique(sample_doc_ids[(sample_doc_ids > 0) & valid_mask[batch_idx]])
            if query_positions.numel() == 0 or doc_values.numel() == 0:
                selected_doc_ids.append([])
                continue
            query = routing_q[batch_idx, :, query_positions]
            scores = []
            doc_id_ints = []
            for doc_id in doc_values.tolist():
                positions = torch.nonzero(
                    (sample_doc_ids == int(doc_id)) & valid_mask[batch_idx],
                    as_tuple=False,
                ).flatten()
                if positions.numel() == 0:
                    continue
                chunks = positions.split(self.pooling_kernel_size)
                chunk_states = []
                for chunk in chunks:
                    chunk_states.append(routing_k[batch_idx, :, chunk].mean(dim=1))
                pooled_k = torch.stack(chunk_states, dim=0)
                score = self._score_doc(query, pooled_k)
                scores.append(score)
                doc_id_ints.append(int(doc_id))
            if not scores:
                selected_doc_ids.append([])
                continue
            score_tensor = torch.stack(scores)
            top_k = min(self.top_k_docs, score_tensor.numel())
            top_indices = torch.topk(score_tensor, k=top_k).indices.tolist()
            chosen = [doc_id_ints[idx] for idx in top_indices]
            selected_doc_ids.append(chosen)
            chosen_tensor = torch.tensor(chosen, device=doc_ids.device, dtype=doc_ids.dtype)
            selected[batch_idx] |= torch.isin(sample_doc_ids, chosen_tensor)
        # Always keep each valid token as a key for itself to avoid empty rows.
        selected |= valid_mask & (doc_ids <= 0)
        return selected, selected_doc_ids

    def _score_doc(self, query: torch.Tensor, pooled_k: torch.Tensor) -> torch.Tensor:
        # query: [H, Q, D], pooled_k: [C, H, D]
        scores = torch.einsum("hqd,chd->hqc", query, pooled_k) * self.scaling
        if self.head_reduce_method == "max":
            scores = scores.max(dim=0).values
        elif self.head_reduce_method == "mean":
            scores = scores.mean(dim=0)
        else:
            raise ValueError(f"Unsupported head_reduce_method: {self.head_reduce_method}")

        if self.query_reduce_method == "max":
            scores = scores.max(dim=0).values
        elif self.query_reduce_method == "mean":
            scores = scores.mean(dim=0)
        elif self.query_reduce_method == "last":
            scores = scores[-1]
        else:
            raise ValueError(f"Unsupported query_reduce_method: {self.query_reduce_method}")

        if self.chunk_reduce_method == "max":
            return scores.max()
        if self.chunk_reduce_method == "mean":
            return scores.mean()
        raise ValueError(f"Unsupported chunk_reduce_method: {self.chunk_reduce_method}")


class Qwen35FullMsaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen35FullMsaAttention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            doc_ids=doc_ids,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen35FullMsaTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen35FullMsaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        doc_ids: Optional[torch.LongTensor] = None,
    ) -> Qwen35FullMsaTextOutput:
        hidden_states = self.embed_tokens(input_ids)
        batch, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if doc_ids is None:
            doc_ids = torch.zeros_like(input_ids)
        if position_ids is None:
            base_position_ids = torch.arange(seq_len, device=input_ids.device).view(1, -1).expand(batch, -1)
            position_ids = base_position_ids[None, ...].expand(4, batch, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            rotary_position_ids = position_ids[1:]
        else:
            rotary_position_ids = position_ids
        position_embeddings = self.rotary_emb(hidden_states, rotary_position_ids)

        selected_by_layer = []
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                doc_ids=doc_ids,
            )
            selected_by_layer.append(layer.self_attn.last_selected_doc_ids)
        hidden_states = self.norm(hidden_states)
        return Qwen35FullMsaTextOutput(
            last_hidden_state=hidden_states,
            selected_doc_ids_by_layer=selected_by_layer,
        )


class Qwen35FullMsaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen35FullMsaTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        doc_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Qwen35FullMsaCausalLMOutput:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            doc_ids=doc_ids,
        )
        logits = self.lm_head(out.last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return Qwen35FullMsaCausalLMOutput(
            logits=logits,
            selected_doc_ids_by_layer=out.selected_doc_ids_by_layer,
            loss=loss,
        )


def save_qwen35_full_msa_checkpoint(
    model: Qwen35FullMsaForCausalLM,
    output_dir: str | Path,
    *,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Save a QTRM custom full-MSA checkpoint.

    This model is intentionally not registered as a Hugging Face
    `PreTrainedModel` yet, so save a small explicit artifact set that can be
    loaded without relying on AutoModel registration.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_dict = _config_to_dict(model.config)
    (out / "config.json").write_text(
        json.dumps(config_dict, ensure_ascii=False, indent=2) + "\n"
    )
    torch.save(model.state_dict(), out / "model_state.pt")
    checkpoint_meta = {
        "architecture": "Qwen35FullMsaForCausalLM",
        "state_file": "model_state.pt",
        "config_file": "config.json",
        "metadata": metadata or {},
    }
    (out / "qtrm_full_msa_checkpoint.json").write_text(
        json.dumps(checkpoint_meta, ensure_ascii=False, indent=2) + "\n"
    )


def load_qwen35_full_msa_checkpoint(
    checkpoint_dir: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Qwen35FullMsaForCausalLM:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    ckpt = Path(checkpoint_dir)
    config_dict = json.loads((ckpt / "config.json").read_text())
    config = Qwen3_5TextConfig.from_dict(config_dict)
    _restore_custom_config_attrs(config, config_dict)
    model = Qwen35FullMsaForCausalLM(config)
    state = _torch_load_state_dict(ckpt / "model_state.pt", map_location=map_location)
    model.load_state_dict(state, strict=True)
    return model


def copy_qwen35_text_weights_into_full_msa(source_model: nn.Module, target_model: Qwen35FullMsaForCausalLM) -> dict:
    """Copy reusable Qwen3.5 text weights into a full-MSA fork.

    Linear-attention layers are intentionally not mapped into MSA attention.
    They remain randomly initialized and are reported as healing targets.
    """

    source_text = _source_text_model(source_model)
    source_lm_head = getattr(source_model, "lm_head", None)
    target_text = target_model.model
    report = {
        "copied_groups": {
            "embeddings": 0,
            "lm_head": 0,
            "mlp": 0,
            "layer_norms": 0,
            "full_attention_seed": 0,
        },
        "reinitialized_msa_layers": [],
        "seeded_msa_layers": [],
        "skipped": [],
    }

    _copy_module(source_text.embed_tokens, target_text.embed_tokens)
    report["copied_groups"]["embeddings"] += 1
    if source_lm_head is not None:
        _copy_module(source_lm_head, target_model.lm_head)
        report["copied_groups"]["lm_head"] += 1

    original_layer_types = list(
        getattr(
            target_model.config,
            "qtrm_original_layer_types",
            getattr(source_text.config, "layer_types", []),
        )
    )
    if len(original_layer_types) != len(target_text.layers):
        raise ValueError(
            "original_layer_types length must match target layers "
            f"({len(original_layer_types)} != {len(target_text.layers)})"
        )

    for layer_idx, (source_layer, target_layer) in enumerate(zip(source_text.layers, target_text.layers)):
        _copy_module(source_layer.mlp, target_layer.mlp)
        report["copied_groups"]["mlp"] += 1
        _copy_module(source_layer.input_layernorm, target_layer.input_layernorm)
        _copy_module(source_layer.post_attention_layernorm, target_layer.post_attention_layernorm)
        report["copied_groups"]["layer_norms"] += 2

        layer_type = original_layer_types[layer_idx]
        if layer_type == "full_attention" and hasattr(source_layer, "self_attn"):
            _copy_full_attention_seed(source_layer.self_attn, target_layer.self_attn)
            report["copied_groups"]["full_attention_seed"] += 1
            report["seeded_msa_layers"].append(layer_idx)
        elif layer_type == "linear_attention":
            report["reinitialized_msa_layers"].append(layer_idx)
        else:
            report["skipped"].append({"layer": layer_idx, "layer_type": layer_type})

    _copy_module(source_text.norm, target_text.norm)
    report["copied_groups"]["layer_norms"] += 1
    return report


def _source_text_model(source_model: nn.Module) -> nn.Module:
    model = getattr(source_model, "model", source_model)
    if hasattr(model, "language_model"):
        return model.language_model
    return model


def _copy_full_attention_seed(source_attn: nn.Module, target_attn: Qwen35FullMsaAttention) -> None:
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
        _copy_module(getattr(source_attn, name), getattr(target_attn, name))


def _copy_module(source: nn.Module, target: nn.Module) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    for key, value in source_state.items():
        if key not in target_state:
            raise ValueError(f"target module missing key {key}")
        if target_state[key].shape != value.shape:
            raise ValueError(
                f"shape mismatch for {key}: source {tuple(value.shape)} "
                f"target {tuple(target_state[key].shape)}"
            )
    target.load_state_dict(source_state, strict=True)


def _config_to_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        data = dict(config.to_dict())
    else:
        data = dict(vars(config))
    for name in (
        "qtrm_original_layer_types",
        "qtrm_full_msa_fork",
        "qtrm_full_msa_layer_type_name",
        "qtrm_full_msa_layer_type_semantics",
        "qtrm_full_msa_source",
        "msa_config",
    ):
        if hasattr(config, name):
            data[name] = getattr(config, name)
    return data


def _restore_custom_config_attrs(config: Any, data: dict[str, Any]) -> None:
    for name in (
        "qtrm_original_layer_types",
        "qtrm_full_msa_fork",
        "qtrm_full_msa_layer_type_name",
        "qtrm_full_msa_layer_type_semantics",
        "qtrm_full_msa_source",
        "msa_config",
    ):
        if name in data:
            setattr(config, name, data[name])


def _torch_load_state_dict(path: Path, *, map_location: str | torch.device) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _valid_token_mask(attention_mask: Optional[torch.Tensor], hidden_states: torch.Tensor) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool)
    if attention_mask.ndim == 2:
        return attention_mask.to(device=hidden_states.device, dtype=torch.bool)
    if attention_mask.ndim == 4:
        return attention_mask[:, 0, -1, :].to(device=hidden_states.device, dtype=torch.bool)
    raise ValueError(f"Unsupported attention_mask ndim: {attention_mask.ndim}")


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return dict(value)
