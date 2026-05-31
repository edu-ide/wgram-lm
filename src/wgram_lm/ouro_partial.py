from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch
from torch import nn


_SAFETENSOR_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


class _UnavailableOuroLayer(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.attention_type = "full_attention"

    def forward(self, *args, **kwargs):  # pragma: no cover - defensive path
        raise RuntimeError(f"Ouro layer {self.layer_idx} was not loaded")


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, object]]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    return int(header_size), header


def _read_tensor_from_safetensors(
    path: Path,
    *,
    header_size: int,
    metadata: dict[str, object],
) -> torch.Tensor:
    dtype_name = str(metadata["dtype"])
    dtype = _SAFETENSOR_DTYPES[dtype_name]
    shape = tuple(int(value) for value in metadata["shape"])
    start, end = (int(value) for value in metadata["data_offsets"])
    absolute_start = 8 + int(header_size) + start
    absolute_end = 8 + int(header_size) + end
    file_size = path.stat().st_size
    if absolute_end > file_size:
        raise ValueError(
            "safetensors tensor is not fully downloaded: "
            f"need byte {absolute_end}, file has {file_size}"
        )
    with path.open("rb") as handle:
        handle.seek(absolute_start)
        raw = handle.read(absolute_end - absolute_start)
    tensor = torch.frombuffer(raw, dtype=dtype).clone()
    return tensor.reshape(shape)


def _load_layer_state_from_partial_safetensors(
    safetensors_path: Path,
    *,
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    header_size, header = _read_safetensors_header(safetensors_path)
    prefix = f"model.layers.{int(layer_idx)}."
    state: dict[str, torch.Tensor] = {}
    for key, metadata in header.items():
        if not key.startswith(prefix) or not isinstance(metadata, dict):
            continue
        if "data_offsets" not in metadata:
            continue
        state[key[len(prefix) :]] = _read_tensor_from_safetensors(
            safetensors_path,
            header_size=header_size,
            metadata=metadata,
        )
    if not state:
        raise ValueError(f"no tensors found for Ouro layer {layer_idx}")
    return state


def build_partial_ouro_model_from_safetensors(
    model_dir: str | Path,
    *,
    layer_indices: Sequence[int],
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Build a lightweight Ouro container with only selected downloaded layers.

    The full Ouro checkpoint is a single safetensors file. While it is still
    downloading, early layers can already be complete in the prefix. This helper
    reads only the selected layer tensors manually and exposes the minimal
    ``.model.layers`` / ``.model.rotary_emb`` interface required by
    ``OuroWeightWrappedRecursiveCore``.
    """

    if not layer_indices:
        raise ValueError("at least one Ouro layer index is required")
    model_path = Path(model_dir)
    safetensors_path = model_path / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(str(safetensors_path))

    try:
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required to build partial Ouro") from exc

    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    config._attn_implementation = "eager"
    decoder_cls = get_class_from_dynamic_module("modeling_ouro.OuroDecoderLayer", str(model_path))
    rotary_cls = get_class_from_dynamic_module("modeling_ouro.OuroRotaryEmbedding", str(model_path))

    selected = tuple(int(index) for index in layer_indices)
    max_index = max(selected)
    layers = []
    for layer_idx in range(max_index + 1):
        if layer_idx in selected:
            layer = decoder_cls(config, layer_idx)
            layer_state = _load_layer_state_from_partial_safetensors(
                safetensors_path,
                layer_idx=layer_idx,
            )
            layer.load_state_dict(layer_state, strict=True)
        else:
            layer = _UnavailableOuroLayer(layer_idx)
        layers.append(layer)

    text_model = nn.Module()
    text_model.config = config
    text_model.layers = nn.ModuleList(layers)
    text_model.rotary_emb = rotary_cls(config=config)

    model = nn.Module()
    model.config = config
    model.model = text_model
    if dtype is not None or device is not None:
        model = model.to(device=device, dtype=dtype)
    model.partial_ouro_info = SimpleNamespace(
        model_dir=str(model_path),
        layer_indices=selected,
        safetensors_path=str(safetensors_path),
    )
    return model
