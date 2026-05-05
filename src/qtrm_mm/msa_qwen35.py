from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MSA_CONFIG: dict[str, Any] = {
    "top_k_docs": 8,
    "pooling_kernel_size": 64,
    "router_layer_idx": "all",
    "head_reduce_method": "mean",
    "query_reduce_method": "max",
    "chunk_reduce_method": "max",
    "decouple_pooling_mode": False,
    "decouple_router": True,
    "aux_loss_method": "INFONCE",
    "rewrite_position": True,
    "pad_free": False,
}


@dataclass(frozen=True)
class Qwen35FullMsaForkPlan:
    config: dict[str, Any]
    manifest: dict[str, Any]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def build_qwen35_full_msa_fork(
    source_config: dict[str, Any],
    *,
    msa_config: dict[str, Any] | None = None,
    layer_type_name: str = "sparse",
) -> Qwen35FullMsaForkPlan:
    """Build a config/manifest for a full-MSA Qwen3.5 text fork.

    This intentionally does not pretend the converted model can use all donor
    weights unchanged. Qwen3.5-2B is hybrid linear/full attention; replacing all
    token mixers with MSA changes the linear-attention layers and requires a
    healing/continual-pretraining stage.
    """

    cfg = deepcopy(source_config)
    text_cfg = _text_config(cfg)
    original_layer_types = list(text_cfg.get("layer_types", []))
    num_layers = int(text_cfg.get("num_hidden_layers", len(original_layer_types)))
    if not original_layer_types:
        original_layer_types = ["unknown"] * num_layers
    if len(original_layer_types) != num_layers:
        raise ValueError(
            "text_config.layer_types length must match text_config.num_hidden_layers "
            f"({len(original_layer_types)} != {num_layers})"
        )

    full_attention_layers = [
        idx for idx, kind in enumerate(original_layer_types) if kind == "full_attention"
    ]
    linear_attention_layers = [
        idx for idx, kind in enumerate(original_layer_types) if kind == "linear_attention"
    ]
    other_layers = [
        idx
        for idx, kind in enumerate(original_layer_types)
        if kind not in {"full_attention", "linear_attention"}
    ]

    merged_msa_config = deepcopy(DEFAULT_MSA_CONFIG)
    if msa_config:
        merged_msa_config.update(deepcopy(msa_config))

    text_cfg["qtrm_original_layer_types"] = original_layer_types
    text_cfg["layer_types"] = [layer_type_name] * num_layers
    text_cfg["msa_config"] = merged_msa_config
    text_cfg["qtrm_full_msa_fork"] = True
    text_cfg["qtrm_full_msa_layer_type_name"] = layer_type_name
    text_cfg["qtrm_full_msa_layer_type_semantics"] = (
        "QTRM uses the Hugging Face allowed 'sparse' layer type to mean "
        "Qwen3.5-native full Memory Sparse Attention."
    )
    text_cfg["qtrm_full_msa_source"] = {
        "source_model_type": source_config.get("model_type"),
        "source_text_model_type": text_cfg.get("model_type"),
        "reference_repo": "references/official/msa",
        "reference_arxiv": "https://arxiv.org/abs/2603.23516",
    }

    cfg["model_type"] = "qwen3_5_full_msa_fork"
    cfg["architectures"] = ["Qwen3_5FullMSAForConditionalGeneration"]

    manifest = {
        "architecture": "qwen3_5_full_msa_fork",
        "source_model_type": source_config.get("model_type"),
        "source_architectures": source_config.get("architectures", []),
        "text_model_type": text_cfg.get("model_type"),
        "num_hidden_layers": num_layers,
        "original_layer_types": original_layer_types,
        "target_layer_types": text_cfg["layer_types"],
        "full_attention_layers_reused_as_msa_seed": full_attention_layers,
        "linear_attention_layers_replaced_by_msa": linear_attention_layers,
        "other_layers": other_layers,
        "msa_config": merged_msa_config,
        "weight_reuse_policy": build_weight_reuse_policy(
            full_attention_layers=full_attention_layers,
            linear_attention_layers=linear_attention_layers,
            other_layers=other_layers,
        ),
        "training_policy": {
            "stage_0": "shape/load smoke only; do not claim quality",
            "stage_1": "freeze embeddings/mlp/norm/lm_head; train new MSA routers and replaced mixer weights on short memory data",
            "stage_2": "unfreeze full text backbone for donor-healing continual pretraining",
            "stage_3": "SFT on MemoryOS traces with doc_ids and evidence labels",
            "stage_4": "plug healed full-MSA donor into QTRM and run donor-only vs QTRM ablations",
        },
        "acceptance_gates": [
            "converted model loads and runs a tiny doc_ids forward pass",
            "perplexity/healing loss recovers against Qwen3.5 donor baseline",
            "MSA routing recall beats current external retrieval on held-out memory cases",
            "QTRM workspace_memory_off/evidence_bottleneck_off drops when MSA evidence is used",
            "generation does not regress into repeated-token collapse",
        ],
    }
    return Qwen35FullMsaForkPlan(config=cfg, manifest=manifest)


def build_weight_reuse_policy(
    *,
    full_attention_layers: list[int],
    linear_attention_layers: list[int],
    other_layers: list[int],
) -> dict[str, Any]:
    return {
        "reusable_without_shape_change": [
            "token embeddings",
            "lm_head / tied embeddings",
            "decoder MLP weights",
            "decoder input/post-attention RMSNorm weights",
            "final RMSNorm",
            "vision tower weights, if multimodal path is kept",
        ],
        "reusable_as_msa_seed": [
            {
                "layers": full_attention_layers,
                "weights": [
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "self_attn.q_norm",
                    "self_attn.k_norm",
                ],
                "note": (
                    "Qwen3.5 full-attention layers use gated q projection. A "
                    "Qwen3.5-native MSA attention should preserve that projection "
                    "contract instead of copying the plain Qwen3 MSA module blindly."
                ),
            }
        ],
        "must_reinitialize_or_heal": [
            {
                "layers": linear_attention_layers,
                "source": "Qwen3_5GatedDeltaNet",
                "target": "Qwen3.5-native MemorySparseAttention",
                "reason": (
                    "linear attention conv/recurrent weights do not map cleanly "
                    "to q/k/v/o sparse-attention weights or MSA router projections"
                ),
            },
            {
                "layers": "all MSA router layers",
                "weights": ["router_q_proj", "router_k_proj"],
                "reason": "new routing heads do not exist in the Qwen3.5 donor",
            },
        ],
        "unknown_or_manual_review": [
            {
                "layers": other_layers,
                "reason": "unrecognized original layer types",
            }
        ],
    }


def write_fork_artifacts(
    output_dir: str | Path,
    plan: Qwen35FullMsaForkPlan,
    *,
    source_config_path: str | Path | None = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", plan.config)
    write_json(out / "conversion_manifest.json", plan.manifest)
    (out / "README.md").write_text(
        build_readme(plan.manifest, source_config_path=source_config_path)
    )


def build_readme(
    manifest: dict[str, Any],
    *,
    source_config_path: str | Path | None = None,
) -> str:
    source = str(source_config_path) if source_config_path is not None else "unknown"
    replaced = manifest["linear_attention_layers_replaced_by_msa"]
    reused = manifest["full_attention_layers_reused_as_msa_seed"]
    return f"""# Qwen3.5-2B Full-MSA Fork Plan

Source config: `{source}`

This directory is a conversion plan, not a trained checkpoint. It describes a
Qwen3.5-2B donor fork where every text mixer layer is replaced by a
Qwen3.5-native Memory Sparse Attention layer.

## Layer Conversion

- Full-attention layers reusable as MSA seeds: `{reused}`
- Linear-attention layers replaced by MSA: `{replaced}`
- Target layer type: `{manifest["target_layer_types"][0]}` with
  `qtrm_full_msa_fork=true`

## Why Healing Is Required

Qwen3.5-2B uses hybrid GatedDeltaNet/full-attention layers. The linear-attention
weights do not map directly into sparse-attention q/k/v/o projections, and MSA
adds router projections plus document-wise cache/routing state. This fork must
therefore be healed with continual pretraining before it is treated as a donor.

## Next Commands

1. Implement/register `Qwen3_5FullMSAForConditionalGeneration`.
2. Load this config with random/new MSA layers and copied reusable weights.
3. Run a tiny doc_ids smoke forward.
4. Run donor-healing pretraining.
5. Plug the healed checkpoint into QTRM as the donor.
"""


def _text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    text_cfg = cfg.get("text_config")
    if not isinstance(text_cfg, dict):
        raise ValueError("Qwen3.5 config must contain a dict text_config")
    return text_cfg
