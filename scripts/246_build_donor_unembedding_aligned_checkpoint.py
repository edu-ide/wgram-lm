#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from wgram_lm.config import DonorConfig, load_config
from wgram_lm.wgram_model import QTRMMultimodalModel
from wgram_lm.qwen_donor import QwenDonorAdapter
from wgram_lm.training.train import load_initial_checkpoint


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a QTRM checkpoint whose LM head is initialized from the donor "
            "output embedding projected into the QTRM projector space."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-config", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--donor-model-id", default=None)
    parser.add_argument("--source", choices=["output", "input"], default="output")
    parser.add_argument("--method", choices=["pinv", "project"], default="pinv")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--rcond", type=float, default=1.0e-4)
    parser.add_argument("--match-existing-row-norm", action="store_true", default=True)
    parser.add_argument("--no-match-existing-row-norm", dest="match_existing_row_norm", action="store_false")
    parser.add_argument(
        "--resolve-base-chain",
        action="store_true",
        help=(
            "Materialize a full model state by recursively loading "
            "base_checkpoint metadata before replacing lm_head.weight. Use this "
            "for trainable-only delta checkpoints."
        ),
    )
    parser.add_argument(
        "--set-text-embed",
        action="store_true",
        help=(
            "Also replace text_embed.weight. Default keeps the trained input "
            "embedding and writes an untied lm_head probe config."
        ),
    )
    parser.add_argument(
        "--set-renderer-use-lm-head",
        action="store_true",
        help=(
            "When writing --output-config, enable the L4 vocab renderer path "
            "that lexicalizes through lm_head instead of its random low-rank "
            "vocab projection."
        ),
    )
    return parser


def donor_to_qtrm_mapping(
    projector_weight: torch.Tensor,
    *,
    method: str,
    rcond: float,
) -> torch.Tensor:
    projector = projector_weight.detach().float().cpu()
    if projector.ndim != 2:
        raise ValueError("projector_weight must be rank-2 [qtrm_dim, donor_dim]")
    if method == "pinv":
        return torch.linalg.pinv(projector, rcond=float(rcond))
    if method == "project":
        return projector.t().contiguous()
    raise ValueError(f"unknown method: {method}")


def project_unembedding_chunk(
    donor_weight_chunk: torch.Tensor,
    mapping: torch.Tensor,
) -> torch.Tensor:
    if donor_weight_chunk.ndim != 2:
        raise ValueError("donor_weight_chunk must be rank-2 [tokens, donor_dim]")
    if donor_weight_chunk.shape[1] != mapping.shape[0]:
        raise ValueError(
            "donor_weight_chunk donor_dim must match mapping first dimension: "
            f"{donor_weight_chunk.shape[1]} != {mapping.shape[0]}"
        )
    return donor_weight_chunk.float().cpu() @ mapping.float().cpu()


def match_mean_row_norm(
    weight: torch.Tensor,
    reference_weight: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    projected = weight.float()
    reference = reference_weight.float()
    source_norm = projected.norm(dim=1).mean().clamp_min(eps)
    target_norm = reference.norm(dim=1).mean().clamp_min(eps)
    scaled = projected * (target_norm / source_norm)
    return scaled, {
        "source_row_norm_mean": float(source_norm.item()),
        "target_row_norm_mean": float(target_norm.item()),
        "scale": float((target_norm / source_norm).item()),
    }


def _state_model_dict(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    model_state = state.get("model", state)
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint does not contain a model state dict")
    return model_state


def _donor_embedding_module(donor_model, source: str):
    if source == "output":
        module = donor_model.get_output_embeddings()
    elif source == "input":
        module = donor_model.get_input_embeddings()
    else:
        raise ValueError(f"unknown donor embedding source: {source}")
    if module is None or not hasattr(module, "weight"):
        raise ValueError(f"donor model does not expose {source} embeddings")
    return module


def _write_untied_probe_config(
    config_path: str,
    output_config: str,
    out_dir: str,
    *,
    set_renderer_use_lm_head: bool = False,
) -> None:
    with Path(config_path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    payload.setdefault("model", {})["tie_embeddings"] = False
    if bool(set_renderer_use_lm_head):
        payload.setdefault("model", {})[
            "core_role_value_state_vocab_renderer_use_lm_head"
        ] = True
    payload.setdefault("train", {})["out_dir"] = out_dir
    out = Path(output_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def build_aligned_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    donor_model_id = args.donor_model_id or cfg.donor.model_id
    if donor_model_id is None:
        raise ValueError("donor model id is required")

    if bool(args.resolve_base_chain):
        model = QTRMMultimodalModel(cfg.model)
        load_initial_checkpoint(model, args.checkpoint, map_location="cpu")
        model_state = model.state_dict()
        state = {
            "model": model_state,
            "format": "full_resolved_donor_unembedding_aligned",
            "source_checkpoint": str(args.checkpoint),
        }
    else:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model_state = _state_model_dict(state)
    required = ["projector.visual_proj.weight", "lm_head.weight", "text_embed.weight"]
    missing = [key for key in required if key not in model_state]
    if missing:
        raise ValueError(f"checkpoint missing required keys: {missing}")

    reference = model_state["lm_head.weight"].detach().cpu()
    projector_weight = model_state["projector.visual_proj.weight"].detach().cpu()
    mapping = donor_to_qtrm_mapping(
        projector_weight,
        method=str(args.method),
        rcond=float(args.rcond),
    )
    if mapping.shape != (int(cfg.model.visual_dim), int(cfg.model.d_model)):
        raise ValueError(
            f"unexpected mapping shape {tuple(mapping.shape)}, expected "
            f"({cfg.model.visual_dim}, {cfg.model.d_model})"
        )

    donor_cfg = DonorConfig(
        model_id=donor_model_id,
        load_in_4bit=bool(cfg.donor.load_in_4bit),
        freeze_donor=True,
        train_lora=False,
        trust_remote_code=bool(cfg.donor.trust_remote_code),
    )
    donor = QwenDonorAdapter(donor_cfg)
    donor_module = _donor_embedding_module(donor.model, str(args.source))
    donor_weight = donor_module.weight
    if donor_weight.shape[0] != reference.shape[0]:
        raise ValueError(
            f"donor vocab size {donor_weight.shape[0]} != QTRM vocab size {reference.shape[0]}"
        )
    if donor_weight.shape[1] != mapping.shape[0]:
        raise ValueError(
            f"donor hidden size {donor_weight.shape[1]} != mapping donor dim {mapping.shape[0]}"
        )

    chunk_size = max(1, int(args.chunk_size))
    aligned_chunks: list[torch.Tensor] = []
    norm_report: dict[str, float] = {}
    for start in range(0, int(donor_weight.shape[0]), chunk_size):
        end = min(start + chunk_size, int(donor_weight.shape[0]))
        chunk = donor_weight[start:end].detach().to(device="cpu", dtype=torch.float32)
        aligned_chunks.append(project_unembedding_chunk(chunk, mapping))
    aligned = torch.cat(aligned_chunks, dim=0)
    if bool(args.match_existing_row_norm):
        aligned, norm_report = match_mean_row_norm(aligned, reference)
    aligned = aligned.to(dtype=reference.dtype)

    model_state["lm_head.weight"] = aligned.contiguous()
    if bool(args.set_text_embed):
        model_state["text_embed.weight"] = aligned.clone().contiguous()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)

    output_config = args.output_config
    if output_config is not None and not bool(args.set_text_embed):
        _write_untied_probe_config(
            args.config,
            output_config,
            out_dir=str(out.parent),
            set_renderer_use_lm_head=bool(args.set_renderer_use_lm_head),
        )

    report = {
        "source_config": args.config,
        "source_checkpoint": args.checkpoint,
        "output_checkpoint": str(out),
        "output_config": output_config,
        "donor_model_id": donor_model_id,
        "donor_embedding_source": args.source,
        "method": args.method,
        "rcond": float(args.rcond),
        "chunk_size": chunk_size,
        "set_text_embed": bool(args.set_text_embed),
        "resolve_base_chain": bool(args.resolve_base_chain),
        "set_renderer_use_lm_head": bool(args.set_renderer_use_lm_head),
        "match_existing_row_norm": bool(args.match_existing_row_norm),
        "mapping_shape": list(mapping.shape),
        "aligned_weight_shape": list(aligned.shape),
        **norm_report,
    }
    report_path = args.report
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = build_aligned_checkpoint(build_arg_parser().parse_args())
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
