#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.utils.data import DataLoader

from qtrm_mm.config import load_config
from qtrm_mm.data.jsonl_dataset import JsonlTextVisionDataset, collate_jsonl
from qtrm_mm.losses import sequence_average_logprob
from qtrm_mm.qtrm_model import QTRMMultimodalModel
from qtrm_mm.qwen_donor import QwenDonorAdapter
from qtrm_mm.training.train import (
    build_core_world_model_actions,
    load_initial_checkpoint,
    prepare_donor_batch,
    strip_training_only_batch_keys,
)


DEFAULT_ABLATION_MODES = [
    "core_off",
    "workspace_off",
    "core_context_off",
    "core_to_text_off",
    "evidence_bottleneck_off",
]

_PRIVATE_FORWARD_KEYS = {
    "labels",
    "preference_rejected_input_ids",
    "preference_rejected_labels",
    "preference_rejected_attention_mask",
    "preference_rejected_text_states",
    "preference_rejected_donor_logits",
    "preference_sample_weight",
    "workspace_counterfactual_text_states",
    "workspace_counterfactual_attention_mask",
    "logical_support_target",
    "logical_refute_target",
    "logical_missing_target",
    "causal_evidence_target",
    "generation_verifier_repeat_target",
    "generation_verifier_stop_target",
    "generation_verifier_quality_target",
    "generation_verifier_sample_weight",
    "evidence_span_start_target",
    "evidence_span_end_target",
    "evidence_span_no_answer_target",
    "evidence_span_sample_weight",
    "answer_decision_target",
    "answer_decision_sample_weight",
    "action_targets",
    "action_sample_weight",
    "controller_signal_target",
    "controller_signal_sample_weight",
}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Probe whether canonical SSoT QTRM residual logits causally depend "
            "on latent core/workspace paths before running slow generation evals."
        )
    )
    ap.add_argument(
        "--config",
        default="configs/qwen35_2b_4090_canonical_ssot_coretotext_causal_s050.yaml",
    )
    ap.add_argument(
        "--checkpoint",
        default="runs/qwen35_2b_4090_canonical_ssot_coretotext_causal_s050/last.pt",
    )
    ap.add_argument(
        "--data-jsonl",
        nargs="+",
        default=["data/filtered/memory_reasoning_synth_traces.jsonl"],
    )
    ap.add_argument("--tokenizer-model-id", default=None)
    ap.add_argument("--multimodal", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--max-batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA autocast during probe forwards.",
    )
    ap.add_argument("--causal-margin-threshold", type=float, default=0.02)
    ap.add_argument("--ablation-mode", action="append", default=list(DEFAULT_ABLATION_MODES))
    ap.add_argument("--jsonl-out", default="runs/eval/canonical_causal_margin_probe.jsonl")
    ap.add_argument("--summary-json-out", default="runs/eval/canonical_causal_margin_probe_summary.json")
    return ap


def ablation_forward_kwargs(mode: str) -> dict[str, bool]:
    if mode == "core_off":
        return {"disable_core": True}
    if mode == "workspace_off":
        return {"disable_workspace": True}
    if mode == "workspace_memory_off":
        return {"disable_workspace_memory_context": True}
    if mode == "core_context_off":
        return {"disable_core_context": True}
    if mode == "core_to_text_off":
        return {"disable_core_to_text": True}
    if mode == "evidence_bottleneck_off":
        return {"disable_evidence_bottleneck": True}
    if mode == "span_reader_off":
        return {"disable_evidence_span_reader": True}
    raise ValueError(f"unknown ablation mode: {mode}")


def answer_logits_from_outputs(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    logits = outputs.get("qtrm_residual_logits")
    if logits is not None:
        return logits
    logits = outputs.get("qtrm_logits")
    if logits is not None:
        return logits
    return outputs["logits"]


def forward_kwargs_from_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in batch.items() if key not in _PRIVATE_FORWARD_KEYS}


def batch_logprobs(
    model: QTRMMultimodalModel,
    input_ids: torch.Tensor,
    model_kwargs: dict[str, Any],
    *,
    labels: torch.Tensor | None,
    forward_overrides: dict[str, bool] | None = None,
    use_amp: bool = True,
) -> torch.Tensor:
    kwargs = dict(model_kwargs)
    if forward_overrides:
        kwargs.update(forward_overrides)
    with torch.amp.autocast(
        "cuda",
        enabled=bool(use_amp and input_ids.device.type == "cuda"),
        dtype=torch.bfloat16,
    ):
        outputs = model(input_ids=input_ids, **kwargs)
    logits = answer_logits_from_outputs(outputs)
    offset = logits.shape[1] - input_ids.shape[1]
    return sequence_average_logprob(
        logits,
        input_ids,
        offset=offset,
        attention_mask=kwargs.get("attention_mask"),
        labels=labels,
    )


def _float_list(values: torch.Tensor) -> list[float]:
    return [float(value) for value in values.detach().float().cpu().tolist()]


def probe_batch(
    model: QTRMMultimodalModel,
    batch: dict[str, torch.Tensor],
    *,
    ablation_modes: list[str],
    use_amp: bool = True,
) -> list[dict[str, Any]]:
    model_kwargs = forward_kwargs_from_batch(batch)
    input_ids = model_kwargs.pop("input_ids")
    labels = batch.get("labels")

    full_logps = batch_logprobs(
        model,
        input_ids,
        model_kwargs,
        labels=labels,
        use_amp=use_amp,
    )
    ablation_logps_by_mode: dict[str, list[float]] = {}
    for mode in ablation_modes:
        ablation_logps = batch_logprobs(
            model,
            input_ids,
            model_kwargs,
            labels=labels,
            forward_overrides=ablation_forward_kwargs(mode),
            use_amp=use_amp,
        )
        ablation_logps_by_mode[mode] = _float_list(ablation_logps)

    full_values = _float_list(full_logps)
    records: list[dict[str, Any]] = []
    for idx, full_logp in enumerate(full_values):
        mode_logps = {
            mode: values[idx]
            for mode, values in ablation_logps_by_mode.items()
        }
        records.append(
            {
                "full_logp": full_logp,
                "ablation_logps": mode_logps,
                "margins": {
                    mode: full_logp - ablated_logp
                    for mode, ablated_logp in mode_logps.items()
                },
            }
        )
    return records


def summarize_probe_records(
    records: list[dict[str, Any]],
    *,
    causal_margin_threshold: float,
) -> dict[str, Any]:
    full_logps = [float(record["full_logp"]) for record in records]
    modes = sorted(
        {
            str(mode)
            for record in records
            for mode in dict(record.get("ablation_logps", {})).keys()
        }
    )
    mode_margins: dict[str, dict[str, Any]] = {}
    threshold = float(causal_margin_threshold)
    for mode in modes:
        margins = [
            float(record["full_logp"]) - float(record["ablation_logps"][mode])
            for record in records
            if mode in dict(record.get("ablation_logps", {}))
        ]
        if not margins:
            continue
        margin_mean = mean(margins)
        mode_margins[mode] = {
            "mean": margin_mean,
            "min": min(margins),
            "max": max(margins),
            "positive_rate": sum(1 for value in margins if value > 0.0) / len(margins),
            "causal": margin_mean >= threshold,
        }
    return {
        "num_records": len(records),
        "full_logp_mean": mean(full_logps) if full_logps else None,
        "causal_margin_threshold": threshold,
        "mode_margins": mode_margins,
        "causal_modes": [
            mode
            for mode, stats in mode_margins.items()
            if bool(stats.get("causal", False))
        ],
    }


def _select_device(cfg_device: str, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() and cfg_device in {"auto", "cuda"} else "cpu"


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    device = _select_device(cfg.train.device, args.device)

    model = QTRMMultimodalModel(cfg.model).to(device).eval()
    missing, unexpected = load_initial_checkpoint(model, args.checkpoint, map_location=device)
    if missing:
        print(f"[checkpoint] missing keys: {len(missing)}")
    if unexpected:
        print(f"[checkpoint] unexpected keys: {len(unexpected)}")
    print(f"[checkpoint] loaded {args.checkpoint}")

    if not cfg.donor.model_id:
        raise ValueError("canonical causal margin probe requires donor.model_id in config")
    donor = QwenDonorAdapter(cfg.donor)
    tokenizer_model_id = args.tokenizer_model_id or cfg.donor.model_id
    batch_size = int(args.batch_size or cfg.train.batch_size)
    ds = JsonlTextVisionDataset(
        files=[str(path) for path in args.data_jsonl],
        vocab_size=cfg.model.vocab_size,
        seq_len=cfg.train.seq_len,
        visual_dim=cfg.model.visual_dim,
        max_visual_tokens=min(cfg.model.max_visual_tokens, 256),
        multimodal=args.multimodal,
        tokenizer_model_id=tokenizer_model_id,
        workspace_evidence_injection=cfg.train.workspace_evidence_injection,
        workspace_evidence_injection_mode=cfg.train.workspace_evidence_injection_mode,
    )
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_jsonl)

    records: list[dict[str, Any]] = []
    iterator = iter(loader)
    with torch.no_grad():
        for batch_idx in range(max(0, int(args.max_batches))):
            batch = {key: value.to(device) for key, value in next(iterator).items()}
            model_batch = strip_training_only_batch_keys(batch)
            model_batch.update(
                prepare_donor_batch(
                    donor,
                    batch,
                    return_logits=bool(cfg.model.donor_logits_scale != 0.0),
                )
            )
            if cfg.model.core_world_model_enabled or cfg.train.loss_core_world_model_weight != 0.0:
                model_batch["core_world_model_actions"] = build_core_world_model_actions(
                    batch,
                    num_steps=cfg.model.outer_steps,
                    num_actions=cfg.model.num_actions,
                    device=device,
                )
            if (
                cfg.train.workspace_evidence_injection
                and cfg.train.workspace_evidence_injection_mode == "ssot"
            ):
                model_batch["evidence_span_reader_context"] = "input"

            batch_records = probe_batch(
                model,
                model_batch,
                ablation_modes=list(args.ablation_mode),
                use_amp=bool(cfg.train.use_amp and not args.no_amp),
            )
            for sample_idx, record in enumerate(batch_records):
                record["batch_idx"] = batch_idx
                record["sample_idx"] = sample_idx
            records.extend(batch_records)

    summary = summarize_probe_records(
        records,
        causal_margin_threshold=float(args.causal_margin_threshold),
    )
    _append_jsonl(args.jsonl_out, records)
    _write_json(args.summary_json_out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
