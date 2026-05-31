#!/usr/bin/env python3
"""EqR-style depth/residual probe for BLT-D PrefixLM checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wgram_lm.models.blt_prefixlm import BLTDByteLatentPrefixLM

IGNORE_LABEL_ID = -100


def load_trainer_module() -> Any:
    path = ROOT / "scripts" / "557_train_blt_d_prefixlm_dataio.py"
    spec = importlib.util.spec_from_file_location("blt_d_prefixlm_trainer_for_depth_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def namespace_from_checkpoint_args(trainer: Any, values: Any, *, sampled_data: str, out_dir: str) -> argparse.Namespace:
    args = trainer.build_arg_parser().parse_args(["--sampled-data", str(sampled_data), "--out-dir", str(out_dir)])
    if isinstance(values, argparse.Namespace):
        source = vars(values)
    elif isinstance(values, dict):
        source = values
    else:
        source = {}
    for key, value in source.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.sampled_data = str(sampled_data or getattr(args, "sampled_data", ""))
    args.out_dir = str(out_dir)
    return args


def _mean(values: list[float]) -> float:
    return float(sum(values) / float(max(1, len(values))))


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def build_depth_residual_report(rows: list[dict[str, Any]], *, checkpoint: str) -> dict[str, Any]:
    by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_depth[int(row["think_steps"])].append(row)
    if not by_depth:
        raise ValueError("no depth rows to summarize")

    summaries: list[dict[str, Any]] = []
    for depth, depth_rows in sorted(by_depth.items()):
        target_tokens = sum(max(0, int(row.get("target_tokens", 0))) for row in depth_rows)
        loss_num = 0.0
        loss_weight = 0
        nonfinite_loss_rows = 0
        nonfinite_loss_target_tokens = 0
        residual_items: list[tuple[float, int]] = []
        nonfinite_residual_rows = 0
        nonfinite_residual_target_tokens = 0
        for row in depth_rows:
            weight = max(0, int(row.get("target_tokens", 0)))
            loss_value = _finite_float(row.get("loss"))
            if loss_value is None:
                nonfinite_loss_rows += 1
                nonfinite_loss_target_tokens += weight
            else:
                loss_num += loss_value * weight
                loss_weight += weight
            residual_value = _finite_float(row.get("fixed_point_residual"))
            if residual_value is None:
                nonfinite_residual_rows += 1
                nonfinite_residual_target_tokens += weight
            else:
                residual_items.append((residual_value, weight))
        loss = loss_num / float(loss_weight) if loss_weight > 0 else None
        residual_weight = sum(max(0, weight) for _, weight in residual_items)
        mean_residual = (
            sum(value * max(0, weight) for value, weight in residual_items) / float(residual_weight)
            if residual_weight > 0
            else None
        )
        summaries.append(
            {
                "think_steps": int(depth),
                "loss": float(loss) if loss is not None else None,
                "target_tokens": int(target_tokens),
                "finite_loss_target_tokens": int(loss_weight),
                "nonfinite_loss_rows": int(nonfinite_loss_rows),
                "nonfinite_loss_target_tokens": int(nonfinite_loss_target_tokens),
                "batch_count": int(len(depth_rows)),
                "finite_residual_target_tokens": int(residual_weight),
                "nonfinite_residual_rows": int(nonfinite_residual_rows),
                "nonfinite_residual_target_tokens": int(nonfinite_residual_target_tokens),
                "mean_fixed_point_residual": float(mean_residual) if mean_residual is not None else None,
            }
        )

    shallow = summaries[0]
    deepest = summaries[-1]
    finite_loss_summaries = [item for item in summaries if item.get("loss") is not None]
    best_loss_summary = (
        min(finite_loss_summaries, key=lambda item: float(item["loss"]))
        if finite_loss_summaries
        else summaries[0]
    )
    passed_checks: list[str] = []
    failed_checks: list[str] = []
    shallow_loss = _finite_float(shallow.get("loss"))
    deepest_loss = _finite_float(deepest.get("loss"))
    if deepest_loss is not None and shallow_loss is not None and deepest_loss < shallow_loss:
        passed_checks.append("deepest_loss_beats_shallowest")
    else:
        failed_checks.append("no_depth_loss_gain")
    shallow_residual = shallow.get("mean_fixed_point_residual")
    deepest_residual = deepest.get("mean_fixed_point_residual")
    if (
        _finite_float(shallow_residual) is not None
        and _finite_float(deepest_residual) is not None
        and float(deepest_residual) < float(shallow_residual)
    ):
        passed_checks.append("deepest_residual_below_shallowest")
    else:
        failed_checks.append("no_depth_residual_gain")
    if best_loss_summary.get("loss") is not None and int(best_loss_summary["think_steps"]) == int(deepest["think_steps"]):
        passed_checks.append("best_loss_at_deepest_depth")
    else:
        failed_checks.append("best_loss_not_at_deepest_depth")
    nonfinite_loss_rows = sum(int(item.get("nonfinite_loss_rows", 0)) for item in summaries)
    nonfinite_residual_rows = sum(int(item.get("nonfinite_residual_rows", 0)) for item in summaries)
    if nonfinite_loss_rows > 0 or nonfinite_residual_rows > 0:
        failed_checks.append("nonfinite_depth_rows_present")

    return {
        "probe_type": "blt_depth_residual_probe",
        "checkpoint": str(checkpoint),
        "depth_summaries": summaries,
        "best_loss": (
            float(best_loss_summary["loss"])
            if best_loss_summary.get("loss") is not None
            else None
        ),
        "best_loss_depth": int(best_loss_summary["think_steps"]),
        "nonfinite_loss_rows": int(nonfinite_loss_rows),
        "nonfinite_residual_rows": int(nonfinite_residual_rows),
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "accepted": not failed_checks,
        "plain_language_read": (
            "If extra latent thinking is useful, deeper think_steps should lower "
            "heldout loss while the hidden state moves less between the last two "
            "depths. Non-finite rows are tracked as a stability failure instead "
            "of being allowed to erase the finite depth trend."
        ),
    }


def _fixed_point_residual(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    think_steps: int,
) -> float:
    labels = torch.full_like(input_ids, int(IGNORE_LABEL_ID))
    grouped_ids, grouped_mask, _, _, _, _ = model.pack_patches(input_ids, attention_mask, labels)
    _, _, _, patch_embeddings = model._grouped_patch_embeddings(grouped_ids, grouped_mask)
    current = model._global_hidden(patch_embeddings, think_steps=int(think_steps))
    previous_steps = max(0, int(think_steps) - 1)
    previous = model._global_hidden(patch_embeddings, think_steps=previous_steps)
    diff = (current - previous).float()
    denom = previous.float().norm(dim=-1).mean().clamp_min(1e-6)
    return float((diff.norm(dim=-1).mean() / denom).detach().cpu().item())


def is_allowed_missing_checkpoint_key(key: str) -> bool:
    """Allow newly added diagnostic-only modules when loading older checkpoints."""

    allowed_missing_prefixes = (
        "hierarchical_chunk_proj.",
        "hierarchical_chunk_gate.",
        "answer_anchor_head.",
        "answer_workspace_selector.",
        "hnet_latent_bridge.",
        "hnet_causal_speaker.",
        "imta_",
        "own_latent_predictor.",
    )
    allowed_missing_exact = {
        "answer_readback_gate_logit",
        "hnet_byte_residual_gate_logit",
        "hnet_latent_residual_gate_logit",
    }
    return str(key) in allowed_missing_exact or any(
        str(key).startswith(prefix) for prefix in allowed_missing_prefixes
    )


def load_checkpoint_model(
    *,
    checkpoint_path: Path,
    sampled_data: str,
    out_dir: str,
    device: torch.device,
    amp_dtype: str,
) -> tuple[Any, Any, argparse.Namespace, dict[str, Any]]:
    trainer = load_trainer_module()
    prefix = trainer.load_prefixlm_module()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"checkpoint does not contain model_state_dict: {checkpoint_path}")
    args = namespace_from_checkpoint_args(
        trainer,
        payload.get("args"),
        sampled_data=str(sampled_data),
        out_dir=str(out_dir),
    )
    args.device = str(device)
    args.amp_dtype = str(amp_dtype)
    model_summary = payload.get("model", {}) if isinstance(payload.get("model"), dict) else {}
    dataset_summary = payload.get("dataset", {}) if isinstance(payload.get("dataset"), dict) else {}
    vocab_size = int(model_summary.get("vocab_size") or dataset_summary.get("model_vocab_size") or 512)
    global_seq_len = int(model_summary.get("global_seq_len") or math.ceil(int(args.seq_len) / float(args.patch_size)))
    global_args = trainer.build_global_args(args, prefix, global_seq_len=global_seq_len)
    global_core = prefix.build_model(global_args, vocab_size=int(vocab_size))
    model = BLTDByteLatentPrefixLM(
        global_core=global_core,
        vocab_size=int(vocab_size),
        d_model=int(args.d_model),
        patch_size=int(args.patch_size),
        mask_token_id=int(args.mask_token_id) if int(args.mask_token_id) >= 0 else int(vocab_size) - 1,
        local_layers=int(args.local_layers),
        local_heads=int(args.local_heads),
        dropout=float(args.dropout),
        clean_boundary_current_latent=not bool(args.no_clean_boundary_current_latent),
        decoder_latent_mode=str(args.decoder_latent_mode),
        patch_boundary_mode=str(args.patch_boundary_mode),
        dynamic_min_patch_size=int(args.dynamic_min_patch_size),
        dynamic_soft_patch_size=int(args.dynamic_soft_patch_size),
        hbf_boundary_threshold=float(args.hbf_boundary_threshold),
        nitp_enabled=float(args.nitp_loss_weight) > 0.0,
        nitp_hidden_dim=int(args.nitp_hidden_dim),
        answer_readback_mode=str(args.answer_readback_mode),
        answer_readback_gate_init=float(args.answer_readback_gate_init),
        answer_readback_temperature=float(args.answer_readback_temperature),
        hnet_one_body_byte_gate_init=float(args.hnet_one_body_byte_gate_init),
        hnet_one_body_latent_gate_init=float(args.hnet_one_body_latent_gate_init),
        imta_trajectories=int(args.imta_trajectories),
        imta_noise_std=float(args.imta_noise_std),
        imta_selector_temperature=float(args.imta_selector_temperature),
        imta_adapter_gate_init=float(args.imta_adapter_gate_init),
        own_latent_prediction_enabled=bool(args.own_latent_prediction_enabled),
    ).to(device)
    adapted_state_dict, key_adaptations = trainer.adapt_resume_state_dict_for_current_model(
        payload["model_state_dict"],
        model.state_dict(),
    )
    incompatible = model.load_state_dict(adapted_state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    disallowed_missing = [key for key in missing if not is_allowed_missing_checkpoint_key(str(key))]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "checkpoint/model mismatch: "
            f"missing={disallowed_missing} unexpected={unexpected}"
        )
    model.eval()
    return trainer, prefix, args, {
        "model": model,
        "model_summary": model_summary,
        "dataset_summary": dataset_summary,
        "checkpoint_key_adaptations": key_adaptations,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(str(args.device))
    trainer, prefix, ckpt_args, loaded = load_checkpoint_model(
        checkpoint_path=Path(args.checkpoint),
        sampled_data=str(args.sampled_data),
        out_dir=str(Path(args.out).parent if str(args.out) else "local_eval/depth_residual_probe"),
        device=device,
        amp_dtype=str(args.amp_dtype),
    )
    model = loaded["model"]
    amp_dtype = trainer.resolve_amp_dtype(str(args.amp_dtype))
    dataset = prefix.DataIOSampledPrefixLMDataset(
        args.sampled_data,
        seq_len=int(args.seq_len or ckpt_args.seq_len),
        epoch=int(args.epoch),
        target_only=True,
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        drop_overlength=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=prefix.collate_prefixlm_rows,
        drop_last=False,
    )
    rows: list[dict[str, Any]] = []
    context = trainer.autocast_context(device, amp_dtype) if str(device.type) == "cuda" else nullcontext()
    for depth in [int(value) for value in args.depths]:
        for batch_index, batch in enumerate(loader):
            if bool(args.trim_batch_to_max_length):
                batch = prefix.trim_prefixlm_batch_to_max_valid_length(batch)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = int((labels != IGNORE_LABEL_ID).sum().detach().cpu().item())
            if targets <= 0:
                continue
            with torch.no_grad(), context:
                loss, metrics = model.forward_losses(
                    input_ids,
                    labels,
                    attention_mask,
                    think_steps=int(depth),
                    diffusion_weight=0.0,
                    diffusion_mask_prob=0.0,
                )
                residual = _fixed_point_residual(
                    model,
                    input_ids,
                    attention_mask,
                    think_steps=int(depth),
                )
            loss_value = _finite_float(metrics.get("clean_loss", float(loss.detach().cpu().item())))
            residual_value = _finite_float(residual)
            rows.append(
                {
                    "case_id": f"batch-{batch_index:05d}",
                    "think_steps": int(depth),
                    "loss": float(loss_value) if loss_value is not None else None,
                    "target_tokens": int(targets),
                    "fixed_point_residual": (
                        float(residual_value) if residual_value is not None else None
                    ),
                    "latent_len": int(metrics.get("latent_len", 0)),
                }
            )
            if int(args.max_batches) > 0 and batch_index + 1 >= int(args.max_batches):
                break
    report = build_depth_residual_report(rows, checkpoint=str(args.checkpoint))
    report["rows"] = rows
    report["sampled_data"] = str(args.sampled_data)
    report["depths"] = [int(value) for value in args.depths]
    report["checkpoint_key_adaptations"] = loaded.get("checkpoint_key_adaptations", {})
    if str(args.out):
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows_path = out_path.with_suffix(".jsonl")
        rows_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sampled-data", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--trim-batch-to-max-length", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_probe(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
