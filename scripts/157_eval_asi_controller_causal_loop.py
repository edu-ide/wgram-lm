#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from qtrm_mm.agentic.causal_gate import evaluate_causal_loop_gate
from qtrm_mm.agentic.cognitive_loop import Action
from qtrm_mm.config import load_config
from qtrm_mm.data.jsonl_dataset import JsonlTextVisionDataset, collate_jsonl
from qtrm_mm.qtrm_model import QTRMMultimodalModel
from qtrm_mm.qwen_donor import QwenDonorAdapter
from qtrm_mm.training.train import prepare_donor_batch


FORWARD_ABLATIONS: dict[str, dict[str, bool]] = {
    "qtrm_harness": {},
    "qtrm_latent_core_off": {"disable_core": True},
    "qtrm_workspace_off": {"disable_workspace": True},
    "qtrm_workspace_memory_off": {"disable_workspace_memory_context": True},
    "qtrm_core_to_text_off": {"disable_core_to_text": True},
    "qtrm_world_model_off": {},
    "qtrm_verifier_off": {},
    "qtrm_controller_signal_off": {"disable_controller_signal": True},
}


def _jsonl_line_count(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _action_name(action_id: int) -> str:
    try:
        return Action.from_id(int(action_id)).value
    except Exception:
        return f"ACTION_{int(action_id)}"


def summarize_action_predictions(
    preds: list[int],
    targets: list[int],
) -> dict[str, Any]:
    if len(preds) != len(targets):
        raise ValueError("preds and targets must have the same length")
    total = len(targets)
    correct = sum(int(pred == target) for pred, target in zip(preds, targets))
    per_target: dict[str, dict[str, int | float]] = {}
    confusion: dict[str, dict[str, int]] = {}
    for pred, target in zip(preds, targets):
        target_name = _action_name(target)
        pred_name = _action_name(pred)
        row = per_target.setdefault(
            target_name,
            {"total": 0, "correct": 0, "accuracy": 0.0},
        )
        row["total"] = int(row["total"]) + 1
        row["correct"] = int(row["correct"]) + int(pred == target)
        confusion.setdefault(target_name, {})
        confusion[target_name][pred_name] = confusion[target_name].get(pred_name, 0) + 1
    for row in per_target.values():
        row["accuracy"] = float(row["correct"]) / max(1, int(row["total"]))
    return {
        "samples": total,
        "accuracy": float(correct) / max(1, total),
        "per_target": per_target,
        "confusion": confusion,
    }


def build_asi_gate_metrics(
    mode_summaries: dict[str, dict[str, Any]],
    *,
    scripted_harness_accuracy: float = 1.0,
    donor_harness_accuracy: float | None = None,
) -> dict[str, float]:
    """Map controller-policy metrics onto the conservative ASI gate schema.

    The current Stage-1 controller is only an action policy. A scripted harness
    has the same retrieve/verify/answer action sequence by construction, so the
    ASI gate should reject unless QTRM later improves a real task score, not
    merely matches the scripted policy.
    """

    qtrm = float(mode_summaries["qtrm_harness"]["accuracy"])
    donor = (
        float(scripted_harness_accuracy)
        if donor_harness_accuracy is None
        else float(donor_harness_accuracy)
    )
    latent_core_off = float(mode_summaries["qtrm_latent_core_off"]["accuracy"])
    return {
        "scripted_harness": float(scripted_harness_accuracy),
        "donor_harness": donor,
        "qtrm_harness": qtrm,
        "qtrm_latent_core_off": latent_core_off,
        "qtrm_world_model_off": float(
            mode_summaries.get("qtrm_world_model_off", {"accuracy": qtrm})["accuracy"]
        ),
        "qtrm_verifier_off": float(
            mode_summaries.get("qtrm_verifier_off", {"accuracy": qtrm})["accuracy"]
        ),
    }


def _model_kwargs_for_mode(
    base_kwargs: dict[str, Any],
    *,
    mode_name: str,
) -> dict[str, Any]:
    kwargs = dict(base_kwargs)
    signal = kwargs.get("controller_signal")
    if mode_name not in {"qtrm_world_model_off", "qtrm_verifier_off"}:
        return kwargs
    if signal is not None:
        signal = signal.clone()
        if mode_name == "qtrm_world_model_off":
            signal[:, 0] = 0.0
        else:
            signal[:, 1] = 0.0
        kwargs["controller_signal"] = signal
    else:
        mask = torch.ones(2, dtype=torch.float32)
        if mode_name == "qtrm_world_model_off":
            mask[0] = 0.0
        else:
            mask[1] = 0.0
        kwargs["controller_signal_mask"] = mask
    return kwargs


def render_markdown(summary: dict[str, Any]) -> str:
    gate = summary["asi_gate"]
    lines = [
        "# ASI Controller Causal Loop Eval",
        "",
        "## Verdict",
        "",
        f"Status: `{gate.get('status', 'unknown')}`",
        "",
            "This is an action-policy gate, not an answer-quality or ASI proof.",
            "",
            "When `controller_signal` is present, it is an oracle scaffold for "
            "wiring future learned world-model and verifier outputs into the "
            "controller. It is not itself proof that those heads are learned.",
            "",
            "## Controller Metrics",
        "",
        "| Mode | Accuracy | Samples |",
        "| --- | ---: | ---: |",
    ]
    for name, mode in sorted(summary["controller_modes"].items()):
        lines.append(
            f"| {name} | {float(mode.get('accuracy', 0.0)):.4f} | "
            f"{int(mode.get('samples', 0))} |"
        )
    lines.extend(
        [
            "",
            "## ASI Gate Metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
    )
    for key, value in sorted(summary["asi_gate_metrics"].items()):
        lines.append(f"| {key} | {float(value):.4f} |")
    lines.extend(["", "## Failed Checks", ""])
    failed = list(gate.get("failed_checks", []))
    if failed:
        lines.extend(f"- `{item}`" for item in failed)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The QTRM controller can imitate the explicit retrieve-verify-answer "
            "trace policy. World-model-off masks controller signal dimension 0 "
            "and verifier-off masks dimension 1, either on an external oracle "
            "signal or on the learned core-derived signal. "
            "The ASI gate should still reject unless QTRM beats the scripted and "
            "donor harness baselines and all required ablation drops are present.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_controller_causal_loop(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    if device == "auto":
        device = "cpu"

    model = QTRMMultimodalModel(cfg.model).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    model.eval()

    donor = QwenDonorAdapter(cfg.donor) if args.use_donor else None
    dataset = JsonlTextVisionDataset(
        [args.data_jsonl],
        vocab_size=cfg.model.vocab_size,
        seq_len=cfg.train.seq_len,
        visual_dim=cfg.model.visual_dim,
        max_visual_tokens=cfg.model.max_visual_tokens,
        multimodal=False,
        shuffle_buffer=max(1, int(args.shuffle_buffer)),
        tokenizer_model_id=cfg.donor.model_id,
    )
    batch_size = int(args.batch_size or cfg.train.batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_jsonl)
    max_batches = int(args.max_batches)
    if max_batches <= 0:
        max_batches = max(1, math.ceil(_jsonl_line_count(args.data_jsonl) / max(1, batch_size)))

    preds_by_mode: dict[str, list[int]] = {name: [] for name in FORWARD_ABLATIONS}
    targets: list[int] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            model_kwargs: dict[str, Any] = {"attention_mask": batch.get("attention_mask")}
            if (
                "controller_signal" in batch
                and str(cfg.model.controller_signal_source).lower() == "external"
            ):
                model_kwargs["controller_signal"] = batch["controller_signal"]
            if donor is not None:
                model_kwargs.update(prepare_donor_batch(donor, batch, return_logits=False))
            with torch.amp.autocast(
                "cuda",
                enabled=(device == "cuda" and bool(cfg.train.use_amp)),
                dtype=torch.bfloat16,
            ):
                for mode_name, ablation_kwargs in FORWARD_ABLATIONS.items():
                    mode_kwargs = _model_kwargs_for_mode(model_kwargs, mode_name=mode_name)
                    outputs = model(
                        input_ids=batch["input_ids"],
                        **mode_kwargs,
                        **ablation_kwargs,
                    )
                    preds_by_mode[mode_name].extend(
                        outputs["action_logits"].argmax(dim=-1).detach().cpu().tolist()
                    )
            targets.extend(batch["action_target"].detach().cpu().tolist())

    mode_summaries = {
        name: summarize_action_predictions(preds, targets)
        for name, preds in preds_by_mode.items()
    }
    metrics = build_asi_gate_metrics(
        mode_summaries,
        scripted_harness_accuracy=args.scripted_harness_accuracy,
        donor_harness_accuracy=(
            None if args.donor_harness_accuracy < 0 else args.donor_harness_accuracy
        ),
    )
    gate = evaluate_causal_loop_gate(
        metrics,
        min_gain=float(args.min_gain),
        min_drop=float(args.min_drop),
    )
    return {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "data_jsonl": args.data_jsonl,
        "samples": len(targets),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "controller_modes": mode_summaries,
        "asi_gate_metrics": metrics,
        "asi_gate": gate,
        "notes": {
            "scope": "action_policy_only",
            "world_model_off_basis": "zeros or masks controller signal dimension 0",
            "verifier_off_basis": "zeros or masks controller signal dimension 1",
            "config_snapshot": asdict(cfg),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage-1 QTRM controller policy against conservative ASI gate metrics."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--use-donor", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--shuffle-buffer", type=int, default=1)
    parser.add_argument("--scripted-harness-accuracy", type=float, default=1.0)
    parser.add_argument(
        "--donor-harness-accuracy",
        type=float,
        default=-1.0,
        help="Negative means use scripted_harness_accuracy as the donor+harness baseline.",
    )
    parser.add_argument("--min-gain", type=float, default=0.02)
    parser.add_argument("--min-drop", type=float, default=0.03)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = evaluate_controller_causal_loop(args)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if args.markdown_out:
        md = Path(args.markdown_out)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(render_markdown(summary), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
