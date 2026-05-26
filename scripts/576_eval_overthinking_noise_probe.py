#!/usr/bin/env python3
"""Evaluate depth-induced answer drift on GD-lite style choice probes."""

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


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_gd_lite_eval_module() -> Any:
    path = ROOT / "scripts" / "567_eval_blt_generalization_dynamics_probe.py"
    spec = importlib.util.spec_from_file_location("gd_lite_eval_for_overthinking_noise", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mean(values: list[float]) -> float:
    return float(sum(values) / float(max(1, len(values))))


def _valid_depth_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("skipped_reason") is not None:
            continue
        by_id[str(row.get("id"))].append(row)
    return by_id


def negative_answers_for_row(row: dict[str, Any]) -> list[str]:
    target = str(row["intelligence_answer"])
    raw_negatives = row.get("negative_answers")
    if isinstance(raw_negatives, list):
        candidates = [str(item) for item in raw_negatives]
    else:
        candidates = [str(row["parrot_answer"])]
    out: list[str] = []
    seen: set[str] = set()
    for answer in candidates:
        if answer == target or answer in seen:
            continue
        seen.add(answer)
        out.append(answer)
    if not out:
        raise ValueError(f"row has no negative answers: {row.get('id')}")
    return out


def build_overthinking_noise_report(
    *,
    rows: list[dict[str, Any]],
    depths: list[int],
    checkpoint: str,
    probe_jsonl: str,
    max_mean_margin_drop: float = 0.02,
    max_row_margin_drop: float = 0.10,
    max_mean_margin_span: float = 0.50,
) -> dict[str, Any]:
    if not depths:
        raise ValueError("depths cannot be empty")
    sorted_depths = sorted({int(depth) for depth in depths})
    shallow_depth = int(sorted_depths[0])
    deepest_depth = int(sorted_depths[-1])

    gd_lite = load_gd_lite_eval_module()
    depth_summaries: list[dict[str, Any]] = []
    for depth in sorted_depths:
        depth_rows = [row for row in rows if int(row["think_steps"]) == int(depth)]
        summary = gd_lite.summarize_rows(depth_rows)
        depth_summaries.append({"think_steps": int(depth), **summary})

    complete_cases: list[dict[str, Any]] = []
    for case_id, case_rows in sorted(_valid_depth_rows(rows).items()):
        by_depth = {int(row["think_steps"]): row for row in case_rows}
        if any(depth not in by_depth for depth in sorted_depths):
            continue
        margins = [float(by_depth[depth]["normalized_margin"]) for depth in sorted_depths]
        corrects = [bool(by_depth[depth]["correct"]) for depth in sorted_depths]
        shallow_row = by_depth[shallow_depth]
        deepest_row = by_depth[deepest_depth]
        margin_drop = float(shallow_row["normalized_margin"]) - float(deepest_row["normalized_margin"])
        complete_cases.append(
            {
                "id": case_id,
                "task": str(deepest_row.get("task")),
                "margins": {str(depth): float(by_depth[depth]["normalized_margin"]) for depth in sorted_depths},
                "correct_by_depth": {str(depth): bool(by_depth[depth]["correct"]) for depth in sorted_depths},
                "margin_span": float(max(margins) - min(margins)),
                "deep_minus_shallow_margin": float(
                    float(deepest_row["normalized_margin"]) - float(shallow_row["normalized_margin"])
                ),
                "shallow_correct_deep_wrong": bool(shallow_row["correct"] and not deepest_row["correct"]),
                "shallow_wrong_deep_correct": bool((not shallow_row["correct"]) and deepest_row["correct"]),
                "correctness_flipped": bool(len(set(corrects)) > 1),
                "row_margin_regressed": bool(margin_drop > float(max_row_margin_drop)),
            }
        )

    if complete_cases:
        mean_margin_span = _mean([float(row["margin_span"]) for row in complete_cases])
        mean_deep_minus_shallow_margin = _mean(
            [float(row["deep_minus_shallow_margin"]) for row in complete_cases]
        )
    else:
        mean_margin_span = float("nan")
        mean_deep_minus_shallow_margin = float("nan")

    flip_to_wrong = [row for row in complete_cases if bool(row["shallow_correct_deep_wrong"])]
    flip_to_right = [row for row in complete_cases if bool(row["shallow_wrong_deep_correct"])]
    any_flip = [row for row in complete_cases if bool(row["correctness_flipped"])]
    margin_regressed = [row for row in complete_cases if bool(row["row_margin_regressed"])]
    wrong_at_all_depths = [
        row
        for row in complete_cases
        if not any(bool(value) for value in dict(row["correct_by_depth"]).values())
    ]

    deepest_summary = next(
        item for item in depth_summaries if int(item["think_steps"]) == int(deepest_depth)
    )
    passed_checks: list[str] = []
    failed_checks: list[str] = []
    if not flip_to_wrong:
        passed_checks.append("no_shallow_correct_to_deep_wrong_flips")
    else:
        failed_checks.append("shallow_correct_answers_lost_at_deeper_depth")
    if math.isfinite(mean_deep_minus_shallow_margin) and mean_deep_minus_shallow_margin >= -float(
        max_mean_margin_drop
    ):
        passed_checks.append("mean_deep_margin_not_degraded")
    else:
        failed_checks.append("mean_deep_margin_degraded")
    if math.isfinite(mean_margin_span) and mean_margin_span <= float(max_mean_margin_span):
        passed_checks.append("mean_margin_span_within_noise_budget")
    else:
        failed_checks.append("mean_margin_span_exceeds_noise_budget")

    stability_accepted = not failed_checks
    quality_accepted = bool(deepest_summary.get("accepted", False))
    return {
        "probe_type": "overthinking_noise_choice_probe",
        "checkpoint": str(checkpoint),
        "probe_jsonl": str(probe_jsonl),
        "depths": sorted_depths,
        "shallow_depth": shallow_depth,
        "deepest_depth": deepest_depth,
        "depth_summaries": depth_summaries,
        "complete_case_count": int(len(complete_cases)),
        "flip_to_wrong_count": int(len(flip_to_wrong)),
        "flip_to_right_count": int(len(flip_to_right)),
        "any_flip_count": int(len(any_flip)),
        "row_margin_regression_count": int(len(margin_regressed)),
        "wrong_at_all_depths_count": int(len(wrong_at_all_depths)),
        "mean_margin_span": float(mean_margin_span),
        "mean_deep_minus_shallow_margin": float(mean_deep_minus_shallow_margin),
        "stability_accepted": bool(stability_accepted),
        "quality_accepted": bool(quality_accepted),
        "accepted": bool(stability_accepted and quality_accepted),
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "plain_language_read": (
            "This gate separates not knowing the answer from overthinking noise. "
            "A source/knowledge failure can be wrong at every depth. Overthinking "
            "noise means extra recurrent steps erase a shallow correct answer or "
            "make the answer margin swing beyond the noise budget."
        ),
        "cases": complete_cases,
    }


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    gd_lite = load_gd_lite_eval_module()
    probe_rows = gd_lite.load_jsonl(Path(args.probe_jsonl))
    depth_probe = gd_lite.load_depth_probe_module()
    device = torch.device(str(args.device))
    trainer, _prefix, ckpt_args, loaded = depth_probe.load_checkpoint_model(
        checkpoint_path=Path(args.checkpoint),
        sampled_data=str(args.sampled_data),
        out_dir=str(Path(args.out).parent if str(args.out) else "local_eval/overthinking_noise_probe"),
        device=device,
        amp_dtype=str(args.amp_dtype),
    )
    model = loaded["model"]
    model.eval()
    amp_dtype = trainer.resolve_amp_dtype(str(args.amp_dtype))

    def make_amp_context() -> Any:
        if str(device.type) != "cuda":
            return nullcontext()
        return trainer.autocast_context(device, amp_dtype)

    tokenizer_info = dict(loaded.get("dataset_summary", {}).get("tokenizer_info") or {})
    byte_offset = int(args.byte_offset if int(args.byte_offset) >= 0 else tokenizer_info.get("byte_offset", 2))
    seq_len = int(args.seq_len or ckpt_args.seq_len)
    selected_probe_rows = probe_rows[: int(args.max_rows) if int(args.max_rows) > 0 else None]
    rows: list[dict[str, Any]] = []
    for depth in [int(value) for value in args.depths]:
        for row in selected_probe_rows:
            out_row: dict[str, Any] = {
                "id": row.get("id"),
                "task": row.get("task"),
                "source": row.get("source"),
                "think_steps": int(depth),
            }
            try:
                intelligence = gd_lite.choice_logprob(
                    model,
                    prompt=str(row["prompt"]),
                    answer=str(row["intelligence_answer"]),
                    seq_len=seq_len,
                    byte_offset=byte_offset,
                    device=device,
                    think_steps=int(depth),
                    amp_context=make_amp_context(),
                )
                negative_answers = negative_answers_for_row(row)
                negative_scores: list[dict[str, Any]] = []
                for answer in negative_answers:
                    score = gd_lite.choice_logprob(
                        model,
                        prompt=str(row["prompt"]),
                        answer=str(answer),
                        seq_len=seq_len,
                        byte_offset=byte_offset,
                        device=device,
                        think_steps=int(depth),
                        amp_context=make_amp_context(),
                    )
                    negative_scores.append({"answer": answer, **score})
                hardest_negative = max(negative_scores, key=lambda item: float(item["mean_logprob"]))
                margin = float(intelligence["mean_logprob"]) - float(hardest_negative["mean_logprob"])
                predicted_answer = (
                    str(row["intelligence_answer"])
                    if margin > 0.0
                    else str(hardest_negative["answer"])
                )
                out_row.update(
                    {
                        "intelligence_mean_logprob": float(intelligence["mean_logprob"]),
                        "parrot_mean_logprob": float(hardest_negative["mean_logprob"]),
                        "intelligence_tokens": int(intelligence["tokens"]),
                        "parrot_tokens": int(hardest_negative["tokens"]),
                        "target_answer": str(row["intelligence_answer"]),
                        "negative_answers": negative_answers,
                        "negative_mean_logprobs": [
                            float(item["mean_logprob"]) for item in negative_scores
                        ],
                        "predicted_answer": predicted_answer,
                        "normalized_margin": float(margin),
                        "correct": bool(margin > 0.0),
                        "skipped_reason": None,
                    }
                )
            except Exception as exc:
                if not bool(args.skip_bad_rows):
                    raise
                out_row.update(
                    {
                        "normalized_margin": float("nan"),
                        "correct": False,
                        "skipped_reason": str(exc),
                    }
                )
            rows.append(out_row)

    report = build_overthinking_noise_report(
        rows=rows,
        depths=[int(value) for value in args.depths],
        checkpoint=str(args.checkpoint),
        probe_jsonl=str(args.probe_jsonl),
        max_mean_margin_drop=float(args.max_mean_margin_drop),
        max_row_margin_drop=float(args.max_row_margin_drop),
        max_mean_margin_span=float(args.max_mean_margin_span),
    )
    report["seq_len"] = int(seq_len)
    report["byte_offset"] = int(byte_offset)
    report["rows"] = rows
    if str(args.out):
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe-jsonl", default="data/eval/generalization_dynamics_lite_probe.jsonl")
    parser.add_argument("--sampled-data", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--byte-offset", type=int, default=-1)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-mean-margin-drop", type=float, default=0.02)
    parser.add_argument("--max-row-margin-drop", type=float, default=0.10)
    parser.add_argument("--max-mean-margin-span", type=float, default=0.50)
    parser.add_argument("--skip-bad-rows", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_eval(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
