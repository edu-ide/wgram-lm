#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from wgram_lm.eval.memory_retrieval import canonical_answer_text, score_answer


UNKNOWN_COMPLETION = "Answer: UNKNOWN"


@dataclass(frozen=True)
class DecisionExample:
    record: dict[str, Any]
    features: list[float]
    label_block: int


class AnswerDecisionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _nested_float(mapping: dict[str, Any], path: list[str], default: float = 0.0) -> float:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return _safe_float(current, default)


def _text_non_ascii_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if ord(ch) > 127) / len(text)


def _truth_gate(record: dict[str, Any]) -> dict[str, Any]:
    meta = dict(record.get("answer_channel_meta") or {})
    return dict(meta.get("truth_gate") or {})


def _effective_no_answer_prob(meta: dict[str, Any]) -> float:
    raw = _safe_float(meta.get("no_answer_prob"), 0.0)
    if (
        str(meta.get("status", "other")) == "span"
        and bool(meta.get("no_answer_deferred_by_source_mask", False))
        and _safe_float(meta.get("source_token_mask_count"), 1.0) > 0.0
    ):
        return 0.0
    return raw


def feature_names(*, include_task_family: bool = False) -> list[str]:
    names = [
        "support_prob",
        "causal_prob",
        "refute_prob",
        "missing_prob",
        "truth_allow",
        "truth_block_reason_count",
        "no_answer_prob",
        "selected_score_scaled",
        "selected_span_len_scaled",
        "completion_token_count_scaled",
        "prompt_token_count_scaled",
        "candidate_is_unknown",
        "status_span",
        "status_no_answer",
        "status_truth_blocked",
        "status_other",
        "question_non_ascii_fraction",
        "completion_non_ascii_fraction",
        "first_step_logit_shift_scaled",
        "workspace_update_gate_mean",
        "workspace_update_gate_last_mean",
        "core_context_gate_mean",
        "core_context_gate_last_mean",
    ]
    if include_task_family:
        names.extend(["task_abstention", "task_conflict", "task_multi_hop", "task_other"])
    return names


def extract_features(
    record: dict[str, Any],
    *,
    include_task_family: bool = False,
) -> list[float]:
    meta = dict(record.get("answer_channel_meta") or {})
    truth = _truth_gate(record)
    status = str(meta.get("status", "other"))
    completion = str(record.get("completion", ""))
    latent_gates = dict(record.get("latent_gates") or {})
    shift = dict(record.get("first_step_logit_shift") or {})
    features = [
        _safe_float(truth.get("support_prob"), 1.0),
        _safe_float(truth.get("causal_prob"), 1.0),
        _safe_float(truth.get("refute_prob"), 0.0),
        _safe_float(truth.get("missing_prob"), 0.0),
        1.0 if bool(truth.get("allow", False)) else 0.0,
        min(1.0, len(list(truth.get("block_reasons") or [])) / 4.0),
        _effective_no_answer_prob(meta),
        _safe_float(meta.get("selected_score"), 0.0) / 32.0,
        min(1.0, len(list(meta.get("selected_token_ids") or [])) / 16.0),
        min(1.0, _safe_float(record.get("completion_token_count"), 0.0) / 32.0),
        min(1.0, _safe_float(record.get("prompt_token_count"), 0.0) / 512.0),
        1.0 if canonical_answer_text(completion) == "UNKNOWN" else 0.0,
        1.0 if status == "span" else 0.0,
        1.0 if status == "no_answer" else 0.0,
        1.0 if status == "truth_gate_blocked" else 0.0,
        1.0 if status not in {"span", "no_answer", "truth_gate_blocked"} else 0.0,
        _text_non_ascii_fraction(str(record.get("question", ""))),
        _text_non_ascii_fraction(completion),
        min(1.0, _safe_float(shift.get("max_abs_delta"), 0.0) / 2.0),
        _safe_float(latent_gates.get("workspace_update_gate_mean"), 0.0),
        _safe_float(latent_gates.get("workspace_update_gate_last_mean"), 0.0),
        _safe_float(latent_gates.get("core_context_gate_mean"), 0.0),
        _safe_float(latent_gates.get("core_context_gate_last_mean"), 0.0),
    ]
    if include_task_family:
        task_family = str(record.get("task_family", "other"))
        features.extend(
            [
                1.0 if task_family == "abstention" else 0.0,
                1.0 if task_family == "conflict" else 0.0,
                1.0 if task_family == "multi_hop" else 0.0,
                1.0 if task_family not in {"abstention", "conflict", "multi_hop"} else 0.0,
            ]
        )
    return features


def load_records(path: str | Path, *, mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if "summary" in record:
                continue
            if record.get("mode") == mode:
                rows.append(record)
    return rows


def label_block_improves(record: dict[str, Any]) -> int:
    aliases = record.get("answer_aliases") or []
    expected_unknown = bool(record.get("expected_unknown", False))
    baseline = score_answer(
        str(record.get("completion", "")),
        aliases,
        expected_unknown=expected_unknown,
    )
    blocked = score_answer(
        UNKNOWN_COMPLETION,
        aliases,
        expected_unknown=expected_unknown,
    )
    return int(bool(blocked["hit"]) and not bool(baseline["hit"]))


def build_examples(
    records: Iterable[dict[str, Any]],
    *,
    include_task_family: bool = False,
) -> list[DecisionExample]:
    return [
        DecisionExample(
            record=record,
            features=extract_features(record, include_task_family=include_task_family),
            label_block=label_block_improves(record),
        )
        for record in records
    ]


def stable_split(
    records: list[dict[str, Any]],
    *,
    calibration_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")

    def key(row: dict[str, Any]) -> str:
        raw = str(row.get("id", "")) + "\0" + str(row.get("question", ""))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    ordered = sorted(records, key=key)
    cut = max(1, min(len(ordered) - 1, int(round(len(ordered) * calibration_fraction))))
    return ordered[:cut], ordered[cut:]


def tensorize(examples: list[DecisionExample]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([example.features for example in examples], dtype=torch.float32)
    y = torch.tensor([example.label_block for example in examples], dtype=torch.float32)
    return x, y


def train_head(
    examples: list[DecisionExample],
    *,
    epochs: int,
    lr: float,
    hidden_dim: int,
    dropout: float,
    seed: int,
) -> AnswerDecisionHead:
    if not examples:
        raise ValueError("no training examples")
    torch.manual_seed(seed)
    x, y = tensorize(examples)
    model = AnswerDecisionHead(x.shape[1], hidden_dim=hidden_dim, dropout=dropout)
    positives = float(y.sum().item())
    negatives = float((1.0 - y).sum().item())
    pos_weight = torch.tensor([max(1.0, negatives / max(1.0, positives))], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def block_probabilities(model: AnswerDecisionHead, examples: list[DecisionExample]) -> list[float]:
    if not examples:
        return []
    x, _ = tensorize(examples)
    probs = torch.sigmoid(model(x)).detach().cpu().tolist()
    return [float(value) for value in probs]


def completion_after_decision(
    record: dict[str, Any],
    *,
    block: bool,
) -> str:
    completion = str(record.get("completion", ""))
    if block and canonical_answer_text(completion) != "UNKNOWN":
        return UNKNOWN_COMPLETION
    return completion


def evaluate_decisions(
    examples: list[DecisionExample],
    probabilities: list[float] | None = None,
    *,
    threshold: float = 1.0,
) -> dict[str, Any]:
    if probabilities is None:
        probabilities = [0.0 for _ in examples]
    hits = 0
    false_positive = 0
    blocked = 0
    block_improved = 0
    block_harmed = 0
    block_neutral = 0
    blocked_positive = 0
    for example, probability in zip(examples, probabilities):
        record = example.record
        baseline_score = score_answer(
            str(record.get("completion", "")),
            record.get("answer_aliases") or [],
            expected_unknown=bool(record.get("expected_unknown", False)),
        )
        block = bool(probability >= threshold)
        completion = completion_after_decision(record, block=block)
        score = score_answer(
            completion,
            record.get("answer_aliases") or [],
            expected_unknown=bool(record.get("expected_unknown", False)),
        )
        if bool(score["hit"]):
            hits += 1
        if (
            bool(record.get("expected_unknown", False))
            and canonical_answer_text(completion) != "UNKNOWN"
            and not bool(score["hit"])
        ):
            false_positive += 1
        if block and canonical_answer_text(str(record.get("completion", ""))) != "UNKNOWN":
            blocked += 1
            if bool(score["hit"]) and not bool(baseline_score["hit"]):
                block_improved += 1
            elif not bool(score["hit"]) and bool(baseline_score["hit"]):
                block_harmed += 1
                if not bool(record.get("expected_unknown", False)):
                    blocked_positive += 1
            else:
                block_neutral += 1
    count = len(examples)
    return {
        "count": count,
        "hits": hits,
        "accuracy": hits / count if count else 0.0,
        "false_positive": false_positive,
        "false_positive_rate": false_positive / count if count else 0.0,
        "blocked": blocked,
        "blocked_rate": blocked / count if count else 0.0,
        "block_improved": block_improved,
        "block_harmed": block_harmed,
        "block_neutral": block_neutral,
        "blocked_positive": blocked_positive,
        "blocked_positive_rate": blocked_positive / count if count else 0.0,
    }


def select_threshold(
    examples: list[DecisionExample],
    probabilities: list[float],
) -> tuple[float, dict[str, Any]]:
    best: tuple[tuple[float, int, int, float], float, dict[str, Any]] | None = None
    for idx in range(5, 96):
        threshold = idx / 100.0
        metrics = evaluate_decisions(examples, probabilities, threshold=threshold)
        rank = (
            float(metrics["accuracy"]),
            -int(metrics["false_positive"]),
            -int(metrics["block_harmed"]),
            -threshold,
        )
        if best is None or rank > best[0]:
            best = (rank, threshold, metrics)
    if best is None:
        raise ValueError("no threshold candidates")
    return best[1], best[2]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Learned Answer Decision Head",
        "",
        "## Verdict",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Setup",
        "",
        f"- train records: `{report['train_records_jsonl']}`",
        f"- eval records: `{report['eval_records_jsonl']}`",
        f"- selected threshold: `{report['selected_threshold']:.2f}`",
        f"- include task-family features: `{report['include_task_family_features']}`",
        "",
        "## Metrics",
        "",
        "| Split | Baseline Acc | Learned Acc | Baseline FP | Learned FP | Block Improved | Block Harmed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ["train", "eval"]:
        base = report[f"{split}_baseline"]
        learned = report[f"{split}_learned"]
        lines.append(
            f"| {split} | {base['accuracy']:.4f} | {learned['accuracy']:.4f} | "
            f"{base['false_positive']} | {learned['false_positive']} | "
            f"{learned['block_improved']} | {learned['block_harmed']} |"
        )
    lines.extend(["", "## Failed Checks", ""])
    failed = list(report.get("failed_checks") or [])
    if failed:
        lines.extend(f"- `{item}`" for item in failed)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a post-hoc learned decision head over recorded answer-channel "
            "telemetry. It is not yet wired into QTRM forward or trained end-to-end. "
            "It is a falsification gate for whether verifier telemetry contains "
            "enough signal to justify an integrated answer-decision module.",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a learned answer-decision head over answer-channel telemetry."
    )
    parser.add_argument("--train-records-jsonl", required=True)
    parser.add_argument("--eval-records-jsonl", default="")
    parser.add_argument("--mode", default="qtrm_residual_with_evidence")
    parser.add_argument("--out-pt", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--include-task-family-features", action="store_true")
    parser.add_argument("--min-eval-gain", type=float, default=0.01)
    parser.add_argument("--max-eval-block-harmed-rate", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    train_records = load_records(args.train_records_jsonl, mode=args.mode)
    if not train_records:
        raise SystemExit("no train records loaded")
    if args.eval_records_jsonl:
        eval_records = load_records(args.eval_records_jsonl, mode=args.mode)
        if not eval_records:
            raise SystemExit("no eval records loaded")
    else:
        train_records, eval_records = stable_split(
            train_records,
            calibration_fraction=float(args.calibration_fraction),
        )
    train_examples = build_examples(
        train_records,
        include_task_family=bool(args.include_task_family_features),
    )
    eval_examples = build_examples(
        eval_records,
        include_task_family=bool(args.include_task_family_features),
    )
    model = train_head(
        train_examples,
        epochs=int(args.epochs),
        lr=float(args.lr),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        seed=int(args.seed),
    )
    train_probs = block_probabilities(model, train_examples)
    selected_threshold, train_learned = select_threshold(train_examples, train_probs)
    eval_probs = block_probabilities(model, eval_examples)
    train_baseline = evaluate_decisions(train_examples)
    eval_baseline = evaluate_decisions(eval_examples)
    eval_learned = evaluate_decisions(
        eval_examples,
        eval_probs,
        threshold=selected_threshold,
    )
    eval_gain = float(eval_learned["accuracy"]) - float(eval_baseline["accuracy"])
    harmed_rate = float(eval_learned["block_harmed"]) / max(1, int(eval_learned["count"]))
    failed: list[str] = []
    if eval_gain < float(args.min_eval_gain):
        failed.append("eval_accuracy_gain_too_small")
    if int(eval_learned["false_positive"]) >= int(eval_baseline["false_positive"]):
        failed.append("false_positives_not_reduced")
    if harmed_rate > float(args.max_eval_block_harmed_rate):
        failed.append("blocks_too_many_correct_answers")
    report = {
        "train_records_jsonl": args.train_records_jsonl,
        "eval_records_jsonl": args.eval_records_jsonl or args.train_records_jsonl,
        "mode": args.mode,
        "train_count": len(train_examples),
        "eval_count": len(eval_examples),
        "feature_names": feature_names(
            include_task_family=bool(args.include_task_family_features)
        ),
        "include_task_family_features": bool(args.include_task_family_features),
        "selected_threshold": selected_threshold,
        "train_baseline": train_baseline,
        "train_learned": train_learned,
        "eval_baseline": eval_baseline,
        "eval_learned": eval_learned,
        "eval_gain": eval_gain,
        "eval_block_harmed_rate": harmed_rate,
        "failed_checks": failed,
        "status": "accepted" if not failed else "rejected",
    }
    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "feature_names": report["feature_names"],
            "selected_threshold": selected_threshold,
            "args": vars(args),
            "report": report,
        },
        out_pt,
    )
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_out:
        out_md = Path(args.markdown_out)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
