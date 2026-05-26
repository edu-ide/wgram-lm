#!/usr/bin/env python3
"""Conservative PrefixLM token-efficiency comparison.

This compares two HRM-Text/Data-IO PrefixLM training reports. It answers one
question only:

    Did the candidate reach the baseline's final loss within baseline_tokens /
    factor observed tokens?

Anything outside the same dataset contract remains unproven.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONTRACT_KEYS = (
    "contract",
    "vocab_size",
    "seq_len",
    "target_only",
    "max_seq_len",
    "total_length",
)

EVAL_DATASET_KEYS = (
    "contract",
    "vocab_size",
    "seq_len",
    "target_only",
    "max_seq_len",
    "total_length",
    "epoch",
    "rows",
    "drop_overlength",
    "eval_protocol",
    "eval_fingerprint",
    "eval_batch_size",
    "eval_max_batches",
)


def load_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    report["_report_path"] = str(report_path)
    return report


def dataset_contract(report: dict[str, Any]) -> dict[str, Any]:
    dataset = report.get("dataset")
    if not isinstance(dataset, dict):
        dataset = {}
    return {key: dataset.get(key) for key in CONTRACT_KEYS}


def eval_dataset_contract(report: dict[str, Any]) -> dict[str, Any]:
    dataset = report.get("eval_dataset")
    if not isinstance(dataset, dict):
        dataset = {}
    return {key: dataset.get(key) for key in EVAL_DATASET_KEYS}


def eval_coverage(report: dict[str, Any]) -> dict[str, Any]:
    history = eval_loss_history(report)
    if not history:
        return {}
    last = history[-1]
    return {
        "eval_tokens": last.get("eval_tokens"),
        "eval_target_tokens": last.get("eval_target_tokens"),
    }


def contract_mismatches(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    baseline_contract = dataset_contract(baseline)
    candidate_contract = dataset_contract(candidate)
    mismatches = {
        key: (baseline_contract.get(key), candidate_contract.get(key))
        for key in CONTRACT_KEYS
        if baseline_contract.get(key) != candidate_contract.get(key)
    }
    if metric_source(baseline) != metric_source(candidate):
        mismatches["metric_source"] = (metric_source(baseline), metric_source(candidate))
    if metric_source(baseline) == metric_source(candidate) == "eval_loss_history":
        baseline_eval_contract = eval_dataset_contract(baseline)
        candidate_eval_contract = eval_dataset_contract(candidate)
        for key in EVAL_DATASET_KEYS:
            if baseline_eval_contract.get(key) != candidate_eval_contract.get(key):
                mismatches[f"eval_dataset.{key}"] = (
                    baseline_eval_contract.get(key),
                    candidate_eval_contract.get(key),
                )
        baseline_coverage = eval_coverage(baseline)
        candidate_coverage = eval_coverage(candidate)
        for key in ("eval_tokens", "eval_target_tokens"):
            if baseline_coverage.get(key) != candidate_coverage.get(key):
                mismatches[key] = (baseline_coverage.get(key), candidate_coverage.get(key))
    return mismatches


def loss_history(report: dict[str, Any]) -> list[dict[str, Any]]:
    history = report.get("loss_history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"report has no loss_history: {report.get('_report_path')}")
    return [row for row in history if isinstance(row, dict)]


def eval_loss_history(report: dict[str, Any]) -> list[dict[str, Any]]:
    history = report.get("eval_loss_history")
    if not isinstance(history, list):
        return []
    return [row for row in history if isinstance(row, dict)]


def metric_source(report: dict[str, Any]) -> str:
    return "eval_loss_history" if eval_loss_history(report) else "loss_history"


def final_loss(report: dict[str, Any]) -> float:
    eval_history = eval_loss_history(report)
    if eval_history:
        value = report.get("final_eval_loss")
        if value is not None:
            return float(value)
        return float(eval_history[-1]["eval_loss"])
    value = report.get("final_logged_loss")
    if value is not None:
        return float(value)
    return float(loss_history(report)[-1]["loss"])


def final_tokens(report: dict[str, Any]) -> int:
    train = report.get("train")
    if isinstance(train, dict) and train.get("tokens_seen") is not None:
        return int(train["tokens_seen"])
    return int(loss_history(report)[-1]["tokens_seen"])


def final_target_tokens(report: dict[str, Any]) -> int:
    train = report.get("train")
    if isinstance(train, dict) and train.get("target_tokens_seen") is not None:
        return int(train["target_tokens_seen"])
    history = eval_loss_history(report) or loss_history(report)
    last = history[-1]
    if last.get("target_tokens_seen") is not None:
        return int(last["target_tokens_seen"])
    return final_tokens(report)


def first_row_at_or_below_loss(report: dict[str, Any], target_loss: float) -> dict[str, Any] | None:
    eval_history = eval_loss_history(report)
    if eval_history:
        for row in eval_history:
            if float(row["eval_loss"]) <= float(target_loss):
                return row
        return None
    for row in loss_history(report):
        if float(row["loss"]) <= float(target_loss):
            return row
    return None


def row_tokens(row: dict[str, Any]) -> int:
    return int(row["tokens_seen"])


def row_target_tokens(row: dict[str, Any]) -> int:
    value = row.get("target_tokens_seen")
    if value is None:
        value = row.get("tokens_seen")
    return int(value)


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": report.get("_report_path"),
        "target_level": report.get("target_level"),
        "contract": dataset_contract(report),
        "eval_contract": eval_dataset_contract(report) if metric_source(report) == "eval_loss_history" else {},
        "eval_coverage": eval_coverage(report),
        "metric_source": metric_source(report),
        "final_loss": final_loss(report),
        "tokens_seen": final_tokens(report),
        "target_tokens_seen": final_target_tokens(report),
    }


def compare_reports(
    baseline_report: str | Path,
    candidate_report: str | Path,
    *,
    factor: float = 10.0,
) -> dict[str, Any]:
    baseline = load_report(baseline_report)
    candidate = load_report(candidate_report)
    mismatches = contract_mismatches(baseline, candidate)
    comparable = not mismatches
    target_loss = final_loss(baseline)
    baseline_tokens = final_tokens(baseline)
    baseline_target_tokens = final_target_tokens(baseline)
    max_tokens = float(baseline_tokens) / float(factor)
    max_target_tokens = float(baseline_target_tokens) / float(factor)
    candidate_row_to_target = first_row_at_or_below_loss(candidate, target_loss) if comparable else None
    candidate_tokens_to_target = row_tokens(candidate_row_to_target) if candidate_row_to_target else None
    candidate_target_tokens_to_target = (
        row_target_tokens(candidate_row_to_target) if candidate_row_to_target else None
    )
    candidate_final = final_loss(candidate)
    candidate_final_tokens = final_tokens(candidate)
    candidate_final_target_tokens = final_target_tokens(candidate)
    baseline_row_to_candidate_final = (
        first_row_at_or_below_loss(baseline, candidate_final) if comparable else None
    )
    baseline_tokens_to_candidate_final = (
        row_tokens(baseline_row_to_candidate_final) if baseline_row_to_candidate_final else None
    )
    baseline_target_tokens_to_candidate_final = (
        row_target_tokens(baseline_row_to_candidate_final) if baseline_row_to_candidate_final else None
    )

    verdict = "unproven"
    reasons: list[str] = []
    if not comparable:
        verdict = "invalid_comparison"
        reasons.append("baseline and candidate PrefixLM dataset contracts differ")
    elif candidate_tokens_to_target is None:
        reasons.append("candidate did not reach the baseline final loss in logged history")
    elif (
        float(candidate_tokens_to_target) <= max_tokens
        and float(candidate_target_tokens_to_target) <= max_target_tokens
    ):
        verdict = "supports_10x_on_prefixlm_loss"
    else:
        if float(candidate_tokens_to_target) > max_tokens:
            reasons.append(
                "candidate reached baseline final loss at "
                f"{candidate_tokens_to_target} total tokens, later than the 10x cutoff "
                f"{max_tokens:.1f}"
            )
        if float(candidate_target_tokens_to_target) > max_target_tokens:
            reasons.append(
                "candidate reached baseline final loss at "
                f"{candidate_target_tokens_to_target} target tokens, later than the 10x cutoff "
                f"{max_target_tokens:.1f}"
            )

    return {
        "verdict": verdict,
        "reasons": reasons,
        "factor": float(factor),
        "comparable": comparable,
        "contract_mismatches": mismatches,
        "baseline": summarize(baseline),
        "candidate": summarize(candidate),
        "metrics": {
            "metric_source": metric_source(baseline)
            if metric_source(baseline) == metric_source(candidate)
            else f"{metric_source(baseline)} vs {metric_source(candidate)}",
            "baseline_final_loss": target_loss,
            "baseline_tokens_seen": baseline_tokens,
            "baseline_target_tokens_seen": baseline_target_tokens,
            "candidate_tokens_to_baseline_loss": candidate_tokens_to_target,
            "candidate_target_tokens_to_baseline_loss": candidate_target_tokens_to_target,
            "max_tokens_for_factor_claim": max_tokens,
            "max_target_tokens_for_factor_claim": max_target_tokens,
            "candidate_final_loss": candidate_final,
            "candidate_final_tokens_seen": candidate_final_tokens,
            "candidate_final_target_tokens_seen": candidate_final_target_tokens,
            "baseline_tokens_to_candidate_final_loss": baseline_tokens_to_candidate_final,
            "baseline_target_tokens_to_candidate_final_loss": baseline_target_tokens_to_candidate_final,
            "observed_candidate_speedup_at_candidate_final_loss": (
                float(baseline_tokens_to_candidate_final) / float(candidate_final_tokens)
                if baseline_tokens_to_candidate_final is not None and candidate_final_tokens > 0
                else None
            ),
            "observed_candidate_target_token_speedup_at_candidate_final_loss": (
                float(baseline_target_tokens_to_candidate_final)
                / float(candidate_final_target_tokens)
                if baseline_target_tokens_to_candidate_final is not None
                and candidate_final_target_tokens > 0
                else None
            ),
        },
        "plain_language_read": (
            "This asks whether both students used the same HRM-Text/Data-IO "
            "textbook and whether the candidate reached the baseline's final "
            "loss after seeing at most one/factor as many total tokens and "
            "supervised target tokens."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--factor", type=float, default=10.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = compare_reports(
        args.baseline_report,
        args.candidate_report,
        factor=float(args.factor),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report["verdict"] == "invalid_comparison":
        raise SystemExit(2)
    if report["verdict"] != "supports_10x_on_prefixlm_loss":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
