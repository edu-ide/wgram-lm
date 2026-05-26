#!/usr/bin/env python3
"""Check whether a current run restores the observability of old Stage56/58 success."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_SIGNAL_NAMES = (
    "teacher_forced_heldout_loss",
    "free_generation_samples",
    "first_response_token_rank_or_topk",
    "repetition_and_eos_rate",
    "selected_vs_oracle_split_when_search_is_used",
    "depth_or_recurrent_core_off_ablation",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _signal(name: str, present: bool, *, required: bool = True, evidence: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "present": bool(present),
        "required": bool(required),
        "evidence": str(evidence),
    }


def _has_teacher_forced_loss(reports: list[dict[str, Any]]) -> tuple[bool, str]:
    for index, report in enumerate(reports):
        if str(report.get("metric_family", "")) == "teacher_forced_loss":
            return True, f"teacher report {index} has metric_family=teacher_forced_loss"
        for key in ("final_eval_loss", "eval_loss", "loss"):
            if _is_number(report.get(key)):
                return True, f"teacher report {index} has {key}={report[key]}"
        language = report.get("language")
        if isinstance(language, dict) and _is_number(language.get("loss")):
            return True, f"teacher report {index} has language.loss={language['loss']}"
    return False, ""


def _has_free_generation_samples(reports: list[dict[str, Any]]) -> tuple[bool, str]:
    for index, report in enumerate(reports):
        generation = report.get("generation") if isinstance(report.get("generation"), dict) else report
        samples = generation.get("samples") if isinstance(generation, dict) else None
        if isinstance(samples, list) and any(isinstance(sample, dict) for sample in samples):
            return True, f"generation report {index} has {len(samples)} sample(s)"
        rows = report.get("rows")
        if isinstance(rows, list) and any(
            isinstance(row, dict) and ("raw_response" in row or "generated" in row or "generated_ids" in row)
            for row in rows
        ):
            return True, f"generation report {index} has row-level generations"
    return False, ""


def _has_first_response_stats(reports: list[dict[str, Any]]) -> tuple[bool, str]:
    for index, report in enumerate(reports):
        first = report.get("first_response")
        if isinstance(first, dict) and any(
            key in first
            for key in (
                "accuracy",
                "gold_probability",
                "eoa_top1_fraction",
                "common_top1",
            )
        ):
            return True, f"generation report {index} has first_response stats"
        if "first_response_token_rank" in report or "first_response_token_topk" in report:
            return True, f"generation report {index} has first-response rank/top-k"
    return False, ""


def _has_repetition_and_eos_stats(reports: list[dict[str, Any]]) -> tuple[bool, str]:
    for index, report in enumerate(reports):
        generation = report.get("generation") if isinstance(report.get("generation"), dict) else report
        if not isinstance(generation, dict):
            continue
        has_repetition = any(
            key in generation
            for key in (
                "repeated_token_loop_fraction",
                "degenerate_repetition_rate",
                "repetition_rate",
            )
        )
        has_eos = any(
            key in generation
            for key in (
                "starts_with_eoa_fraction",
                "ended_with_eoa_fraction",
                "eos_rate",
                "eoa_top1_fraction",
            )
        )
        if has_repetition and has_eos:
            return True, f"generation report {index} has repetition and EOS/EOA stats"
    return False, ""


def _has_selected_oracle_split(reports: list[dict[str, Any]]) -> tuple[bool, str]:
    for index, report in enumerate(reports):
        if (
            _is_number(report.get("selected_accuracy"))
            and _is_number(report.get("oracle_accuracy"))
        ):
            return True, f"search report {index} has selected_accuracy and oracle_accuracy"
        if (
            _is_number(report.get("mean_selected_accuracy_oracle_depth"))
            and _is_number(report.get("mean_oracle_accuracy"))
        ):
            return True, f"search report {index} has mean selected/oracle accuracy"
        history = report.get("history")
        if isinstance(history, list) and history:
            last = history[-1]
            eval_payload = last.get("eval") if isinstance(last, dict) else None
            if isinstance(eval_payload, dict) and (
                _is_number(eval_payload.get("mean_selected_accuracy_oracle_depth"))
                and _is_number(eval_payload.get("mean_oracle_accuracy"))
            ):
                return True, f"search report {index} has history[-1].eval selected/oracle"
    return False, ""


def _has_depth_or_recurrent_ablation(reports: list[dict[str, Any]]) -> tuple[bool, str]:
    for index, report in enumerate(reports):
        depth_summaries = report.get("depth_summaries")
        if isinstance(depth_summaries, list) and len(depth_summaries) >= 2:
            depths = [
                int(row.get("think_steps"))
                for row in depth_summaries
                if isinstance(row, dict) and row.get("think_steps") is not None
            ]
            return True, f"depth report {index} has depth_summaries={depths}"
        rows = report.get("rows")
        if isinstance(rows, list):
            depths = {
                int(row["think_steps"])
                for row in rows
                if isinstance(row, dict) and row.get("think_steps") is not None
            }
            if len(depths) >= 2:
                return True, f"depth report {index} has row depths={sorted(depths)}"
        if any(
            key in report
            for key in (
                "core_off_loss",
                "recurrent_off_loss",
                "depth_off_loss",
                "ablation_generation_accuracy",
                "core_off_generation_accuracy",
            )
        ):
            return True, f"depth report {index} has recurrent/core-off ablation"
    return False, ""


def _metric_warnings(
    *,
    generation_reports: list[dict[str, Any]],
    depth_reports: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    for report in generation_reports:
        generation = report.get("generation") if isinstance(report.get("generation"), dict) else report
        if not isinstance(generation, dict):
            continue
        if float(generation.get("exact_fraction", 1.0) or 0.0) <= 0.0:
            warnings.append("free_generation_exact_zero")
        if (
            "ended_with_eoa_fraction" in generation
            and float(generation.get("ended_with_eoa_fraction") or 0.0) <= 0.0
        ):
            warnings.append("generation_never_reaches_eos")
        if float(generation.get("repeated_token_loop_fraction", 0.0) or 0.0) >= 0.25:
            warnings.append("high_repeated_token_loop_fraction")
        first_response = report.get("first_response")
        if isinstance(first_response, dict):
            common_targets = first_response.get("common_targets")
            positions = int(first_response.get("positions", 0) or 0)
            if isinstance(common_targets, list) and common_targets and positions > 0:
                top_target = common_targets[0]
                if (
                    isinstance(top_target, dict)
                    and str(top_target.get("decoded", "")) == " "
                    and int(top_target.get("count", 0) or 0) == positions
                ):
                    warnings.append("first_response_space_only_target")
        continuation = report.get("response_continuation")
        if isinstance(continuation, dict):
            if float(continuation.get("continuation_accuracy", 1.0) or 0.0) < 0.2:
                warnings.append("low_response_continuation_accuracy")
            if (
                int(continuation.get("eos_targets", 0) or 0) > 0
                and float(continuation.get("eos_top1_accuracy", 1.0) or 0.0) <= 0.0
            ):
                warnings.append("eos_teacher_forced_top1_zero")
    for report in depth_reports:
        if report.get("accepted") is False:
            warnings.append("depth_probe_rejected")
        failed = report.get("failed_checks")
        if isinstance(failed, list):
            for item in failed:
                warnings.append(f"depth_failed:{item}")
    return sorted(set(warnings))


def build_restoration_gate_report(
    *,
    teacher_forced_reports: list[dict[str, Any]],
    generation_reports: list[dict[str, Any]],
    search_reports: list[dict[str, Any]],
    depth_reports: list[dict[str, Any]],
    require_search_split: bool,
) -> dict[str, Any]:
    teacher_present, teacher_evidence = _has_teacher_forced_loss(teacher_forced_reports)
    generation_present, generation_evidence = _has_free_generation_samples(generation_reports)
    first_present, first_evidence = _has_first_response_stats(generation_reports)
    repeat_present, repeat_evidence = _has_repetition_and_eos_stats(generation_reports)
    search_present, search_evidence = _has_selected_oracle_split(search_reports)
    depth_present, depth_evidence = _has_depth_or_recurrent_ablation(depth_reports)

    signals = [
        _signal("teacher_forced_heldout_loss", teacher_present, evidence=teacher_evidence),
        _signal("free_generation_samples", generation_present, evidence=generation_evidence),
        _signal("first_response_token_rank_or_topk", first_present, evidence=first_evidence),
        _signal("repetition_and_eos_rate", repeat_present, evidence=repeat_evidence),
        _signal(
            "selected_vs_oracle_split_when_search_is_used",
            search_present,
            required=bool(require_search_split),
            evidence=search_evidence if require_search_split else (
                search_evidence or "not required because search split was not claimed for this run"
            ),
        ),
        _signal("depth_or_recurrent_core_off_ablation", depth_present, evidence=depth_evidence),
    ]
    missing = [signal["name"] for signal in signals if signal["required"] and not signal["present"]]
    all_required = not missing
    warnings = _metric_warnings(
        generation_reports=generation_reports,
        depth_reports=depth_reports,
    )
    if not all_required:
        checkpoint_recommendation = "not_interpretable"
    elif warnings:
        checkpoint_recommendation = "do_not_promote_current_checkpoint"
    else:
        checkpoint_recommendation = "review_metric_quality_before_promotion"
    return {
        "gate_type": "past_success_restoration_gate",
        "all_required_signals_present": bool(all_required),
        "missing_required_signals": missing,
        "metric_warnings": warnings,
        "signals": signals,
        "launch_recommendation": (
            "restoration_gate_exists_review_metrics"
            if all_required
            else "do_not_launch_long_run_missing_restoration_signals"
        ),
        "current_checkpoint_recommendation": checkpoint_recommendation,
        "plain_korean_read": (
            "이 gate는 시험지를 한 번에 본다: loss가 내려갔는지, 직접 말할 수 있는지, "
            "첫 답 토큰이 막혔는지, 반복/EOS로 망가지는지, 후보-검증을 쓰면 selected/oracle을 "
            "분리했는지, 그리고 생각 depth나 recurrent core가 실제 차이를 냈는지 확인한다."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Past-Success Restoration Gate",
        "",
        report["plain_korean_read"],
        "",
        f"Launch recommendation: `{report.get('launch_recommendation', '')}`",
        f"Current checkpoint recommendation: `{report.get('current_checkpoint_recommendation', '')}`",
        "",
        "| Signal | Required | Present | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for signal in report.get("signals", []):
        lines.append(
            "| {name} | {required} | {present} | {evidence} |".format(
                name=signal.get("name", ""),
                required=signal.get("required", ""),
                present=signal.get("present", ""),
                evidence=signal.get("evidence", ""),
            )
        )
    if report.get("metric_warnings"):
        lines.extend(["", "Metric warnings:", ""])
        lines.extend(f"- `{warning}`" for warning in report.get("metric_warnings", []))
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-forced-report", action="append", default=[])
    parser.add_argument("--generation-report", action="append", default=[])
    parser.add_argument("--search-report", action="append", default=[])
    parser.add_argument("--depth-report", action="append", default=[])
    parser.add_argument("--require-search-split", action="store_true")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    report = build_restoration_gate_report(
        teacher_forced_reports=[_load_json(Path(path)) for path in args.teacher_forced_report],
        generation_reports=[_load_json(Path(path)) for path in args.generation_report],
        search_reports=[_load_json(Path(path)) for path in args.search_report],
        depth_reports=[_load_json(Path(path)) for path in args.depth_report],
        require_search_split=bool(args.require_search_split),
    )
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)


if __name__ == "__main__":
    main()
