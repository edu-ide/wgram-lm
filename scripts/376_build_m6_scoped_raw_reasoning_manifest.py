#!/usr/bin/env python3
"""Build the M6 scoped raw-reasoning comparison manifest.

M6 is a model-win gate, not a harness-only gate. A QTRM report is not enough:
the same deterministic suite needs a Qwen3.6-27B baseline report before this
script can accept the milestone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SUITE_ID = "qtrm_native_text_reasoning_modchain_revchain_checksum_program4_mod32"
DEFAULT_PROMPT_PROTOCOL = "operation_definitions_v1"
ABLATION_KEYS = (
    "state_reset",
    "op_zero",
    "thinking_block_off",
    "z_l_zero",
    "z_h_zero",
    "carrier_off",
)


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None or str(path).strip() == "":
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _float_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _metric_score(row: dict[str, Any] | None, key: str = "generation_exact") -> float | None:
    if not isinstance(row, dict):
        return None
    return _float_or_none(row.get(key))


def _preferred_think_key(report: dict[str, Any], preferred: str) -> str:
    metrics = report.get("eval_metrics", {})
    if preferred and preferred in metrics:
        return preferred
    train = report.get("train", {})
    eval_steps = train.get("eval_think_steps")
    if isinstance(eval_steps, int) and f"think{eval_steps}" in metrics:
        return f"think{eval_steps}"
    think_keys = [
        key
        for key, value in metrics.items()
        if key.startswith("think")
        and key[5:].isdigit()
        and isinstance(value, dict)
        and _metric_score(value) is not None
    ]
    if not think_keys:
        return ""
    return max(think_keys, key=lambda key: _metric_score(metrics[key]) or -1.0)


def summarize_qtrm_report(
    *,
    path: str | Path,
    report: dict[str, Any],
    suite_id: str,
    preferred_think_key: str = "",
) -> dict[str, Any]:
    metrics = report.get("eval_metrics", {})
    decisive = report.get("decisive_metrics", {})
    full_key = _preferred_think_key(report, preferred_think_key)
    full_row = metrics.get(full_key, {}) if full_key else {}
    full_score = (
        _float_or_none(decisive.get("full_generation_exact"))
        if isinstance(decisive, dict)
        else None
    )
    if full_score is None:
        full_score = _metric_score(full_row)
    think0_score = (
        _float_or_none(decisive.get("think0_generation_exact"))
        if isinstance(decisive, dict)
        else None
    )
    if think0_score is None:
        think0_score = _metric_score(metrics.get("think0", {}))

    ablations: dict[str, float] = {}
    for key in ABLATION_KEYS:
        score = _metric_score(metrics.get(key, {}))
        if score is not None:
            ablations[key] = score
    worst_ablation = max(ablations.values()) if ablations else None
    ablation_drop = (
        _float_or_none(decisive.get("full_minus_worst_ablation"))
        if isinstance(decisive, dict)
        else None
    )
    if ablation_drop is None and full_score is not None and worst_ablation is not None:
        ablation_drop = float(full_score - worst_ablation)

    core_gain = (
        _float_or_none(decisive.get("full_minus_think0"))
        if isinstance(decisive, dict)
        else None
    )
    if core_gain is None and full_score is not None and think0_score is not None:
        core_gain = float(full_score - think0_score)

    by_family = full_row.get("by_family", {}) if isinstance(full_row, dict) else {}
    family_scores = [
        _metric_score(value)
        for value in by_family.values()
        if isinstance(value, dict) and _metric_score(value) is not None
    ]
    min_family = (
        _float_or_none(decisive.get("min_family_generation_exact"))
        if isinstance(decisive, dict)
        else None
    )
    if min_family is None and family_scores:
        min_family = min(family_scores)

    train = report.get("train", {}) if isinstance(report.get("train", {}), dict) else {}
    suite_spec = {
        "suite_id": str(suite_id),
        "task_families": (
            report.get("task_families")
            or report.get("eval_task_families")
            or train.get("eval_task_families")
            or train.get("task_families")
            or ""
        ),
        "program_len": train.get("program_len"),
        "modulus": train.get("modulus"),
        "eval_cases": train.get("eval_cases"),
        "eval_seed": train.get("eval_seed"),
        "include_family_tag": report.get("include_family_tag", train.get("include_family_tag")),
    }
    return {
        "path": str(path),
        "accepted": bool(report.get("accepted", False)),
        "decision": str(report.get("decision", "")),
        "target_level": str(report.get("target_level", "")),
        "suite": suite_spec,
        "selected_think_key": full_key,
        "full_generation_exact": full_score,
        "think0_generation_exact": think0_score,
        "core_gain": core_gain,
        "ablation_scores": ablations,
        "worst_ablation_generation_exact": worst_ablation,
        "ablation_drop": ablation_drop,
        "min_family_generation_exact": min_family,
        "cases": full_row.get("cases") if isinstance(full_row, dict) else None,
    }


def qwen36_baseline_summary(
    report: dict[str, Any] | None,
    *,
    path: str = "",
) -> dict[str, Any] | None:
    if report is None:
        return None
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics", {}), dict) else {}
    score = _first_float(
        report.get("score"),
        report.get("generation_exact"),
        metrics.get("generation_exact"),
        metrics.get("exact"),
        metrics.get("accuracy"),
    )
    return {
        "path": str(path),
        "model": str(report.get("model", "Qwen/Qwen3.6-27B")),
        "suite_id": str(report.get("suite_id", "")),
        "prompt_protocol": str(report.get("prompt_protocol", "")),
        "score": score,
        "cases": report.get("cases") or metrics.get("cases"),
        "scorer": str(report.get("scorer", "exact match")),
        "accepted": bool(report.get("accepted", score is not None)),
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    qtrm_reports = []
    for path in getattr(args, "qtrm_report", []) or []:
        report = _load_json(path)
        if report is None:
            continue
        qtrm_reports.append(
            summarize_qtrm_report(
                path=path,
                report=report,
                suite_id=str(args.suite_id),
                preferred_think_key=str(args.preferred_think_key),
            )
        )
    best = (
        max(
            qtrm_reports,
            key=lambda row: row["full_generation_exact"]
            if row.get("full_generation_exact") is not None
            else -1.0,
        )
        if qtrm_reports
        else None
    )
    baseline_report = _load_json(args.qwen36_baseline_report)
    baseline = qwen36_baseline_summary(
        baseline_report,
        path=str(args.qwen36_baseline_report).strip(),
    )

    qtrm_score = best.get("full_generation_exact") if best else None
    qwen_score = baseline.get("score") if baseline else None
    core_gain = best.get("core_gain") if best else None
    ablation_drop = best.get("ablation_drop") if best else None
    min_family = best.get("min_family_generation_exact") if best else None
    suite_matches = bool(
        baseline
        and baseline.get("suite_id")
        and str(baseline.get("suite_id")) == str(args.suite_id)
    )
    protocol_matches = bool(
        baseline
        and baseline.get("prompt_protocol")
        and str(baseline.get("prompt_protocol")) == str(args.prompt_protocol)
    )
    beats_baseline = bool(
        qtrm_score is not None
        and qwen_score is not None
        and float(qtrm_score) >= float(qwen_score) + float(args.min_margin)
    )
    checks = {
        "qtrm_report_present": bool(qtrm_reports),
        "qtrm_best_report_accepted": bool(best and best.get("accepted")),
        "qtrm_has_generation_exact": qtrm_score is not None,
        "qtrm_core_gain_ge_min": bool(
            core_gain is not None and float(core_gain) >= float(args.min_core_gain)
        ),
        "qtrm_ablation_drop_ge_min": bool(
            ablation_drop is not None
            and float(ablation_drop) >= float(args.min_ablation_drop)
        ),
        "qtrm_min_family_ge_min": bool(
            min_family is not None and float(min_family) >= float(args.min_family_exact)
        ),
        "qwen36_baseline_present": baseline is not None,
        "qwen36_baseline_has_score": qwen_score is not None,
        "qwen36_suite_id_matches": suite_matches,
        "qwen36_prompt_protocol_matches": protocol_matches,
        "qtrm_beats_qwen36_baseline": beats_baseline,
    }
    reject_reasons = [key for key, value in checks.items() if not value]
    accepted = not reject_reasons
    return {
        "decision": "accepted_m6_scoped_raw_reasoning_win" if accepted else "rejected",
        "accepted": accepted,
        "target_level": "M6 scoped raw reasoning win over Qwen3.6-27B",
        "suite_id": str(args.suite_id),
        "prompt_protocol": str(args.prompt_protocol),
        "comparison_mode": "matched_scoped_suite_direct_baseline",
        "qtrm_native_reports": qtrm_reports,
        "best_qtrm_native": best,
        "qwen36_baseline": baseline,
        "thresholds": {
            "min_margin": float(args.min_margin),
            "min_core_gain": float(args.min_core_gain),
            "min_ablation_drop": float(args.min_ablation_drop),
            "min_family_exact": float(args.min_family_exact),
        },
        "acceptance_checks": checks,
        "reject_reasons": reject_reasons,
        "limitations": [
            "M6 cannot be accepted from a QTRM report alone.",
            "The Qwen3.6-27B baseline must be measured on the same deterministic scoped suite.",
            "This is a scoped raw-reasoning win only; it is not public benchmark parity.",
        ],
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    best = manifest.get("best_qtrm_native") or {}
    baseline = manifest.get("qwen36_baseline") or {}
    lines = [
        "# M6 Scoped Raw-Reasoning Manifest",
        "",
        f"Decision: `{manifest['decision']}`",
        f"Accepted: `{manifest['accepted']}`",
        f"Suite: `{manifest['suite_id']}`",
        "",
        "## Scores",
        "",
        "| Model | Score | Cases | Report |",
        "|---|---:|---:|---|",
        (
            f"| QTRM-Native | {best.get('full_generation_exact')} | "
            f"{best.get('cases')} | `{best.get('path', '')}` |"
        ),
        (
            f"| Qwen3.6-27B | {baseline.get('score')} | "
            f"{baseline.get('cases')} | `{baseline.get('path', '')}` |"
        ),
        "",
        "## Causality",
        "",
        f"- Core gain over think0: `{best.get('core_gain')}`",
        f"- Drop versus worst ablation: `{best.get('ablation_drop')}`",
        f"- Min family exact: `{best.get('min_family_generation_exact')}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in manifest["acceptance_checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Reject Reasons", ""])
    for reason in manifest["reject_reasons"]:
        lines.append(f"- `{reason}`")
    lines.extend(["", "## Limitations", ""])
    for item in manifest["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", default="local_eval/m6_scoped_raw_reasoning_manifest/report.json")
    parser.add_argument("--out-md", default="local_eval/m6_scoped_raw_reasoning_manifest/report.md")
    parser.add_argument("--suite-id", default=DEFAULT_SUITE_ID)
    parser.add_argument("--prompt-protocol", default=DEFAULT_PROMPT_PROTOCOL)
    parser.add_argument("--qtrm-report", action="append", default=[])
    parser.add_argument("--qwen36-baseline-report", default="")
    parser.add_argument("--preferred-think-key", default="")
    parser.add_argument("--min-margin", type=float, default=0.01)
    parser.add_argument("--min-core-gain", type=float, default=0.05)
    parser.add_argument("--min-ablation-drop", type=float, default=0.05)
    parser.add_argument("--min-family-exact", type=float, default=0.20)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest = build_manifest(args)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(manifest), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    raise SystemExit(0 if bool(manifest["accepted"]) else 1)


if __name__ == "__main__":
    main()
