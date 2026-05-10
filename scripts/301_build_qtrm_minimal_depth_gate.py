#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_DIR = (
    "local_eval/qwen35_2b_pure_recursive_primitive_transition_promptattn_s480_from_s720"
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def answer_runtime_accuracy(data: dict[str, Any]) -> float:
    summary = data.get("summary", data)
    if "answer_accuracy" not in summary:
        raise ValueError("answer runtime summary requires answer_accuracy")
    return float(summary["answer_accuracy"])


def answer_runtime_cases(data: dict[str, Any]) -> int:
    summary = data.get("summary", data)
    return int(summary.get("cases", 0))


def donor_accuracy(data: dict[str, Any], *, mode: str) -> float | None:
    summary = data.get("summary", data)
    by_mode = summary.get("by_mode")
    if not isinstance(by_mode, dict) or mode not in by_mode:
        return None
    return float(by_mode[mode].get("accuracy", 0.0))


def build_gate_report(
    *,
    full_json: str | Path,
    core_off_json: str | Path,
    donor_json: str | Path | None,
    out_dir: str | Path,
    min_full_accuracy: float,
    min_core_off_drop: float,
    min_donor_gain: float,
) -> dict[str, Any]:
    full = load_json(full_json)
    core_off = load_json(core_off_json)
    donor = load_json(donor_json) if donor_json else None

    full_accuracy = answer_runtime_accuracy(full)
    core_off_accuracy = answer_runtime_accuracy(core_off)
    donor_forced = donor_accuracy(donor, mode="forced_choice") if donor else None
    donor_greedy = donor_accuracy(donor, mode="greedy") if donor else None
    full_minus_core_off = full_accuracy - core_off_accuracy
    donor_baseline = donor_forced if donor_forced is not None else donor_greedy
    full_minus_donor = (
        full_accuracy - float(donor_baseline)
        if donor_baseline is not None
        else None
    )

    reject_reasons: list[str] = []
    if full_accuracy < float(min_full_accuracy):
        reject_reasons.append("full_accuracy_below_min")
    if full_minus_core_off < float(min_core_off_drop):
        reject_reasons.append("core_off_drop_below_min")
    if donor_baseline is not None and full_minus_donor is not None:
        if full_minus_donor < float(min_donor_gain):
            reject_reasons.append("donor_gain_below_min")
    elif donor_json:
        reject_reasons.append("donor_baseline_missing")

    decision = "accepted_l2" if not reject_reasons else "rejected"
    report = {
        "status": "complete",
        "target_level": "L2 local gate",
        "decision": decision,
        "accepted": not reject_reasons,
        "reject_reasons": reject_reasons,
        "source_artifacts": {
            "full_json": str(full_json),
            "core_off_json": str(core_off_json),
            "donor_json": str(donor_json) if donor_json else None,
        },
        "metrics": {
            "cases": answer_runtime_cases(full),
            "full_answer_accuracy": full_accuracy,
            "core_off_answer_accuracy": core_off_accuracy,
            "full_minus_core_off": full_minus_core_off,
            "donor_forced_choice_accuracy": donor_forced,
            "donor_greedy_accuracy": donor_greedy,
            "full_minus_donor": full_minus_donor,
        },
        "next_action": (
            "open renderer/canonical-LLM-path gate; primitive executor success is "
            "not yet normal autoregressive text generation"
            if not reject_reasons
            else "redesign QTRM minimal depth path before renderer or memory work"
        ),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an L2 QTRM minimal-depth gate from primitive runtime artifacts. "
            "This proves core-on beats donor/core-off only for the scaffold "
            "primitive executor path, not for normal LM generation."
        )
    )
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--full-json", default="")
    parser.add_argument("--core-off-json", default="")
    parser.add_argument("--donor-json", default="")
    parser.add_argument("--min-full-accuracy", type=float, default=0.95)
    parser.add_argument("--min-core-off-drop", type=float, default=0.50)
    parser.add_argument("--min-donor-gain", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    artifact_dir = Path(args.artifact_dir)
    full_json = args.full_json or artifact_dir / "eval_answer_runtime_heldout7000.json"
    core_off_json = args.core_off_json or artifact_dir / "eval_answer_runtime_heldout7000_coreoff.json"
    donor_json = args.donor_json or artifact_dir / "eval_donor_only_heldout7000.json"
    report = build_gate_report(
        full_json=full_json,
        core_off_json=core_off_json,
        donor_json=donor_json,
        out_dir=args.out_dir,
        min_full_accuracy=args.min_full_accuracy,
        min_core_off_drop=args.min_core_off_drop,
        min_donor_gain=args.min_donor_gain,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
