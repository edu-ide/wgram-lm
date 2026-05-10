#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_GENERATION_JSONL = (
    "/mnt/nvme1n1p2/qtrm-runs/"
    "qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_loop_"
    "joint_decoder_s040_from_selfrollout/generation_smoke8.jsonl"
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "mode" not in row:
                raise ValueError(f"{path}:{line_no}: missing mode")
            if "hit" not in row:
                raise ValueError(f"{path}:{line_no}: missing hit")
            rows.append(row)
    if not rows:
        raise ValueError(f"no generation rows in {path}")
    return rows


def summarize_by_mode(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_mode: dict[str, dict[str, Any]] = {}
    for row in rows:
        mode = str(row["mode"])
        bucket = by_mode.setdefault(mode, {"hits": 0, "total": 0})
        bucket["hits"] += int(bool(row.get("hit")))
        bucket["total"] += 1
    for bucket in by_mode.values():
        bucket["accuracy"] = float(bucket["hits"]) / max(1, int(bucket["total"]))
        bucket["exact"] = f"{bucket['hits']}/{bucket['total']}"
    return by_mode


def mode_accuracy(by_mode: dict[str, dict[str, Any]], mode: str) -> float | None:
    item = by_mode.get(str(mode))
    if item is None:
        return None
    return float(item["accuracy"])


def build_gate_report(
    *,
    generation_jsonl: str | Path,
    out_dir: str | Path,
    full_mode: str,
    core_off_mode: str,
    donor_mode: str,
    ablation_mode: str,
    min_full_accuracy: float,
    min_core_off_drop: float,
    min_ablation_drop: float,
) -> dict[str, Any]:
    rows = load_jsonl(generation_jsonl)
    by_mode = summarize_by_mode(rows)
    full = mode_accuracy(by_mode, full_mode)
    core_off = mode_accuracy(by_mode, core_off_mode)
    donor = mode_accuracy(by_mode, donor_mode)
    ablation = mode_accuracy(by_mode, ablation_mode)

    reject_reasons: list[str] = []
    if full is None:
        reject_reasons.append("missing_full_mode")
        full = 0.0
    if full < float(min_full_accuracy):
        reject_reasons.append("full_generation_accuracy_below_min")
    full_minus_core_off = None
    if core_off is None:
        reject_reasons.append("missing_core_off_mode")
    else:
        full_minus_core_off = float(full) - float(core_off)
        if full_minus_core_off < float(min_core_off_drop):
            reject_reasons.append("core_off_drop_below_min")
    full_minus_ablation = None
    if ablation is not None:
        full_minus_ablation = float(full) - float(ablation)
        if full_minus_ablation < float(min_ablation_drop):
            reject_reasons.append("ablation_drop_below_min")
    full_minus_donor = None
    if donor is not None:
        full_minus_donor = float(full) - float(donor)

    decision = "accepted_l3_candidate" if not reject_reasons else "rejected"
    report = {
        "status": "complete",
        "target_level": "L3 candidate",
        "decision": decision,
        "accepted": not reject_reasons,
        "reject_reasons": reject_reasons,
        "source_artifacts": {
            "generation_jsonl": str(generation_jsonl),
        },
        "modes": {
            "full_mode": full_mode,
            "core_off_mode": core_off_mode,
            "donor_mode": donor_mode,
            "ablation_mode": ablation_mode,
        },
        "metrics": {
            "full_generation_accuracy": full,
            "core_off_generation_accuracy": core_off,
            "donor_generation_accuracy": donor,
            "ablation_generation_accuracy": ablation,
            "full_minus_core_off": full_minus_core_off,
            "full_minus_donor": full_minus_donor,
            "full_minus_ablation": full_minus_ablation,
        },
        "by_mode": by_mode,
        "next_action": (
            "promote renderer candidate to broader held-out generation gate"
            if not reject_reasons
            else "renderer remains bottleneck; design a donor-compatible text renderer before memory/metacognition expansion"
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
            "Build a canonical LM renderer gate from generation JSONL. "
            "This gate rejects forced-choice or primitive-executor success unless "
            "normal greedy/autoregressive generation improves."
        )
    )
    parser.add_argument("--generation-jsonl", default=DEFAULT_GENERATION_JSONL)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--full-mode", default="qtrm_core_steps_8_no_evidence")
    parser.add_argument("--core-off-mode", default="qtrm_core_off_no_evidence")
    parser.add_argument("--donor-mode", default="donor_only_no_evidence")
    parser.add_argument(
        "--ablation-mode",
        default="qtrm_core_steps_8_answer_next_token_decoder_off_no_evidence",
    )
    parser.add_argument("--min-full-accuracy", type=float, default=0.50)
    parser.add_argument("--min-core-off-drop", type=float, default=0.25)
    parser.add_argument("--min-ablation-drop", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_gate_report(
        generation_jsonl=args.generation_jsonl,
        out_dir=args.out_dir,
        full_mode=args.full_mode,
        core_off_mode=args.core_off_mode,
        donor_mode=args.donor_mode,
        ablation_mode=args.ablation_mode,
        min_full_accuracy=args.min_full_accuracy,
        min_core_off_drop=args.min_core_off_drop,
        min_ablation_drop=args.min_ablation_drop,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
