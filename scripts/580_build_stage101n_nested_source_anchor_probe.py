#!/usr/bin/env python3
"""Build Stage101N nested source-microburst plus anchor-lock curriculum."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_stage101k_builder() -> Any:
    path = ROOT / "scripts" / "577_build_stage101k_polarity_balanced_source_probe.py"
    spec = importlib.util.spec_from_file_location("stage101k_builder_for_stage101n", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def anchor_rows(paths: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        if not path:
            continue
        for item in load_jsonl(Path(path)):
            row_id = str(item.get("id", ""))
            if row_id.startswith("gd_lite_") or row_id.startswith("stage101b_"):
                out.append(item)
    return out


def grouped_source_rows(source_rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        concept = str(row.get("source_concept") or "unknown")
        grouped[concept].append(row)
    return [(concept, grouped[concept]) for concept in sorted(grouped)]


def clone_with_phase(
    row: dict[str, Any],
    *,
    suffix: str,
    phase: str,
    burst_concept: str,
) -> dict[str, Any]:
    cloned = dict(row)
    cloned["id"] = f"{row['id']}_{suffix}"
    cloned["nested_phase"] = phase
    cloned["nested_burst_concept"] = burst_concept
    return cloned


def build_microburst_curriculum(
    *,
    source_rows: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    source_replay_factor: int,
    anchor_lock_replay_factor: int,
) -> list[dict[str, Any]]:
    if not source_rows:
        raise ValueError("source_rows cannot be empty")
    if not anchors:
        raise ValueError("anchors cannot be empty")
    source_replay_factor = max(1, int(source_replay_factor))
    anchor_lock_replay_factor = max(1, int(anchor_lock_replay_factor))
    out: list[dict[str, Any]] = []
    for burst_index, (concept, concept_rows) in enumerate(grouped_source_rows(source_rows)):
        for replay_index in range(source_replay_factor):
            for row in concept_rows:
                out.append(
                    clone_with_phase(
                        row,
                        suffix=f"sourceburst{burst_index:02d}_replay{replay_index:02d}",
                        phase="source_microburst",
                        burst_concept=concept,
                    )
                )
        for replay_index in range(anchor_lock_replay_factor):
            for row in anchors:
                out.append(
                    clone_with_phase(
                        row,
                        suffix=f"anchorlock{burst_index:02d}_replay{replay_index:02d}",
                        phase="anchor_lock",
                        burst_concept=concept,
                    )
                )
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    stage101k = load_stage101k_builder()
    source_rows = stage101k.polarity_balanced_rows()
    anchors = anchor_rows([str(args.anchor_jsonl), str(args.extra_anchor_jsonl)])
    train_rows = build_microburst_curriculum(
        source_rows=source_rows,
        anchors=anchors,
        source_replay_factor=int(args.source_replay_factor),
        anchor_lock_replay_factor=int(args.anchor_lock_replay_factor),
    )
    base_eval = load_jsonl(Path(args.base_eval_jsonl)) if str(args.base_eval_jsonl) else []
    eval_rows = base_eval + stage101k.heldout_rows()
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    source_microburst_rows = sum(1 for row in train_rows if row.get("nested_phase") == "source_microburst")
    anchor_lock_rows = sum(1 for row in train_rows if row.get("nested_phase") == "anchor_lock")
    report = {
        "decision": "built_stage101n_nested_source_anchor_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "source_base_rows": int(len(source_rows)),
        "anchor_base_rows": int(len(anchors)),
        "source_replay_factor": int(args.source_replay_factor),
        "anchor_lock_replay_factor": int(args.anchor_lock_replay_factor),
        "source_microburst_rows": int(source_microburst_rows),
        "anchor_lock_rows": int(anchor_lock_rows),
        "train_rows": int(len(train_rows)),
        "eval_rows": int(len(eval_rows)),
        "plain_language_read": (
            "Stage101N teaches one source concept as a short burst, then "
            "immediately locks the old shortcut-resistance attractor with "
            "anchor replay. This tests a nested-learning schedule instead of "
            "a flat source/anchor mixture."
        ),
    }
    if str(args.report_out):
        report_out = Path(args.report_out)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-jsonl", default="data/eval/generalization_dynamics_lite_probe.jsonl")
    parser.add_argument("--extra-anchor-jsonl", default="data/eval/stage101b_solution_attractor_heldout_probe.jsonl")
    parser.add_argument("--base-eval-jsonl", default="data/eval/stage101g_source_grounded_paraphrase_heldout_probe.jsonl")
    parser.add_argument("--train-out", default="data/eval/stage101n_nested_source_anchor_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101n_nested_source_anchor_heldout_probe.jsonl")
    parser.add_argument("--source-replay-factor", type=int, default=1)
    parser.add_argument("--anchor-lock-replay-factor", type=int, default=1)
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
