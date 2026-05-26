#!/usr/bin/env python3
"""Build Stage101L balanced source/anchor replay curriculum."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_stage101k_builder() -> Any:
    path = ROOT / "scripts" / "577_build_stage101k_polarity_balanced_source_probe.py"
    spec = importlib.util.spec_from_file_location("stage101k_builder_for_stage101l", path)
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


def clone_rows(rows: list[dict[str, Any]], *, prefix: str, factor: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for replay_index in range(max(1, int(factor))):
        for item in rows:
            cloned = dict(item)
            cloned["id"] = f"{item['id']}_{prefix}{replay_index:02d}"
            out.append(cloned)
    return out


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


def interleave_source_and_anchors(
    *,
    source_rows: list[dict[str, Any]],
    anchor_rows_replayed: list[dict[str, Any]],
    anchors_per_source: int,
) -> list[dict[str, Any]]:
    if not source_rows:
        raise ValueError("source_rows cannot be empty")
    if not anchor_rows_replayed:
        raise ValueError("anchor_rows_replayed cannot be empty")
    anchors_per_source = max(1, int(anchors_per_source))
    out: list[dict[str, Any]] = []
    anchor_index = 0
    for source_row in source_rows:
        for _ in range(anchors_per_source):
            out.append(anchor_rows_replayed[anchor_index % len(anchor_rows_replayed)])
            anchor_index += 1
        out.append(source_row)
    while anchor_index < len(anchor_rows_replayed):
        out.append(anchor_rows_replayed[anchor_index])
        anchor_index += 1
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    stage101k = load_stage101k_builder()
    source_base = stage101k.polarity_balanced_rows()
    anchors_base = anchor_rows([str(args.anchor_jsonl), str(args.extra_anchor_jsonl)])
    source_rows = clone_rows(source_base, prefix="source_replay", factor=int(args.source_replay_factor))
    anchor_replayed = clone_rows(anchors_base, prefix="anchor_replay", factor=int(args.anchor_replay_factor))
    train_rows = interleave_source_and_anchors(
        source_rows=source_rows,
        anchor_rows_replayed=anchor_replayed,
        anchors_per_source=int(args.anchors_per_source),
    )
    base_eval = load_jsonl(Path(args.base_eval_jsonl)) if str(args.base_eval_jsonl) else []
    eval_rows = base_eval + stage101k.heldout_rows()
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    report = {
        "decision": "built_stage101l_balanced_replay_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "source_base_rows": int(len(source_base)),
        "anchor_base_rows": int(len(anchors_base)),
        "source_replay_factor": int(args.source_replay_factor),
        "anchor_replay_factor": int(args.anchor_replay_factor),
        "anchors_per_source": int(args.anchors_per_source),
        "train_rows": int(len(train_rows)),
        "eval_rows": int(len(eval_rows)),
        "plain_language_read": (
            "Stage101L keeps the source-reading curriculum but makes old "
            "shortcut/truthy anchors appear repeatedly between source rows, so "
            "the student cannot learn source rows by forgetting the old exam."
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
    parser.add_argument("--train-out", default="data/eval/stage101l_balanced_replay_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101l_balanced_replay_heldout_probe.jsonl")
    parser.add_argument("--source-replay-factor", type=int, default=1)
    parser.add_argument("--anchor-replay-factor", type=int, default=6)
    parser.add_argument("--anchors-per-source", type=int, default=2)
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
