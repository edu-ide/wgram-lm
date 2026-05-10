#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


DEFAULT_CONFIG = (
    "configs/"
    "qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_"
    "typed_algorithmic_value_state_s080.yaml"
)
DEFAULT_CHECKPOINT = (
    "local_eval/"
    "qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_"
    "typed_algorithmic_value_state_len1113_s080_from_joint_s080/last.pt"
)
DEFAULT_STATE_KEYS = (
    "typed_algorithmic_kind_logits,"
    "typed_algorithmic_raw_list_offset_logits,"
    "typed_algorithmic_doubled_list_offset_logits,"
    "typed_algorithmic_scalar_coeff_logits,"
    "typed_algorithmic_scalar_residual_logits,"
    "typed_algorithmic_final_residual_logits"
)
DEFAULT_TRAIN_SOURCES = (
    "data/filtered/pure_recursive_reasoning_arith_chain_train128_start18.jsonl",
    "data/filtered/"
    "pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_"
    "len1113_probe_train40000_v0to5_mixed_only.jsonl",
)
DEFAULT_EVAL_SOURCES = (
    "data/eval/pure_recursive_reasoning_arith_chain_heldout18.jsonl",
    "data/eval/"
    "pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_"
    "len1113_probe_eval60000_v6to7_len11_13_mixed_only.jsonl",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _family(row: dict[str, Any]) -> str:
    return str(row.get("task_family") or row.get("category") or "unknown")


def _load_jsonl(path: str | Path, *, max_cases: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("id"):
                row["id"] = f"{Path(path).stem}-{line_no}"
            if not row.get("answer_aliases"):
                answer = str(row.get("answer") or row.get("chosen") or "").strip()
                if answer:
                    row["answer_aliases"] = [answer]
            if not row.get("answer_aliases"):
                raise ValueError(f"{path}:{line_no}: missing answer_aliases")
            if row.get("evidence"):
                raise ValueError(f"{path}:{line_no}: small-general gate forbids evidence")
            rows.append(row)
            if int(max_cases) > 0 and len(rows) >= int(max_cases):
                break
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def interleave_groups(groups: Iterable[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    buckets = [list(group) for group in groups if group]
    out: list[dict[str, Any]] = []
    max_len = max((len(group) for group in buckets), default=0)
    for index in range(max_len):
        for group in buckets:
            if index < len(group):
                out.append(group[index])
    return out


def build_mixed_gate_cases(
    *,
    sources: Iterable[str | Path],
    max_per_source: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    source_summary: list[dict[str, Any]] = []
    for source in sources:
        rows = _load_jsonl(source, max_cases=int(max_per_source))
        for row in rows:
            row = row
            row.setdefault("raw_intelligence_axis", "pure_recursive_reasoning")
            row["small_general_source"] = str(source)
        groups.append(rows)
        source_summary.append(
            {
                "path": str(source),
                "rows": len(rows),
                "families": sorted({_family(row) for row in rows}),
            }
        )
    mixed = interleave_groups(groups)
    summary = {
        "rows": len(mixed),
        "families": sorted({_family(row) for row in mixed}),
        "sources": source_summary,
    }
    return mixed, summary


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _load_generation_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_generation_by_family(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    by_mode_family: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        mode = str(row.get("mode") or "unknown")
        family = str(row.get("task_family") or row.get("category") or "unknown")
        bucket = by_mode_family.setdefault(mode, {}).setdefault(
            family,
            {"hits": 0, "exact": 0, "total": 0},
        )
        bucket["hits"] += int(bool(row.get("hit")))
        bucket["exact"] += int(
            bool(row.get("exact_match") or row.get("normalized_exact"))
        )
        bucket["total"] += 1
    for family_map in by_mode_family.values():
        for bucket in family_map.values():
            total = max(1, int(bucket["total"]))
            bucket["accuracy"] = float(bucket["hits"]) / float(total)
            bucket["exact_accuracy"] = float(bucket["exact"]) / float(total)
    return by_mode_family


def _mode_accuracy(generation: dict[str, Any], mode: str) -> float:
    item = generation.get(mode) or {}
    return float(item.get("accuracy", 0.0))


def build_gate_report(
    *,
    soft_prefix_report: dict[str, Any],
    generation_rows: list[dict[str, Any]],
    train_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    out_dir: str | Path,
    min_full_accuracy: float,
    min_donor_margin: float,
    min_core_off_margin: float,
    min_state_off_margin: float,
    min_eval_families: int,
    require_family_full_hit: bool,
) -> dict[str, Any]:
    generation = soft_prefix_report.get("generation") or {}
    full = _mode_accuracy(generation, "soft_full_no_evidence")
    donor = _mode_accuracy(generation, "donor_only_no_evidence")
    core_off = _mode_accuracy(generation, "soft_core_off_no_evidence")
    state_off = _mode_accuracy(generation, "soft_state_off_no_evidence")
    by_mode_family = summarize_generation_by_family(generation_rows)
    full_by_family = by_mode_family.get("soft_full_no_evidence", {})
    family_count = len(eval_summary.get("families") or [])

    reject_reasons: list[str] = []
    if family_count < int(min_eval_families):
        reject_reasons.append("eval_family_count_below_min")
    if full < float(min_full_accuracy):
        reject_reasons.append("full_generation_accuracy_below_min")
    if full - donor <= float(min_donor_margin):
        reject_reasons.append("full_does_not_beat_donor")
    if full - core_off <= float(min_core_off_margin):
        reject_reasons.append("full_does_not_beat_core_off")
    state_dim = int(
        ((soft_prefix_report.get("adapter") or {}).get("state_dim") or 0)
    )
    if state_dim > 0 and full - state_off <= float(min_state_off_margin):
        reject_reasons.append("full_does_not_beat_state_off")
    if bool(require_family_full_hit):
        missed = [
            family
            for family in (eval_summary.get("families") or [])
            if int((full_by_family.get(family) or {}).get("hits", 0)) <= 0
        ]
        if missed:
            reject_reasons.append("full_has_zero_hit_family")
    accepted = not reject_reasons
    report = {
        "status": "complete",
        "target_level": "L2 local gate / L3 candidate",
        "major_bottleneck": (
            "recursive core + state codec + autoregressive final answer path "
            "must beat donor-only on a mixed small reasoning gate"
        ),
        "decision": (
            "accepted_l3_candidate_small_general_reasoning"
            if accepted
            else "rejected"
        ),
        "accepted": accepted,
        "reject_reasons": reject_reasons,
        "metrics": {
            "full_generation_accuracy": full,
            "donor_generation_accuracy": donor,
            "core_off_generation_accuracy": core_off,
            "state_off_generation_accuracy": state_off,
            "full_minus_donor": full - donor,
            "full_minus_core_off": full - core_off,
            "full_minus_state_off": full - state_off,
            "eval_family_count": family_count,
        },
        "thresholds": {
            "min_full_accuracy": float(min_full_accuracy),
            "min_donor_margin": float(min_donor_margin),
            "min_core_off_margin": float(min_core_off_margin),
            "min_state_off_margin": float(min_state_off_margin),
            "min_eval_families": int(min_eval_families),
            "require_family_full_hit": bool(require_family_full_hit),
        },
        "train_summary": train_summary,
        "eval_summary": eval_summary,
        "by_mode_family": by_mode_family,
        "soft_prefix_report": soft_prefix_report,
        "next_action": (
            "promote to a broader held-out universal LLM causal-path gate"
            if accepted
            else (
                "fix the first failing axis: state codec if state_off ties/full loses; "
                "renderer if donor/core_off tie; data schedule if train split was under-covered"
            )
        ),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_soft_prefix_command(args: argparse.Namespace, train_jsonl: Path, eval_jsonl: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/304_train_core_soft_prefix_donor.py",
        "--config",
        str(args.config),
        "--checkpoint",
        str(args.checkpoint),
        "--train-cases",
        str(train_jsonl),
        "--eval-cases",
        str(eval_jsonl),
        "--out-dir",
        str(Path(args.out_dir) / "soft_prefix"),
        "--max-train-cases",
        str(args.max_train_cases),
        "--max-eval-cases",
        str(args.max_eval_cases),
        "--max-length",
        str(args.max_length),
        "--max-target-tokens",
        str(args.max_target_tokens),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--core-steps",
        str(args.core_steps),
        "--prefix-tokens",
        str(args.prefix_tokens),
        "--rank",
        str(args.rank),
        "--scale",
        str(args.scale),
        "--steps",
        str(args.soft_prefix_steps),
        "--lr",
        str(args.lr),
        "--scheduled-sampling-prob",
        str(args.scheduled_sampling_prob),
        "--scheduled-sampling-warmup-steps",
        str(args.scheduled_sampling_warmup_steps),
        "--state-logits-key",
        str(args.state_logits_key),
        "--state-feature-mode",
        str(args.state_feature_mode),
        "--device",
        str(args.device),
        "--log-every",
        str(args.log_every),
    ]
    if bool(args.append_eos_target):
        command.append("--append-eos-target")
    if bool(args.suppress_visible_reasoning_tokens):
        command.append("--suppress-visible-reasoning-tokens")
    return command


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small mixed reasoning gate where recursive core + state codec "
            "+ donor autoregressive answer path must beat donor-only and ablations."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-source", action="append", default=None)
    parser.add_argument("--eval-source", action="append", default=None)
    parser.add_argument("--max-train-per-source", type=int, default=8)
    parser.add_argument("--max-eval-per-source", type=int, default=4)
    parser.add_argument("--max-train-cases", type=int, default=16)
    parser.add_argument("--max-eval-cases", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-target-tokens", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--append-eos-target", action="store_true")
    parser.add_argument("--core-steps", type=int, default=8)
    parser.add_argument("--prefix-tokens", type=int, default=4)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--soft-prefix-steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--scheduled-sampling-prob", type=float, default=0.0)
    parser.add_argument("--scheduled-sampling-warmup-steps", type=int, default=0)
    parser.add_argument("--state-logits-key", default=DEFAULT_STATE_KEYS)
    parser.add_argument(
        "--state-feature-mode",
        choices=["softmax", "argmax_onehot", "logits"],
        default="softmax",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--suppress-visible-reasoning-tokens", action="store_true")
    parser.add_argument("--min-full-accuracy", type=float, default=0.25)
    parser.add_argument("--min-donor-margin", type=float, default=0.0)
    parser.add_argument("--min-core-off-margin", type=float, default=0.0)
    parser.add_argument("--min-state-off-margin", type=float, default=0.0)
    parser.add_argument("--min-eval-families", type=int, default=2)
    parser.add_argument("--no-require-family-full-hit", action="store_true")
    parser.add_argument(
        "--skip-soft-prefix-run",
        action="store_true",
        help="Reuse an existing soft_prefix/report.json and generation.jsonl.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root = repo_root()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_sources = tuple(args.train_source or DEFAULT_TRAIN_SOURCES)
    eval_sources = tuple(args.eval_source or DEFAULT_EVAL_SOURCES)
    train_rows, train_summary = build_mixed_gate_cases(
        sources=train_sources,
        max_per_source=int(args.max_train_per_source),
    )
    eval_rows, eval_summary = build_mixed_gate_cases(
        sources=eval_sources,
        max_per_source=int(args.max_eval_per_source),
    )
    train_jsonl = out_dir / "small_general_train.jsonl"
    eval_jsonl = out_dir / "small_general_eval.jsonl"
    write_jsonl(train_jsonl, train_rows)
    write_jsonl(eval_jsonl, eval_rows)
    summary = {
        "train": train_summary,
        "eval": eval_summary,
        "target_level": "L2 local gate / L3 candidate",
        "required_path": (
            "prompt -> tokenizer -> frozen donor hidden states -> recursive core "
            "-> state codec -> donor soft-prefix -> autoregressive answer"
        ),
    }
    (out_dir / "case_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    soft_prefix_dir = out_dir / "soft_prefix"
    if not bool(args.skip_soft_prefix_run):
        command = build_soft_prefix_command(args, train_jsonl, eval_jsonl)
        print(
            json.dumps(
                {
                    "target_level": "L2 local gate / L3 candidate",
                    "major_bottleneck": "small general reasoning causal answer path",
                    "baseline_to_beat": "donor_only_no_evidence",
                    "required_ablations": [
                        "soft_core_off_no_evidence",
                        "soft_state_off_no_evidence",
                    ],
                    "command": command,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        subprocess.run(command, cwd=root, check=True)

    soft_report_path = soft_prefix_dir / "report.json"
    generation_path = soft_prefix_dir / "generation.jsonl"
    if not soft_report_path.exists():
        raise FileNotFoundError(f"missing soft-prefix report: {soft_report_path}")
    if not generation_path.exists():
        raise FileNotFoundError(f"missing soft-prefix generation: {generation_path}")
    soft_report = json.loads(soft_report_path.read_text(encoding="utf-8"))
    generation_rows = _load_generation_rows(generation_path)
    report = build_gate_report(
        soft_prefix_report=soft_report,
        generation_rows=generation_rows,
        train_summary=train_summary,
        eval_summary=eval_summary,
        out_dir=out_dir,
        min_full_accuracy=float(args.min_full_accuracy),
        min_donor_margin=float(args.min_donor_margin),
        min_core_off_margin=float(args.min_core_off_margin),
        min_state_off_margin=float(args.min_state_off_margin),
        min_eval_families=int(args.min_eval_families),
        require_family_full_hit=not bool(args.no_require_family_full_hit),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
