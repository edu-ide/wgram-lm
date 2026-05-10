#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


DEFAULT_CONFIG = "configs/qwen35_2b_4090_source_copy_pointer_renderer_scaffold.yaml"
DEFAULT_CHECKPOINT = (
    "/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/"
    "qtrm_source_copy_state_ce_s040_fix/last.pt"
)
DEFAULT_CASES = (
    "data/eval/"
    "pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_"
    "len1113_probe_eval60000_v6to7_len11_13_mixed_only.jsonl"
)
DEFAULT_OUT_DIR = (
    "/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/"
    "qtrm_mixed_noncopy_lm_gate"
)

DONOR_MODE = "donor_only_no_evidence"
CORE_OFF_MODE = "qtrm_core_off_no_evidence"
FULL_MODE = "qtrm_core_steps_8_no_evidence"
DEFAULT_MODES = [DONOR_MODE, CORE_OFF_MODE, FULL_MODE]


def load_jsonl(path: str | Path, *, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if int(max_rows) > 0 and len(rows) >= int(max_rows):
                break
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def output_is_complete(path: str | Path, *, expected_rows: int) -> bool:
    out = Path(path)
    if not out.exists():
        return False
    try:
        return len(load_jsonl(out)) == int(expected_rows)
    except (OSError, json.JSONDecodeError):
        return False


def chunk_rows(rows: list[dict[str, Any]], *, chunk_size: int) -> Iterable[list[dict[str, Any]]]:
    size = max(1, int(chunk_size))
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def command_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HOME"] = str(args.hf_home)
    env["TMPDIR"] = str(args.tmpdir)
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        "src" if not current_pythonpath else f"src{os.pathsep}{current_pythonpath}"
    )
    return env


def eval_command(
    args: argparse.Namespace,
    *,
    mode: str,
    cases_path: Path,
    out_path: Path,
) -> list[str]:
    return [
        args.python_bin,
        "scripts/192_eval_raw_intelligence.py",
        "--config",
        str(args.config),
        "--checkpoint",
        str(args.checkpoint),
        "--cases",
        str(cases_path),
        "--out",
        str(out_path),
        "--max-cases",
        str(int(args.chunk_size)),
        "--max-length",
        str(int(args.max_length)),
        "--max-new-tokens",
        str(int(args.max_new_tokens)),
        "--scoring",
        "generation",
        "--choice-score-normalization",
        "mean",
        "--suppress-visible-reasoning-tokens",
        "--no-repeat-ngram-size",
        str(int(args.no_repeat_ngram_size)),
        "--mode",
        str(mode),
    ]


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return int(completed.returncode)


def summarize_generation(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_mode: dict[str, dict[str, Any]] = {}
    by_family: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        mode = str(row.get("mode") or "unknown")
        hit = bool(row.get("hit") or row.get("exact_match") or row.get("normalized_exact"))
        bucket = by_mode.setdefault(mode, {"hits": 0, "total": 0})
        bucket["hits"] += int(hit)
        bucket["total"] += 1
        family = str(row.get("task_family") or row.get("category") or "unknown")
        fam_bucket = by_family.setdefault(mode, {}).setdefault(
            family,
            {"hits": 0, "total": 0},
        )
        fam_bucket["hits"] += int(hit)
        fam_bucket["total"] += 1
    for mode, bucket in by_mode.items():
        total = max(1, int(bucket["total"]))
        bucket["accuracy"] = float(bucket["hits"]) / float(total)
        bucket["by_family"] = {}
        for family, fam_bucket in by_family.get(mode, {}).items():
            fam_total = max(1, int(fam_bucket["total"]))
            bucket["by_family"][family] = {
                **fam_bucket,
                "accuracy": float(fam_bucket["hits"]) / float(fam_total),
            }
    return dict(sorted(by_mode.items()))


def mode_accuracy(summary: dict[str, dict[str, Any]], mode: str) -> float:
    return float((summary.get(mode) or {}).get("accuracy", 0.0))


def build_report(
    rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    commands: list[dict[str, Any]],
    exit_codes: list[dict[str, Any]],
    min_full_accuracy: float,
    min_donor_margin: float,
    min_core_off_margin: float,
) -> dict[str, Any]:
    summary = summarize_generation(rows)
    full = mode_accuracy(summary, FULL_MODE)
    donor = mode_accuracy(summary, DONOR_MODE)
    core_off = mode_accuracy(summary, CORE_OFF_MODE)
    reject_reasons: list[str] = []
    if full < float(min_full_accuracy):
        reject_reasons.append("full_generation_accuracy_below_min")
    if full - donor <= float(min_donor_margin):
        reject_reasons.append("full_does_not_beat_donor")
    if full - core_off <= float(min_core_off_margin):
        reject_reasons.append("full_does_not_beat_core_off")
    failed_commands = [item for item in exit_codes if int(item.get("exit_code", 0)) != 0]
    if failed_commands:
        reject_reasons.append("eval_command_failed")
    accepted = not reject_reasons
    return {
        "decision": "accepted_mixed_noncopy_lm_gate" if accepted else "rejected_noncopy_lm_gate",
        "accepted": accepted,
        "target_level": "L4 mixed non-copy LM gate",
        "major_bottleneck": "non-copy latent-state-to-autoregressive answer synthesis",
        "out_dir": str(out_dir),
        "rows": len(rows),
        "generation_summary": summary,
        "reject_reasons": reject_reasons,
        "decisive_metrics": {
            "full_generation_accuracy": full,
            "donor_generation_accuracy": donor,
            "core_off_generation_accuracy": core_off,
            "full_minus_donor": full - donor,
            "full_minus_core_off": full - core_off,
        },
        "thresholds": {
            "min_full_accuracy": float(min_full_accuracy),
            "min_donor_margin": float(min_donor_margin),
            "min_core_off_margin": float(min_core_off_margin),
        },
        "commands": commands,
        "exit_codes": exit_codes,
        "next_action": (
            "promote to broader mixed-family non-copy eval and add ablations"
            if accepted
            else "redesign non-copy answer synthesis; source-copy lexicalization is insufficient"
        ),
    }


def compact_stdout_report(report: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "decision",
        "accepted",
        "target_level",
        "major_bottleneck",
        "reject_reasons",
        "decisive_metrics",
        "thresholds",
        "next_action",
        "generation_jsonl",
        "report_path",
        "config",
        "checkpoint",
        "cases",
        "max_cases",
        "chunk_size",
    ]
    return {key: report[key] for key in keys if key in report}


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.cases, max_rows=int(args.max_cases))
    if not rows:
        raise ValueError(f"no rows loaded from {args.cases}")

    env = command_env(args)
    commands: list[dict[str, Any]] = []
    exit_codes: list[dict[str, Any]] = []
    output_parts: list[Path] = []
    expected_total = 0
    for chunk_index, chunk in enumerate(chunk_rows(rows, chunk_size=int(args.chunk_size))):
        chunk_cases = out_dir / "chunks" / f"cases_{chunk_index:04d}.jsonl"
        write_jsonl(chunk_cases, chunk)
        for mode in args.mode:
            out_part = out_dir / "chunks" / f"eval_{chunk_index:04d}_{mode}.jsonl"
            expected_total += len(chunk)
            command = eval_command(
                args,
                mode=mode,
                cases_path=chunk_cases,
                out_path=out_part,
            )
            commands.append(
                {
                    "chunk": chunk_index,
                    "mode": mode,
                    "command": command,
                    "out": str(out_part),
                }
            )
            if bool(args.resume) and output_is_complete(out_part, expected_rows=len(chunk)):
                exit_code = 0
            else:
                exit_code = run_command(
                    command,
                    cwd=root,
                    env=env,
                    stdout_path=out_dir / "logs" / f"eval_{chunk_index:04d}_{mode}.stdout.log",
                    stderr_path=out_dir / "logs" / f"eval_{chunk_index:04d}_{mode}.stderr.log",
                )
            exit_codes.append({"chunk": chunk_index, "mode": mode, "exit_code": exit_code})
            output_parts.append(out_part)
            print(
                json.dumps(
                    {
                        "chunk": chunk_index,
                        "mode": mode,
                        "exit_code": exit_code,
                        "out": str(out_part),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    eval_rows: list[dict[str, Any]] = []
    for part in output_parts:
        if part.exists():
            eval_rows.extend(load_jsonl(part))
    generation_jsonl = out_dir / "generation.jsonl"
    write_jsonl(generation_jsonl, eval_rows)

    report = build_report(
        eval_rows,
        out_dir=out_dir,
        commands=commands,
        exit_codes=exit_codes,
        min_full_accuracy=float(args.min_full_accuracy),
        min_donor_margin=float(args.min_donor_margin),
        min_core_off_margin=float(args.min_core_off_margin),
    )
    report.update(
        {
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "cases": str(args.cases),
            "max_cases": int(args.max_cases),
            "chunk_size": int(args.chunk_size),
            "modes": list(args.mode),
            "expected_rows": int(expected_total),
            "generation_jsonl": str(generation_jsonl),
        }
    )
    report_path = out_dir / "report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether the current canonical LM path solves mixed-family "
            "non-copy reasoning. This is a gate for the post source-copy L4 bottleneck."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--hf-home", default="/mnt/nvme1n1p2/hf-cache-qtrm")
    parser.add_argument("--tmpdir", default="/mnt/nvme0n1p2/tmp")
    parser.add_argument("--max-cases", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--min-full-accuracy", type=float, default=0.10)
    parser.add_argument("--min-donor-margin", type=float, default=0.01)
    parser.add_argument("--min-core-off-margin", type=float, default=0.01)
    parser.add_argument("--mode", action="append", default=list(DEFAULT_MODES))
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = run_eval(args)
    print(json.dumps(compact_stdout_report(report), ensure_ascii=False, indent=2))
    return 0 if not any(item.get("exit_code") for item in report["exit_codes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
