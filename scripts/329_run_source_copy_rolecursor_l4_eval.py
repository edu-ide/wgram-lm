#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
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
DEFAULT_CASES = "data/eval/qtrm_source_copy_lexicalization_eval128.jsonl"
DEFAULT_OUT_DIR = (
    "/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/"
    "qtrm_source_copy_rolecursor_l4_eval"
)


def _load_l4_runner() -> Any:
    path = Path(__file__).resolve().with_name("322_run_source_pointer_l4_lm_path_gate.py")
    spec = importlib.util.spec_from_file_location("source_pointer_l4_lm_path_gate", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"failed to load L4 runner: {path}")
    spec.loader.exec_module(module)
    return module


_runner = _load_l4_runner()

FULL_MODE = _runner.FULL_MODE
DONOR_MODE = _runner.DONOR_MODE
CORE_OFF_MODE = _runner.CORE_OFF_MODE
PRIMITIVE_OFF_MODE = _runner.PRIMITIVE_OFF_MODE
SOURCE_SLOT_OFF_MODE = _runner.SOURCE_SLOT_OFF_MODE
SOURCE_BINDER_OFF_MODE = _runner.SOURCE_BINDER_OFF_MODE
VOCAB_RENDERER_OFF_MODE = _runner.VOCAB_RENDERER_OFF_MODE

DEFAULT_MODES = [
    DONOR_MODE,
    CORE_OFF_MODE,
    FULL_MODE,
    PRIMITIVE_OFF_MODE,
    SOURCE_SLOT_OFF_MODE,
    SOURCE_BINDER_OFF_MODE,
    VOCAB_RENDERER_OFF_MODE,
]


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
        rows = load_jsonl(out)
    except (OSError, json.JSONDecodeError):
        return False
    return len(rows) == int(expected_rows)


def chunk_rows(
    rows: list[dict[str, Any]],
    *,
    chunk_size: int,
) -> Iterable[list[dict[str, Any]]]:
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(rows), chunk_size):
        yield rows[start : start + chunk_size]


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
        "--token-numeric-source-slots",
        "--token-numeric-source-slot-vocab-size",
        str(int(args.token_numeric_source_slot_vocab_size)),
        "--token-numeric-source-slot-max-slots",
        str(int(args.token_numeric_source_slot_max_slots)),
        "--token-numeric-source-slot-gate-min",
        str(float(args.token_numeric_source_slot_gate_min)),
        "--token-numeric-source-slot-predicate-feedback",
        "--token-numeric-source-slot-predicate-gate-min",
        str(float(args.token_numeric_source_slot_predicate_gate_min)),
        "--core-source-position-binder",
        "--core-source-position-binder-gate-min",
        str(float(args.core_source_position_binder_gate_min)),
        "--core-source-position-binder-state-gate-min",
        str(float(args.core_source_position_binder_state_gate_min)),
        "--core-source-position-binder-state-st",
        "--core-source-position-binder-source-slots-only",
        "--core-source-position-binder-raw-source-slots",
        "--mode",
        str(mode),
    ]


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


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> int:
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


def build_report(
    rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    commands: list[dict[str, Any]],
    exit_codes: list[dict[str, Any]],
    min_full_accuracy: float,
    min_donor_margin: float,
    min_core_off_margin: float,
    min_primitive_drop: float,
    min_source_slot_drop: float,
    min_source_binder_drop: float,
    min_vocab_renderer_drop: float,
) -> dict[str, Any]:
    summary = _runner.summarize_generation(rows)
    decision = _runner.build_decision(
        summary=summary,
        min_full_accuracy=float(min_full_accuracy),
        min_donor_margin=float(min_donor_margin),
        min_core_off_margin=float(min_core_off_margin),
        min_primitive_drop=float(min_primitive_drop),
        min_source_slot_drop=float(min_source_slot_drop),
        min_source_binder_drop=float(min_source_binder_drop),
        min_bridge_drop=0.0,
        min_vocab_renderer_drop=float(min_vocab_renderer_drop),
        min_answer_recurrent_drop=0.0,
        min_answer_halt_gate_drop=0.0,
        min_answer_next_token_decoder_drop=0.0,
    )
    return {
        **decision,
        "gate": "source_copy_rolecursor_l4_eval",
        "out_dir": str(out_dir),
        "rows": len(rows),
        "generation_summary": summary,
        "commands": commands,
        "exit_codes": exit_codes,
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
        "gate",
        "out_dir",
        "rows",
        "generation_jsonl",
        "report_path",
        "checkpoint",
        "config",
        "cases",
        "chunk_size",
        "max_cases",
    ]
    return {key: report[key] for key in keys if key in report}


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.cases, max_rows=int(args.max_cases))
    if not rows:
        raise ValueError(f"no rows loaded from {args.cases}")
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_entries: list[tuple[Path, int]] = []
    for chunk_index, chunk in enumerate(
        chunk_rows(rows, chunk_size=int(args.chunk_size))
    ):
        chunk_path = chunks_dir / f"cases_{chunk_index:04d}.jsonl"
        write_jsonl(chunk_path, chunk)
        chunk_entries.append((chunk_path, len(chunk)))

    modes = list(args.mode or DEFAULT_MODES)
    commands: list[dict[str, Any]] = []
    exit_codes: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    env = command_env(args)
    if bool(args.dry_run):
        for mode in modes:
            for chunk_index, (chunk_path, _chunk_len) in enumerate(chunk_entries):
                out_path = chunks_dir / f"{mode}_{chunk_index:04d}.jsonl"
                commands.append(
                    {
                        "mode": mode,
                        "chunk_index": chunk_index,
                        "command": eval_command(
                            args,
                            mode=mode,
                            cases_path=chunk_path,
                            out_path=out_path,
                        ),
                    }
                )
        report = {
            "gate": "source_copy_rolecursor_l4_eval",
            "decision": "dry_run",
            "accepted": False,
            "commands": commands,
            "chunk_count": len(chunk_entries),
            "modes": modes,
            "out_dir": str(out_dir),
        }
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    for mode in modes:
        for chunk_index, (chunk_path, chunk_len) in enumerate(chunk_entries):
            out_path = chunks_dir / f"{mode}_{chunk_index:04d}.jsonl"
            command = eval_command(
                args,
                mode=mode,
                cases_path=chunk_path,
                out_path=out_path,
            )
            commands.append(
                {
                    "mode": mode,
                    "chunk_index": chunk_index,
                    "command": command,
                    "out": str(out_path),
                }
            )
            resumed = bool(args.resume) and output_is_complete(
                out_path,
                expected_rows=chunk_len,
            )
            if resumed:
                print(
                    f"[skip] mode={mode} chunk={chunk_index:04d} rows={chunk_len}",
                    flush=True,
                )
                exit_code = 0
            else:
                print(
                    f"[run] mode={mode} chunk={chunk_index:04d} rows={chunk_len}",
                    flush=True,
                )
                exit_code = run_command(
                    command,
                    cwd=root,
                    env=env,
                    stdout_path=chunks_dir / f"{mode}_{chunk_index:04d}.stdout.log",
                    stderr_path=chunks_dir / f"{mode}_{chunk_index:04d}.stderr.log",
                )
                print(
                    f"[done] mode={mode} chunk={chunk_index:04d} exit={exit_code}",
                    flush=True,
                )
            exit_codes.append(
                {
                    "mode": mode,
                    "chunk_index": chunk_index,
                    "exit_code": exit_code,
                    "resumed": resumed,
                }
            )
            if exit_code != 0:
                report = {
                    "gate": "source_copy_rolecursor_l4_eval",
                    "decision": "eval_failed",
                    "accepted": False,
                    "failed_mode": mode,
                    "failed_chunk_index": chunk_index,
                    "commands": commands,
                    "exit_codes": exit_codes,
                    "out_dir": str(out_dir),
                }
                (out_dir / "report.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                return report
            combined_rows.extend(load_jsonl(out_path))

    combined_path = out_dir / "generation_eval.jsonl"
    write_jsonl(combined_path, combined_rows)
    report = build_report(
        combined_rows,
        out_dir=out_dir,
        commands=commands,
        exit_codes=exit_codes,
        min_full_accuracy=float(args.min_full_accuracy),
        min_donor_margin=float(args.min_donor_margin),
        min_core_off_margin=float(args.min_core_off_margin),
        min_primitive_drop=float(args.min_primitive_drop),
        min_source_slot_drop=float(args.min_source_slot_drop),
        min_source_binder_drop=float(args.min_source_binder_drop),
        min_vocab_renderer_drop=float(args.min_vocab_renderer_drop),
    )
    report["generation_jsonl"] = str(combined_path)
    report["checkpoint"] = str(args.checkpoint)
    report["config"] = str(args.config)
    report["cases"] = str(args.cases)
    report["chunk_size"] = int(args.chunk_size)
    report["max_cases"] = int(args.max_cases)
    report["report_path"] = str(out_dir / "report.json")
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the source-copy answer-role-cursor L4 generation gate in "
            "small chunks so GPU memory is released between modes/cases."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--mode", action="append", default=None)
    parser.add_argument("--max-cases", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--token-numeric-source-slot-vocab-size", type=int, default=128)
    parser.add_argument("--token-numeric-source-slot-max-slots", type=int, default=5)
    parser.add_argument("--token-numeric-source-slot-gate-min", type=float, default=1.0)
    parser.add_argument(
        "--token-numeric-source-slot-predicate-gate-min",
        type=float,
        default=1.0,
    )
    parser.add_argument("--core-source-position-binder-gate-min", type=float, default=1.0)
    parser.add_argument(
        "--core-source-position-binder-state-gate-min",
        type=float,
        default=1.0,
    )
    parser.add_argument("--min-full-accuracy", type=float, default=0.70)
    parser.add_argument("--min-donor-margin", type=float, default=0.05)
    parser.add_argument("--min-core-off-margin", type=float, default=0.05)
    parser.add_argument("--min-primitive-drop", type=float, default=0.05)
    parser.add_argument("--min-source-slot-drop", type=float, default=0.05)
    parser.add_argument("--min-source-binder-drop", type=float, default=0.05)
    parser.add_argument("--min-vocab-renderer-drop", type=float, default=0.05)
    parser.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME", "/mnt/nvme1n1p2/hf-cache-qtrm"),
    )
    parser.add_argument(
        "--tmpdir",
        default=os.environ.get("TMPDIR", "/mnt/nvme1n1p2/tmp"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed chunk outputs in out-dir instead of recomputing them.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    report = run_eval(build_arg_parser().parse_args())
    print(json.dumps(compact_stdout_report(report), ensure_ascii=False, indent=2))
    return 1 if report.get("decision") == "eval_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
