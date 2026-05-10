#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Any


DEFAULT_CONFIG = (
    "configs/qwen35_2b_4090_list_order_source_pointer_internal_binder_"
    "forced_gate_roles12_s300.yaml"
)
DEFAULT_CHECKPOINT = (
    "/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/"
    "qtrm_l2_source_pointer_roles12_targetfix_s120/"
    "accepted_l2_source_pointer_roles12_step_000040.pt"
)
DEFAULT_DATA_JSONL = "data/eval/qtrm_source_pointer_l3_hard_eval128.jsonl"
DEFAULT_DATA_SUMMARY = "data/eval/qtrm_source_pointer_l3_hard_eval128.summary.json"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_absolute_ordered_builder(root: Path | None = None) -> Any:
    root = root or repo_root()
    script = root / "scripts" / "317_build_absolute_ordered_state_gate_data.py"
    spec = importlib.util.spec_from_file_location("absolute_ordered_state_builder", script)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"failed to load data builder: {script}")
    spec.loader.exec_module(module)
    return module


def _prompt_for_question(question: str) -> str:
    return (
        "Answer with only the final answer. Do not write reasoning.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _retarget_prompt(row: dict[str, Any], question: str) -> dict[str, Any]:
    row = dict(row)
    row["question"] = question
    row["prompt"] = _prompt_for_question(question)
    return row


def _make_row(
    builder: Any,
    *,
    case_prefix: str,
    variant: str,
    index: int,
    values: list[int],
    question: str | None = None,
) -> dict[str, Any]:
    row = builder._make_case(  # noqa: SLF001 - reuse the canonical gate row schema.
        case_id=f"{case_prefix}-{variant}-{index:04d}",
        values=[int(value) for value in values],
    )
    if question is not None:
        row = _retarget_prompt(row, question)
    row["hard_variant"] = variant
    row["role_value_list_class_mode"] = "source_position"
    row["role_value_supervise_null_slots"] = True
    row["expected_paradigm"] = "latent_recurrent_source_pointer"
    row["l3_hard_split"] = True
    return row


def _random_values_with_even_count(
    rng: random.Random,
    *,
    low: int,
    high: int,
    length: int,
    min_even: int,
) -> list[int]:
    values = [rng.randrange(int(low), int(high)) for _ in range(int(length))]
    even_candidates = [value for value in range(int(low), int(high)) if value % 2 == 0]
    if not even_candidates:
        raise ValueError("hard range must contain at least one even value")
    while sum(1 for value in values if value % 2 == 0) < int(min_even):
        odd_indices = [idx for idx, value in enumerate(values) if value % 2 != 0]
        replace_at = odd_indices[0] if odd_indices else rng.randrange(len(values))
        values[replace_at] = even_candidates[rng.randrange(len(even_candidates))]
    return values


def build_l3_hard_rows(
    *,
    count_per_variant: int = 32,
    seed: int = 321,
) -> list[dict[str, Any]]:
    if int(count_per_variant) <= 0:
        raise ValueError("count_per_variant must be positive")
    builder = load_absolute_ordered_builder()
    rng = random.Random(int(seed))
    rows: list[dict[str, Any]] = []
    case_prefix = f"source-pointer-l3-s{int(seed)}"
    variants = (
        "range_shift_v32to63",
        "fifth_position_single_even",
        "duplicate_even_binding",
        "surface_paraphrase",
    )
    for index in range(int(count_per_variant)):
        rows.append(
            _make_row(
                builder,
                case_prefix=case_prefix,
                variant="range_shift_v32to63",
                index=index,
                values=_random_values_with_even_count(
                    rng,
                    low=32,
                    high=64,
                    length=5,
                    min_even=2,
                ),
            )
        )

        odd_values = [value for value in range(1, 32, 2)]
        even_values = [value for value in range(0, 32, 2)]
        first_four = [odd_values[rng.randrange(len(odd_values))] for _ in range(4)]
        values = first_four + [even_values[rng.randrange(len(even_values))]]
        rows.append(
            _make_row(
                builder,
                case_prefix=case_prefix,
                variant="fifth_position_single_even",
                index=index,
                values=values,
            )
        )

        duplicate_even = even_values[rng.randrange(len(even_values))]
        values = [
            duplicate_even,
            odd_values[rng.randrange(len(odd_values))],
            duplicate_even,
            odd_values[rng.randrange(len(odd_values))],
            even_values[rng.randrange(len(even_values))],
        ]
        rows.append(
            _make_row(
                builder,
                case_prefix=case_prefix,
                variant="duplicate_even_binding",
                index=index,
                values=values,
            )
        )

        values = _random_values_with_even_count(
            rng,
            low=0,
            high=32,
            length=5,
            min_even=2,
        )
        question = (
            f"Given this sequence {values}, select the even integers, multiply "
            "each selected integer by two, and output the results as "
            "comma-separated values with no spaces. Output EMPTY if no integer "
            "qualifies."
        )
        rows.append(
            _make_row(
                builder,
                case_prefix=case_prefix,
                variant="surface_paraphrase",
                index=index,
                values=values,
                question=question,
            )
        )
    seen_variants = {row["hard_variant"] for row in rows}
    missing = sorted(set(variants) - seen_variants)
    if missing:
        raise AssertionError(f"missing L3 variants: {missing}")
    return rows


def write_l3_hard_rows(
    *,
    data_jsonl: str | Path = DEFAULT_DATA_JSONL,
    summary_json: str | Path = DEFAULT_DATA_SUMMARY,
    count_per_variant: int = 32,
    seed: int = 321,
) -> dict[str, Any]:
    rows = build_l3_hard_rows(
        count_per_variant=int(count_per_variant),
        seed=int(seed),
    )
    out = Path(data_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_variant: dict[str, int] = {}
    for row in rows:
        by_variant[str(row["hard_variant"])] = by_variant.get(str(row["hard_variant"]), 0) + 1
    summary = {
        "split_type": "source_pointer_l3_hard_perturbation",
        "target_level": "L3 major bottleneck candidate",
        "rows": len(rows),
        "count_per_variant": int(count_per_variant),
        "seed": int(seed),
        "by_variant": dict(sorted(by_variant.items())),
        "variants": [
            "range_shift_v32to63",
            "fifth_position_single_even",
            "duplicate_even_binding",
            "surface_paraphrase",
        ],
    }
    summary_path = Path(summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def python_bin(args: argparse.Namespace) -> str:
    return str(args.python_bin or sys.executable)


def eval_command(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    out_json: Path,
    primitive_off: bool = False,
    token_numeric_off: bool = False,
    source_binder_off: bool = False,
) -> list[str]:
    command = [
        python_bin(args),
        "scripts/238_eval_qtrm_algorithmic_value_state.py",
        "--config",
        str(args.config),
        "--checkpoint",
        str(checkpoint),
        "--data-jsonl",
        str(args.data_jsonl),
        "--out-json",
        str(out_json),
        "--tokenizer-model-id",
        str(args.tokenizer_model_id),
        "--core-steps",
        str(int(args.core_steps)),
        "--max-cases",
        str(int(args.max_eval_cases)),
        "--include-family",
        "list_transform",
        "--use-role-value-state",
        "--use-core-primitive-role-value-state",
        "--role-value-list-class-mode",
        "source_position",
        "--core-source-position-binder",
        "--core-source-position-binder-gate-min",
        str(float(args.core_source_position_binder_gate_min)),
        "--core-source-position-binder-state-gate-min",
        str(float(args.core_source_position_binder_state_gate_min)),
    ]
    if bool(args.token_numeric_source_slots):
        command.extend(
            [
                "--token-numeric-source-slots",
                "--token-numeric-source-slot-vocab-size",
                str(int(args.token_numeric_source_slot_vocab_size)),
                "--token-numeric-source-slot-max-slots",
                str(int(args.token_numeric_source_slot_max_slots)),
                "--token-numeric-source-slot-gate-min",
                str(float(args.token_numeric_source_slot_gate_min)),
            ]
        )
        if bool(args.token_numeric_source_slot_predicate_feedback):
            command.extend(
                [
                    "--token-numeric-source-slot-predicate-feedback",
                    "--token-numeric-source-slot-predicate-gate-min",
                    str(float(args.token_numeric_source_slot_predicate_gate_min)),
                ]
            )
    else:
        command.extend(
            [
                "--token-numeric-value-features",
                "--token-numeric-value-vocab-size",
                str(int(args.token_numeric_value_vocab_size)),
            ]
        )
    if bool(args.core_source_position_binder_state_st):
        command.append("--core-source-position-binder-state-st")
    if bool(args.core_source_position_binder_source_slots_only):
        command.append("--core-source-position-binder-source-slots-only")
    if bool(args.core_source_position_binder_raw_source_slots):
        command.append("--core-source-position-binder-raw-source-slots")
    if bool(primitive_off):
        command.append("--disable-core-primitive-role-value-executor")
    if bool(token_numeric_off):
        command.append(
            "--disable-token-numeric-source-slots"
            if bool(args.token_numeric_source_slots)
            else "--disable-token-numeric-value-features"
        )
    if bool(source_binder_off):
        command.append("--disable-core-source-position-binder")
    return command


def run_command(
    command: list[str],
    *,
    out_path: Path,
    env: dict[str, str],
    cwd: Path,
) -> int:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    out_path.with_suffix(".stdout.log").write_text(completed.stdout, encoding="utf-8")
    out_path.with_suffix(".stderr.log").write_text(completed.stderr, encoding="utf-8")
    return int(completed.returncode)


def load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _summarize_role_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_values = sum(int(record.get("total_values", 0)) for record in records)
    correct_values = sum(int(record.get("correct_values", 0)) for record in records)
    total_steps = sum(int(record.get("total_steps", 0)) for record in records)
    exact_steps = sum(int(record.get("exact_steps", 0)) for record in records)
    exact_rows = sum(int(bool(record.get("trace_exact"))) for record in records)
    return {
        "rows": len(records),
        "exact_rows": exact_rows,
        "trace_exact_accuracy": float(exact_rows) / float(len(records)) if records else 0.0,
        "correct_values": correct_values,
        "total_values": total_values,
        "value_accuracy": float(correct_values) / float(total_values)
        if total_values
        else 0.0,
        "exact_steps": exact_steps,
        "total_steps": total_steps,
        "step_exact_accuracy": float(exact_steps) / float(total_steps)
        if total_steps
        else 0.0,
    }


def summarize_by_variant(
    *,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    variant_by_id = {str(row["id"]): str(row.get("hard_variant", "unknown")) for row in rows}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in report.get("records", []):
        variant = variant_by_id.get(str(record.get("id")), "unknown")
        grouped.setdefault(variant, []).append(record)
    return {
        variant: _summarize_role_records(records)
        for variant, records in sorted(grouped.items())
    }


def summarize_l3_decision(
    *,
    full_summary: dict[str, Any],
    primitive_off_summary: dict[str, Any],
    token_numeric_off_summary: dict[str, Any],
    source_binder_off_summary: dict[str, Any],
    full_by_variant: dict[str, dict[str, Any]],
    min_trace_exact: float,
    min_value_accuracy: float,
    min_primitive_value_drop: float,
    min_token_numeric_value_drop: float,
    min_source_binder_value_drop: float,
    min_variant_value_accuracy: float,
) -> dict[str, Any]:
    full_trace = float(full_summary.get("trace_exact_accuracy", 0.0))
    full_value = float(full_summary.get("value_accuracy", 0.0))
    primitive_value = float(primitive_off_summary.get("value_accuracy", 0.0))
    token_numeric_value = float(token_numeric_off_summary.get("value_accuracy", 0.0))
    source_binder_value = float(source_binder_off_summary.get("value_accuracy", 0.0))
    primitive_drop = full_value - primitive_value
    token_numeric_drop = full_value - token_numeric_value
    source_binder_drop = full_value - source_binder_value
    variant_values = {
        variant: float(summary.get("value_accuracy", 0.0))
        for variant, summary in full_by_variant.items()
    }
    min_variant_value = min(variant_values.values()) if variant_values else 0.0
    reject_reasons: list[str] = []
    if full_trace < float(min_trace_exact):
        reject_reasons.append("full trace exact below L3 minimum")
    if full_value < float(min_value_accuracy):
        reject_reasons.append("full value accuracy below L3 minimum")
    if primitive_drop < float(min_primitive_value_drop):
        reject_reasons.append("primitive-off ablation does not drop enough")
    if token_numeric_drop < float(min_token_numeric_value_drop):
        reject_reasons.append("token-numeric-off ablation does not drop enough")
    if source_binder_drop < float(min_source_binder_value_drop):
        reject_reasons.append("source-binder-off ablation does not drop enough")
    if min_variant_value < float(min_variant_value_accuracy):
        reject_reasons.append("at least one hard variant is below minimum value accuracy")
    accepted = not reject_reasons
    return {
        "decision": "accepted_l3" if accepted else "rejected_l3",
        "accepted": accepted,
        "target_level": "L3 major bottleneck",
        "major_bottleneck": "source-position recurrent state generalization",
        "full_trace_exact_accuracy": full_trace,
        "full_value_accuracy": full_value,
        "primitive_off_value_accuracy": primitive_value,
        "primitive_value_drop": primitive_drop,
        "token_numeric_off_value_accuracy": token_numeric_value,
        "token_numeric_value_drop": token_numeric_drop,
        "source_binder_off_value_accuracy": source_binder_value,
        "source_binder_value_drop": source_binder_drop,
        "min_variant_value_accuracy": min_variant_value,
        "variant_value_accuracy": variant_values,
        "reject_reasons": reject_reasons,
    }


def preserve_if_accepted(*, checkpoint: Path, out_dir: Path, decision: dict[str, Any]) -> str | None:
    if not bool(decision.get("accepted")):
        return None
    accepted_path = out_dir / f"accepted_l3_{checkpoint.stem}.pt"
    if accepted_path.exists():
        return str(accepted_path)
    try:
        os.link(checkpoint, accepted_path)
    except OSError:
        shutil.copy2(checkpoint, accepted_path)
    return str(accepted_path)


def run_l3_gate(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_summary = write_l3_hard_rows(
        data_jsonl=root / str(args.data_jsonl),
        summary_json=root / str(args.data_summary_json),
        count_per_variant=int(args.count_per_variant),
        seed=int(args.seed),
    )
    rows = load_rows(root / str(args.data_jsonl))
    checkpoint = Path(args.checkpoint)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"src{os.pathsep}.{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    if args.hf_home:
        env["HF_HOME"] = str(args.hf_home)
    if args.tmpdir:
        env["TMPDIR"] = str(args.tmpdir)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    evals = {
        "full": eval_command(args, checkpoint=checkpoint, out_json=out_dir / "full.json"),
        "primitive_off": eval_command(
            args,
            checkpoint=checkpoint,
            out_json=out_dir / "primitive_off.json",
            primitive_off=True,
        ),
        "token_numeric_off": eval_command(
            args,
            checkpoint=checkpoint,
            out_json=out_dir / "token_numeric_off.json",
            token_numeric_off=True,
        ),
        "source_binder_off": eval_command(
            args,
            checkpoint=checkpoint,
            out_json=out_dir / "source_binder_off.json",
            source_binder_off=True,
        ),
    }
    report: dict[str, Any] = {
        "target_level": "L3 major bottleneck",
        "major_bottleneck": "source-position recurrent state generalization",
        "gate": {
            "baseline_to_beat": "accepted L2 source-pointer checkpoint under hard perturbation",
            "required_score": {
                "min_trace_exact": float(args.min_trace_exact),
                "min_value_accuracy": float(args.min_value_accuracy),
                "min_variant_value_accuracy": float(args.min_variant_value_accuracy),
            },
            "required_ablation_drop": {
                "primitive_value_drop": float(args.min_primitive_value_drop),
                "token_numeric_value_drop": float(
                    args.min_token_numeric_value_drop
                ),
                "source_binder_value_drop": float(args.min_source_binder_value_drop),
            },
            "perturbation_held_out_split": data_summary,
            "promotion_if_pass": (
                "preserve checkpoint as accepted_l3 and allow L3->L4 LM path "
                "tests; source-binder drop is diagnostic unless a nonzero "
                "--min-source-binder-value-drop is requested"
            ),
            "kill_if_fail": "keep L2 only; inspect failed variant before adding more heads/losses",
        },
        "commands": evals,
        "artifacts": {
            "checkpoint": str(checkpoint),
            "data_jsonl": str(args.data_jsonl),
            "data_summary_json": str(args.data_summary_json),
            "out_dir": str(out_dir),
        },
    }
    if bool(args.dry_run):
        report.update({"decision": "dry_run", "accepted": False})
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    exits: dict[str, int] = {}
    for name, command in evals.items():
        exits[name] = run_command(
            command,
            out_path=out_dir / name,
            env=env,
            cwd=root,
        )
    report["exit_codes"] = exits
    if any(code != 0 for code in exits.values()):
        report.update({"decision": "command_failed", "accepted": False})
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    full = load_report(out_dir / "full.json")
    primitive_off = load_report(out_dir / "primitive_off.json")
    token_numeric_off = load_report(out_dir / "token_numeric_off.json")
    source_binder_off = load_report(out_dir / "source_binder_off.json")
    full_by_variant = summarize_by_variant(report=full, rows=rows)
    primitive_off_by_variant = summarize_by_variant(report=primitive_off, rows=rows)
    source_binder_off_by_variant = summarize_by_variant(report=source_binder_off, rows=rows)
    decision = summarize_l3_decision(
        full_summary=full["summary"],
        primitive_off_summary=primitive_off["summary"],
        token_numeric_off_summary=token_numeric_off["summary"],
        source_binder_off_summary=source_binder_off["summary"],
        full_by_variant=full_by_variant,
        min_trace_exact=float(args.min_trace_exact),
        min_value_accuracy=float(args.min_value_accuracy),
        min_primitive_value_drop=float(args.min_primitive_value_drop),
        min_token_numeric_value_drop=float(args.min_token_numeric_value_drop),
        min_source_binder_value_drop=float(args.min_source_binder_value_drop),
        min_variant_value_accuracy=float(args.min_variant_value_accuracy),
    )
    accepted_path = preserve_if_accepted(
        checkpoint=checkpoint,
        out_dir=out_dir,
        decision=decision,
    )
    report.update(
        {
            **decision,
            "full_summary": full["summary"],
            "primitive_off_summary": primitive_off["summary"],
            "token_numeric_off_summary": token_numeric_off["summary"],
            "source_binder_off_summary": source_binder_off["summary"],
            "full_by_variant": full_by_variant,
            "primitive_off_by_variant": primitive_off_by_variant,
            "source_binder_off_by_variant": source_binder_off_by_variant,
            "accepted_checkpoint": accepted_path,
        }
    )
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the accepted L2 source-pointer checkpoint on an L3 hard "
            "perturbation split with causal ablations."
        )
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-jsonl", default=DEFAULT_DATA_JSONL)
    parser.add_argument("--data-summary-json", default=DEFAULT_DATA_SUMMARY)
    parser.add_argument("--count-per-variant", type=int, default=32)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--core-steps", type=int, default=8)
    parser.add_argument("--max-eval-cases", type=int, default=128)
    parser.add_argument("--token-numeric-value-vocab-size", type=int, default=128)
    parser.add_argument("--token-numeric-source-slots", action="store_true")
    parser.add_argument("--token-numeric-source-slot-vocab-size", type=int, default=128)
    parser.add_argument("--token-numeric-source-slot-max-slots", type=int, default=5)
    parser.add_argument("--token-numeric-source-slot-gate-min", type=float, default=0.0)
    parser.add_argument(
        "--token-numeric-source-slot-predicate-feedback",
        action="store_true",
    )
    parser.add_argument(
        "--token-numeric-source-slot-predicate-gate-min",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--core-source-position-binder-gate-min",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--core-source-position-binder-state-gate-min",
        type=float,
        default=0.0,
    )
    parser.add_argument("--core-source-position-binder-state-st", action="store_true")
    parser.add_argument(
        "--core-source-position-binder-source-slots-only",
        action="store_true",
    )
    parser.add_argument(
        "--core-source-position-binder-raw-source-slots",
        action="store_true",
    )
    parser.add_argument("--min-trace-exact", type=float, default=0.10)
    parser.add_argument("--min-value-accuracy", type=float, default=0.40)
    parser.add_argument("--min-primitive-value-drop", type=float, default=0.20)
    parser.add_argument(
        "--min-token-numeric-value-drop",
        type=float,
        default=0.0,
        help=(
            "Optional strict prompt-derived token/source input causality "
            "requirement. With --token-numeric-source-slots this is the "
            "source-slot-off drop."
        ),
    )
    parser.add_argument(
        "--min-source-binder-value-drop",
        type=float,
        default=0.0,
        help=(
            "Optional strict source-binder causality requirement. L3's "
            "canonical recursive-core gate defaults this to 0 because the "
            "source binder is an auxiliary prompt initializer, not the core "
            "recurrent executor."
        ),
    )
    parser.add_argument("--min-variant-value-accuracy", type=float, default=0.30)
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/mnt/nvme1n1p2/hf-cache-qtrm"))
    parser.add_argument("--tmpdir", default=os.environ.get("TMPDIR", "/mnt/nvme1n1p2/tmp"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = run_l3_gate(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("decision") != "command_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
