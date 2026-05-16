#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ACTIVE_CANDIDATES = {
    "dual_path_reverse",
    "reversed_hybrid_3to1",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed QTRM-native dual-path reverse length gate."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--profile", default="short", choices=("smoke", "short", "standard"))
    parser.add_argument("--lengths", default="4,6,8")
    parser.add_argument("--candidates", default="official,dual_path_reverse")
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--train-cases", type=int, default=None)
    parser.add_argument("--eval-cases", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-kv-heads", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _candidate_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _min_present(values: list[float | None]) -> float | None:
    present = [item for item in values if item is not None]
    if not present:
        return None
    return min(present)


def _active_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("candidate")) in ACTIVE_CANDIDATES]


def _official_by_length(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_len: dict[int, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("candidate")) != "official":
            continue
        try:
            by_len[int(row["program_len"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return by_len


def build_report(
    *,
    args: argparse.Namespace,
    command: list[str],
    exit_code: int,
    summary: dict[str, Any] | None,
) -> dict[str, Any]:
    if summary is None:
        return {
            "decision": "dry_run" if args.dry_run else "command_failed",
            "accepted": False,
            "target_level": "QTRM-native fixed dual-path reverse length gate",
            "active_architecture": "trm_dual_z_reversed_hybrid_3to1",
            "command": command,
            "subprocess_exit_code": exit_code,
            "next_action": "run the gate and inspect stdout/stderr before promotion",
        }

    rows = list(summary.get("rows") or [])
    fail_fast_stop = summary.get("fail_fast_stop")
    active_rows = _active_rows(rows)
    official_rows = _official_by_length(rows)
    requested_candidates = set(_candidate_list(args.candidates))
    requested_lengths = [
        int(item)
        for item in _candidate_list(args.lengths)
        if str(item).strip().lstrip("-").isdigit()
    ]

    reject_reasons: list[str] = []
    if not active_rows:
        reject_reasons.append("no_dual_path_reverse_rows")
    active_by_len: dict[int, dict[str, Any]] = {}
    for row in active_rows:
        try:
            active_by_len[int(row["program_len"])] = row
        except (KeyError, TypeError, ValueError):
            reject_reasons.append(f"bad_active_row_length:{row.get('run')}")

    fail_fast_len = None
    if isinstance(fail_fast_stop, dict):
        try:
            fail_fast_len = int(fail_fast_stop.get("program_len"))
        except (TypeError, ValueError):
            fail_fast_len = None

    for length in requested_lengths:
        if fail_fast_len is not None and length > fail_fast_len:
            continue
        if "dual_path_reverse" in requested_candidates and length not in active_by_len:
            reject_reasons.append(f"missing_dual_path_reverse_len{length}")

    if isinstance(fail_fast_stop, dict):
        reasons = ",".join(str(item) for item in fail_fast_stop.get("reject_reasons") or [])
        reject_reasons.append(
            f"fail_fast_len{fail_fast_stop.get('program_len')}_{fail_fast_stop.get('candidate')}:{reasons}"
        )

    for row in active_rows:
        if not bool(row.get("accepted")):
            reasons = ",".join(str(item) for item in row.get("reject_reasons") or [])
            reject_reasons.append(f"{row.get('run')}:strict_reject:{reasons}")
        length = row.get("program_len")
        try:
            official = official_rows.get(int(length))
        except (TypeError, ValueError):
            official = None
        active_exact = _float(row.get("full_generation_exact"))
        official_exact = _float(official.get("full_generation_exact")) if official else None
        if active_exact is not None and official_exact is not None and active_exact < official_exact:
            reject_reasons.append(
                f"{row.get('run')}:below_official:{active_exact:.6g}<{official_exact:.6g}"
            )
        active_target_exact = _float(row.get("target_active_len_generation_exact"))
        official_target_exact = (
            _float(official.get("target_active_len_generation_exact")) if official else None
        )
        if (
            active_target_exact is not None
            and official_target_exact is not None
            and active_target_exact < official_target_exact
        ):
            reject_reasons.append(
                f"{row.get('run')}:target_len_below_official:"
                f"{active_target_exact:.6g}<{official_target_exact:.6g}"
            )

    accepted = not reject_reasons and exit_code == 0
    active_exact_values = [_float(row.get("full_generation_exact")) for row in active_rows]
    active_depth_gains = [_float(row.get("full_minus_think0")) for row in active_rows]
    active_ablation_margins = [
        _float(row.get("full_minus_worst_ablation")) for row in active_rows
    ]
    active_target_len_exact_values = [
        _float(row.get("target_active_len_generation_exact")) for row in active_rows
    ]
    deltas_vs_official: list[float | None] = []
    target_len_deltas_vs_official: list[float | None] = []
    for row in active_rows:
        try:
            official = official_rows.get(int(row["program_len"]))
        except (KeyError, TypeError, ValueError):
            official = None
        active_exact = _float(row.get("full_generation_exact"))
        official_exact = _float(official.get("full_generation_exact")) if official else None
        if active_exact is None or official_exact is None:
            deltas_vs_official.append(None)
        else:
            deltas_vs_official.append(active_exact - official_exact)
        active_target_exact = _float(row.get("target_active_len_generation_exact"))
        official_target_exact = (
            _float(official.get("target_active_len_generation_exact")) if official else None
        )
        if active_target_exact is None or official_target_exact is None:
            target_len_deltas_vs_official.append(None)
        else:
            target_len_deltas_vs_official.append(active_target_exact - official_target_exact)

    decisive_metrics = {
        "active_architecture": "trm_dual_z_reversed_hybrid_3to1",
        "active_rows": len(active_rows),
        "min_active_full_generation_exact": _min_present(active_exact_values),
        "min_active_full_minus_think0": _min_present(active_depth_gains),
        "min_active_full_minus_worst_ablation": _min_present(active_ablation_margins),
        "min_active_target_len_generation_exact": _min_present(
            active_target_len_exact_values
        ),
        "min_active_minus_official": _min_present(deltas_vs_official),
        "min_active_target_len_minus_official": _min_present(
            target_len_deltas_vs_official
        ),
    }
    return {
        "decision": (
            "accepted_dual_path_reverse_length_gate"
            if accepted
            else "rejected_dual_path_reverse_length_gate"
        ),
        "accepted": accepted,
        "target_level": "QTRM-native fixed dual-path reverse length gate",
        "major_bottleneck": (
            "prove the fixed dual-path reverse TRM core scales len4->len6->len8 "
            "through the ordinary native LM path"
        ),
        "active_architecture": "trm_dual_z_reversed_hybrid_3to1",
        "baseline_architecture": "official_trm_think",
        "command": command,
        "subprocess_exit_code": exit_code,
        "summary_path": str(Path(args.out_dir) / "summary.json"),
        "decisive_metrics": decisive_metrics,
        "rows": rows,
        "fail_fast_stop": fail_fast_stop,
        "reject_reasons": reject_reasons,
        "next_action": (
            "promote this architecture to the native language non-regression gate"
            if accepted
            else "do not switch architecture again; fix the failed dual-path reverse length/depth/ablation axis"
        ),
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = ["bash", "scripts/346_run_dual_trm_3to1_length_gate.sh"]

    env = dict(os.environ)
    env["PYTHON_BIN"] = args.python_bin
    env["OUT_ROOT"] = str(out_dir)
    env["PROFILE"] = args.profile
    env["LENGTHS"] = args.lengths
    env["CANDIDATES"] = args.candidates
    if args.device:
        env["DEVICE"] = args.device

    overrides = {
        "STEPS": args.steps,
        "TRAIN_CASES": args.train_cases,
        "EVAL_CASES": args.eval_cases,
        "BATCH_SIZE": args.batch_size,
        "D_MODEL": args.d_model,
        "D_FF": args.d_ff,
        "N_HEADS": args.n_heads,
        "N_KV_HEADS": args.n_kv_heads,
        "SEED_BASE": args.seed_base,
        "EVAL_SEED": args.eval_seed,
        "LOG_EVERY": args.log_every,
    }
    for name, value in overrides.items():
        if value is not None:
            env[name] = str(value)

    if args.dry_run:
        report = build_report(args=args, command=command, exit_code=0, summary=None)
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    completed = subprocess.run(command, cwd=repo_root(), env=env, text=True, check=False)
    summary_path = out_dir / "summary.json"
    summary = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = build_report(
        args=args,
        command=command,
        exit_code=int(completed.returncode),
        summary=summary,
    )
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if completed.returncode == 0 else int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
