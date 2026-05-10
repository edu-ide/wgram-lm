#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROLE_GROUPS = {
    "raw_list": lambda num_roles: range(0, max(1, (int(num_roles) - 2) // 2)),
    "doubled_list": lambda num_roles: range(
        max(1, (int(num_roles) - 2) // 2),
        2 * max(1, (int(num_roles) - 2) // 2),
    ),
    "scalar": lambda num_roles: range(
        2 * max(1, (int(num_roles) - 2) // 2),
        min(int(num_roles), 2 * max(1, (int(num_roles) - 2) // 2) + 2),
    ),
    "scalar_coeff": lambda num_roles: range(
        2 * max(1, (int(num_roles) - 2) // 2),
        min(int(num_roles), 2 * max(1, (int(num_roles) - 2) // 2) + 1),
    ),
    "scalar_residual": lambda num_roles: range(
        min(int(num_roles), 2 * max(1, (int(num_roles) - 2) // 2) + 1),
        min(int(num_roles), 2 * max(1, (int(num_roles) - 2) // 2) + 2),
    ),
    "all": lambda num_roles: range(0, int(num_roles)),
}


def parse_role_spec(spec: str, *, num_roles: int) -> set[int]:
    roles: set[int] = set()
    for raw_part in str(spec or "").split(","):
        part = raw_part.strip()
        if not part:
            continue
        if part in ROLE_GROUPS:
            roles.update(int(value) for value in ROLE_GROUPS[part](num_roles))
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            roles.update(range(int(start), int(end) + 1))
            continue
        roles.add(int(part))
    return {role for role in roles if 0 <= int(role) < int(num_roles)}


def _step_matches_with_oracle_roles(
    predicted: list[int],
    target: list[int],
    *,
    oracle_roles: set[int],
) -> bool:
    for index, target_value in enumerate(target):
        if int(target_value) == -100:
            continue
        if int(index) in oracle_roles:
            continue
        if index >= len(predicted) or int(predicted[index]) != int(target_value):
            return False
    return True


def trace_exact_with_oracle_roles(
    record: dict[str, Any],
    *,
    role_spec: str,
) -> bool:
    predicted_steps = record.get("predicted_values")
    target_steps = record.get("target_values")
    if not isinstance(predicted_steps, list) or not isinstance(target_steps, list):
        raise ValueError("record must contain predicted_values and target_values lists")
    num_roles = max((len(step) for step in target_steps if isinstance(step, list)), default=0)
    oracle_roles = parse_role_spec(role_spec, num_roles=num_roles)
    for step_index, target in enumerate(target_steps):
        if not isinstance(target, list):
            return False
        predicted = predicted_steps[step_index] if step_index < len(predicted_steps) else []
        if not isinstance(predicted, list):
            predicted = []
        if not _step_matches_with_oracle_roles(
            list(predicted),
            list(target),
            oracle_roles=oracle_roles,
        ):
            return False
    return True


def oracle_role_report(data: dict[str, Any], *, role_specs: list[str]) -> dict[str, Any]:
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError("input JSON must contain records")
    rows = len(records)
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    reports = []
    for role_spec in role_specs:
        exact_rows = sum(
            int(trace_exact_with_oracle_roles(record, role_spec=role_spec))
            for record in records
        )
        reports.append(
            {
                "role_spec": str(role_spec),
                "exact_rows": int(exact_rows),
                "rows": int(rows),
                "trace_exact_accuracy": float(exact_rows) / float(rows) if rows else 0.0,
            }
        )
    return {
        "rows": int(rows),
        "raw_exact_rows": int(summary.get("exact_rows", 0)),
        "oracle_roles": reports,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze which role groups would need to be oracle-gold before a "
            "predicted recurrent value trace becomes exact."
        )
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument(
        "--role-spec",
        action="append",
        default=[],
        help=(
            "Role group or comma/range spec to oracle-replace. Built-ins: "
            "raw_list,doubled_list,scalar,scalar_coeff,scalar_residual,all."
        ),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    role_specs = args.role_spec or [
        "scalar_residual",
        "scalar",
        "raw_list",
        "doubled_list",
        "raw_list,doubled_list",
        "all",
    ]
    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    report = oracle_role_report(data, role_specs=role_specs)
    report["input_json"] = str(args.input_json)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if str(args.out_json).strip():
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
