#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def role_name(role_index: int, *, num_roles: int) -> str:
    max_list_fields = max(1, (int(num_roles) - 2) // 2)
    doubled_start = max_list_fields
    scalar_coeff = 2 * max_list_fields
    scalar_residual = scalar_coeff + 1
    if int(role_index) < doubled_start:
        return f"raw_list_{role_index}"
    if int(role_index) < scalar_coeff:
        return f"doubled_list_{int(role_index) - doubled_start}"
    if int(role_index) == scalar_coeff:
        return "scalar_coeff"
    if int(role_index) == scalar_residual:
        return "scalar_residual"
    return f"role_{role_index}"


def load_eval(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data.get("records"), list):
        raise ValueError("input JSON must contain records")
    return data


def load_rows_by_id(path: str | Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            row_id = str(row.get("id") or "")
            if not row_id:
                raise ValueError(f"{path}:{line_no}: missing row id")
            rows[row_id] = row
    return rows


def _action_for_record(record: dict[str, Any], step_index: int) -> int | None:
    for key in ("target_codes", "predicted_codes"):
        values = record.get(key)
        if isinstance(values, list) and int(step_index) < len(values):
            value = values[int(step_index)]
            if value is not None and int(value) >= 0:
                return int(value)
    return None


def _action_from_source_rows(
    source_rows: dict[str, dict[str, Any]],
    record: dict[str, Any],
    step_index: int,
) -> int | None:
    row = source_rows.get(str(record.get("id") or ""))
    if not row:
        return None
    codes = row.get("transition_state_codes")
    if not isinstance(codes, dict):
        return None
    value = codes.get(str(int(step_index) + 1))
    if value is None:
        return None
    return int(value)


def summarize(
    data: dict[str, Any],
    *,
    source_rows: dict[str, dict[str, Any]] | None = None,
    top_examples: int = 8,
) -> dict[str, Any]:
    records = data["records"]
    role_totals: Counter[str] = Counter()
    role_errors: Counter[str] = Counter()
    step_totals: Counter[str] = Counter()
    step_errors: Counter[str] = Counter()
    action_totals: Counter[str] = Counter()
    action_errors: Counter[str] = Counter()
    role_action_totals: Counter[str] = Counter()
    role_action_errors: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for record in records:
        predicted_steps = record.get("predicted_values")
        target_steps = record.get("target_values")
        if not isinstance(predicted_steps, list) or not isinstance(target_steps, list):
            labelled = record.get("labelled_values")
            if not isinstance(labelled, list):
                continue
            target_steps = []
            predicted_steps = []
            for item in labelled:
                step_index = int(item["step_index"])
                role_index = int(item["role_index"])
                while len(target_steps) <= step_index:
                    target_steps.append([])
                    predicted_steps.append([])
                while len(target_steps[step_index]) <= role_index:
                    target_steps[step_index].append(-100)
                    predicted_steps[step_index].append(-100)
                target_steps[step_index][role_index] = int(item["target"])
                predicted_steps[step_index][role_index] = int(item["predicted"])

        num_roles = max((len(step) for step in target_steps), default=0)
        for step_index, target in enumerate(target_steps):
            predicted = (
                predicted_steps[step_index]
                if step_index < len(predicted_steps)
                and isinstance(predicted_steps[step_index], list)
                else []
            )
            action = _action_for_record(record, step_index)
            if action is None and source_rows is not None:
                action = _action_from_source_rows(source_rows, record, step_index)
            action_key = "unknown" if action is None else str(action)
            for role_index, target_value in enumerate(target):
                if int(target_value) < 0:
                    continue
                predicted_value = (
                    int(predicted[role_index])
                    if role_index < len(predicted)
                    else -100
                )
                role_key = role_name(role_index, num_roles=num_roles)
                step_key = str(int(step_index) + 1)
                role_action_key = f"{role_key}|action={action_key}"
                hit = predicted_value == int(target_value)
                role_totals[role_key] += 1
                step_totals[step_key] += 1
                action_totals[action_key] += 1
                role_action_totals[role_action_key] += 1
                if not hit:
                    role_errors[role_key] += 1
                    step_errors[step_key] += 1
                    action_errors[action_key] += 1
                    role_action_errors[role_action_key] += 1
                    if len(examples) < int(top_examples):
                        examples.append(
                            {
                                "id": record.get("id", ""),
                                "step": int(step_index) + 1,
                                "action": action,
                                "role_index": int(role_index),
                                "role": role_key,
                                "predicted": predicted_value,
                                "target": int(target_value),
                            }
                        )

    def rates(errors: Counter[str], totals: Counter[str]) -> list[dict[str, Any]]:
        rows = []
        for key, total in totals.items():
            error = int(errors.get(key, 0))
            rows.append(
                {
                    "key": key,
                    "errors": error,
                    "total": int(total),
                    "error_rate": float(error) / float(total) if total else 0.0,
                }
            )
        rows.sort(key=lambda row: (-float(row["error_rate"]), -int(row["errors"]), row["key"]))
        return rows

    return {
        "rows": len(records),
        "summary": data.get("summary", {}),
        "by_role": rates(role_errors, role_totals),
        "by_step": rates(step_errors, step_totals),
        "by_action": rates(action_errors, action_totals),
        "by_role_action": rates(role_action_errors, role_action_totals),
        "examples": examples,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize role-value recurrent state errors by role, step, and action."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--rows-jsonl", default="")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--top-examples", type=int, default=8)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    source_rows = (
        load_rows_by_id(args.rows_jsonl) if str(args.rows_jsonl).strip() else None
    )
    report = summarize(
        load_eval(args.input_json),
        source_rows=source_rows,
        top_examples=int(args.top_examples),
    )
    report["input_json"] = str(args.input_json)
    if source_rows is not None:
        report["rows_jsonl"] = str(args.rows_jsonl)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if str(args.out_json).strip():
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
