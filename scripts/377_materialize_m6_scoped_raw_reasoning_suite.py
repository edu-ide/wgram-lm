#!/usr/bin/env python3
"""Materialize the M6 scoped raw-reasoning suite for Qwen3.6 baseline runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_script_module(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_reasoning = _load_script_module(
    "337_train_qtrm_native_mixed_text_reasoning_probe.py",
    "qtrm_native_mixed_text_reasoning_probe_for_m6_suite",
)
_m6 = _load_script_module(
    "376_build_m6_scoped_raw_reasoning_manifest.py",
    "m6_scoped_raw_reasoning_manifest_for_suite",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def operation_definitions(*, modulus: int) -> list[str]:
    rows: list[str] = []
    for op_id, spec in enumerate(_reasoning.OP_SPECS):
        name, param = spec
        if op_id == 0:
            text = "leave the value unchanged"
        elif name == "add":
            text = f"add {param}"
        elif name == "mul":
            text = f"multiply by {param}"
        elif name == "affine":
            text = f"replace value with value * {param} + {param + 1}"
        else:
            raise ValueError(f"unknown op spec: {spec}")
        rows.append(f"{op_id:02d}: {text} modulo {int(modulus)}")
    return rows


def qwen_prompt(case, *, modulus: int) -> str:
    ops = " ".join(f"{int(op_id):02d}" for op_id in case.op_ids)
    definitions = "\n".join(f"- {row}" for row in operation_definitions(modulus=modulus))
    return (
        "Solve this deterministic modular arithmetic task.\n"
        f"All values are modulo {int(modulus)}.\n"
        "Return only the final two-digit answer, with no explanation.\n\n"
        "Operation IDs:\n"
        f"{definitions}\n\n"
        "Task families:\n"
        "- modchain: start from the value and apply each operation left to right.\n"
        "- revchain: start from the value and apply each operation right to left.\n"
        "- checksum: ignore operation meanings; add the numeric operation IDs to the start value.\n\n"
        f"Task: {case.family} start {int(case.start):02d} ops {ops}\n"
        "Answer:"
    )


def build_cases_from_report(report: dict[str, Any], *, max_cases: int = 0) -> list[Any]:
    train = report.get("train", {}) if isinstance(report.get("train", {}), dict) else {}
    families_value = (
        report.get("eval_task_families")
        or train.get("eval_task_families")
        or report.get("task_families")
        or train.get("task_families")
        or "modchain,revchain,checksum"
    )
    if isinstance(families_value, list):
        families = tuple(str(item) for item in families_value)
    else:
        families = _reasoning.parse_families(str(families_value))
    count = int(max_cases) if int(max_cases) > 0 else int(train.get("eval_cases", 768))
    kwargs = {
        "count": count,
        "seed": int(train.get("eval_seed", 9337)),
        "program_len": int(train.get("program_len", 4)),
        "modulus": int(train.get("modulus", 32)),
        "families": families,
    }
    if bool(train.get("eval_family_order_invariant", False)):
        return _reasoning.build_family_order_invariant_eval_cases(**kwargs)
    return _reasoning.build_cases(**kwargs)


def build_suite(args: argparse.Namespace) -> dict[str, Any]:
    report = _load_json(args.qtrm_report)
    train = report.get("train", {}) if isinstance(report.get("train", {}), dict) else {}
    modulus = int(train.get("modulus", 32))
    cases = build_cases_from_report(report, max_cases=int(args.max_cases))
    rows = []
    for case in cases:
        answer = _reasoning.case_answer(case).strip()
        rows.append(
            {
                "suite_id": str(args.suite_id),
                "prompt_protocol": str(args.prompt_protocol),
                "case_id": str(case.case_id),
                "family": str(case.family),
                "start": int(case.start),
                "op_ids": [int(op_id) for op_id in case.op_ids],
                "qtrm_prompt": _reasoning.case_prompt(
                    case,
                    include_family_tag=bool(report.get("include_family_tag", False)),
                ),
                "qwen_prompt": qwen_prompt(case, modulus=modulus),
                "answer_text": str(answer),
                "answer_value": int(answer),
            }
        )
    return {
        "suite_id": str(args.suite_id),
        "prompt_protocol": str(args.prompt_protocol),
        "source_qtrm_report": str(args.qtrm_report),
        "case_count": len(rows),
        "modulus": modulus,
        "operation_definitions": operation_definitions(modulus=modulus),
        "rows": rows,
    }


def write_suite(suite: dict[str, Any], *, out_jsonl: str | Path, out_meta: str | Path) -> None:
    jsonl_path = Path(out_jsonl)
    meta_path = Path(out_meta)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in suite["rows"]),
        encoding="utf-8",
    )
    meta = {key: value for key, value in suite.items() if key != "rows"}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qtrm-report",
        default="local_eval/research_gate_runner/qtrm_native_l5_multifamily_standard/report.json",
    )
    parser.add_argument("--suite-id", default=_m6.DEFAULT_SUITE_ID)
    parser.add_argument("--prompt-protocol", default=_m6.DEFAULT_PROMPT_PROTOCOL)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument(
        "--out-jsonl",
        default="local_eval/m6_scoped_raw_reasoning_suite/cases.jsonl",
    )
    parser.add_argument(
        "--out-meta",
        default="local_eval/m6_scoped_raw_reasoning_suite/metadata.json",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    suite = build_suite(args)
    write_suite(suite, out_jsonl=args.out_jsonl, out_meta=args.out_meta)
    print(
        json.dumps(
            {
                "suite_id": suite["suite_id"],
                "prompt_protocol": suite["prompt_protocol"],
                "case_count": suite["case_count"],
                "out_jsonl": str(args.out_jsonl),
                "out_meta": str(args.out_meta),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
