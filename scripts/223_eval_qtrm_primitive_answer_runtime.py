#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from wgram_lm.agentic.solver_state_machine import (
    answer_from_primitive_operations,
    operation_names_from_logits,
)


def load_rows(path: str | Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("prompt"):
                raise ValueError(f"{path}:{line_no}: missing prompt")
            if not (row.get("chosen") or row.get("answer")):
                raise ValueError(f"{path}:{line_no}: missing chosen/answer")
            rows.append(row)
            if max_rows is not None and len(rows) >= int(max_rows):
                break
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def runtime_row_from_eval_row(row: Mapping[str, Any]) -> dict[str, Any]:
    runtime_row = {"prompt": str(row.get("prompt") or "")}
    if row.get("question"):
        runtime_row["question"] = str(row["question"])
    return runtime_row


def target_answer_from_row(row: Mapping[str, Any]) -> str:
    return str(row.get("chosen") or row.get("answer") or "")


def score_answer_report(
    row: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    target = target_answer_from_row(row)
    predicted = str(report.get("answer") or "")
    return {
        "id": row.get("id"),
        "source_id": row.get("source_id"),
        "task_family": row.get("task_family") or row.get("category"),
        "target_answer": target,
        "predicted_answer": predicted,
        "answer_exact_match": predicted == target,
        "predicted_operations": report.get("predicted_operations", []),
        "executed_operations": report.get("executed_operations", []),
        "states": report.get("states", []),
        "records": report.get("records", []),
    }


def _ratio(hits: int, total: int) -> str:
    return f"{int(hits)}/{int(total)}"


def summarize_answer_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(results)
    hits = sum(1 for result in results if bool(result.get("answer_exact_match")))
    by_family: dict[str, dict[str, int]] = {}
    mismatch_examples: list[dict[str, Any]] = []
    for result in results:
        family = str(result.get("task_family") or "")
        fam = by_family.setdefault(family, {"hits": 0, "total": 0})
        fam["hits"] += int(bool(result.get("answer_exact_match")))
        fam["total"] += 1
        if not bool(result.get("answer_exact_match")) and len(mismatch_examples) < 12:
            mismatch_examples.append(dict(result))
    formatted_by_family = {
        family: {
            "answer_exact": _ratio(values["hits"], values["total"]),
            "answer_accuracy": values["hits"] / max(1, values["total"]),
        }
        for family, values in by_family.items()
    }
    return {
        "cases": total,
        "answer_exact": _ratio(hits, total),
        "answer_accuracy": hits / max(1, total),
        "by_family": formatted_by_family,
        "mismatch_examples": mismatch_examples,
    }


def _load_depth_train_module():
    path = Path(__file__).with_name("196_train_pure_recursive_depth_supervised.py")
    spec = importlib.util.spec_from_file_location("depth_supervised_train", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    spec.loader.exec_module(module)
    return module


def _runtime_report_from_operations(
    runtime_row: dict[str, Any],
    operations: Sequence[str],
) -> dict[str, Any]:
    answer_info = answer_from_primitive_operations(runtime_row, operations)
    return {
        "prompt": runtime_row.get("prompt") or runtime_row.get("question") or "",
        "predicted_operations": [str(operation) for operation in operations],
        "executed_operations": answer_info["executed_operations"],
        "states": answer_info["states"],
        "answer": answer_info["answer"],
        "records": answer_info["records"],
    }


def runtime_report_from_operations_safe(
    runtime_row: dict[str, Any],
    operations: Sequence[str],
) -> dict[str, Any]:
    try:
        report = _runtime_report_from_operations(runtime_row, operations)
        report["error"] = ""
        return report
    except Exception as exc:  # noqa: BLE001 - eval should preserve model/executor failures
        return {
            "prompt": runtime_row.get("prompt") or runtime_row.get("question") or "",
            "predicted_operations": [str(operation) for operation in operations],
            "executed_operations": [],
            "states": [],
            "answer": "",
            "records": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def evaluate_rows(
    *,
    rows: Sequence[dict[str, Any]],
    config_path: str,
    checkpoint_path: str,
    tokenizer_model_id: str,
    max_length: int | None,
    core_steps: int,
    disable_core: bool,
    state_constrained_operations: bool = False,
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from wgram_lm.config import load_config
    from wgram_lm.wgram_model import QTRMMultimodalModel
    from wgram_lm.qwen_donor import QwenDonorAdapter
    from wgram_lm.training.train import load_initial_checkpoint

    depth_train = _load_depth_train_module()
    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = QTRMMultimodalModel(cfg.model).to(device)
    load_initial_checkpoint(model, checkpoint_path, map_location=device)
    model.eval()
    donor = QwenDonorAdapter(cfg.donor)
    operation_to_id = depth_train.primitive_transition_operation_id_map(
        int(cfg.model.primitive_transition_num_operations)
    )
    id_to_operation = {idx: operation for operation, idx in operation_to_id.items()}
    seq_len = int(max_length or cfg.train.seq_len)
    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            runtime_row = runtime_row_from_eval_row(row)
            prompt = runtime_row["prompt"]
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=seq_len,
            ).to(device)
            donor_out = donor.encode_inputs(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                return_logits=False,
            )
            old_steps = model.cfg.outer_steps
            model.cfg.outer_steps = int(core_steps)
            try:
                with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
                    outputs = model(
                        enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        text_states=donor_out["text_states"].detach().to(device),
                        disable_core=bool(disable_core),
                    )
            finally:
                model.cfg.outer_steps = old_steps
            operations = operation_names_from_logits(
                outputs["primitive_transition_operation_logits"],
                id_to_operation,
                row=runtime_row,
                state_constrained=bool(state_constrained_operations),
            )
            report = runtime_report_from_operations_safe(runtime_row, operations)
            results.append(score_answer_report(row, report))
    return {
        "checkpoint": checkpoint_path,
        "config": config_path,
        "disable_core": bool(disable_core),
        "core_steps": int(core_steps),
        "state_constrained_operations": bool(state_constrained_operations),
        "summary": summarize_answer_results(results),
        "results": results,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate QTRM primitive answer runtime using prompt and chosen/answer only."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--core-steps", type=int, default=4)
    parser.add_argument("--disable-core", action="store_true")
    parser.add_argument(
        "--state-constrained-operations",
        action="store_true",
        help=(
            "Decode primitive operations with executor-state legality constraints. "
            "This is a scaffold-safe runtime metric, not the raw operation argmax metric."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = load_rows(args.data_jsonl, max_rows=args.max_rows)
    report = evaluate_rows(
        rows=rows,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        tokenizer_model_id=args.tokenizer_model_id,
        max_length=args.max_length,
        core_steps=args.core_steps,
        disable_core=args.disable_core,
        state_constrained_operations=args.state_constrained_operations,
    )
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
