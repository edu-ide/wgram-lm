#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from wgram_lm.agentic.solver_state_machine import (
    operation_names_from_logits,
    rollout_solver_trace_from_operations,
)


def load_rows(path: str | Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row.get("solver_trace"), list):
                raise ValueError(f"{path}:{line_no}: missing solver_trace")
            rows.append(row)
            if max_rows is not None and len(rows) >= int(max_rows):
                break
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def _load_depth_train_module():
    path = Path(__file__).with_name("196_train_pure_recursive_depth_supervised.py")
    spec = importlib.util.spec_from_file_location("depth_supervised_train", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    spec.loader.exec_module(module)
    return module


def _ratio(hits: int, total: int) -> str:
    return f"{int(hits)}/{int(total)}"


def summarize_rollouts(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    operation_hits = 0
    state_hits = 0
    total_steps = 0
    final_hits = 0
    by_family: dict[str, dict[str, int]] = {}
    mismatch_examples: list[dict[str, Any]] = []
    for result in results:
        family = str(result.get("task_family") or result.get("category") or "")
        rollout = result["rollout"]
        op_hits = int(rollout["operation_exact_count"])
        st_hits = int(rollout["state_exact_count"])
        steps = int(rollout["total_steps"])
        final = bool(rollout["final_exact_match"])
        operation_hits += op_hits
        state_hits += st_hits
        total_steps += steps
        final_hits += int(final)
        fam = by_family.setdefault(
            family,
            {"operation_hits": 0, "state_hits": 0, "total_steps": 0, "final_hits": 0, "cases": 0},
        )
        fam["operation_hits"] += op_hits
        fam["state_hits"] += st_hits
        fam["total_steps"] += steps
        fam["final_hits"] += int(final)
        fam["cases"] += 1
        if (op_hits != steps or st_hits != steps or not final) and len(mismatch_examples) < 12:
            mismatch_examples.append(
                {
                    "id": result.get("id"),
                    "task_family": family,
                    "predicted_operations": result.get("predicted_operations"),
                    "predicted_final": rollout.get("predicted_final"),
                    "target_final": rollout.get("target_final"),
                    "records": [
                        record
                        for record in rollout.get("records", [])
                        if not record.get("operation_exact_match")
                        or not record.get("state_exact_match")
                    ],
                }
            )
    formatted_by_family: dict[str, dict[str, Any]] = {}
    for family, values in by_family.items():
        formatted_by_family[family] = {
            "operation_exact": _ratio(values["operation_hits"], values["total_steps"]),
            "state_exact": _ratio(values["state_hits"], values["total_steps"]),
            "final_exact": _ratio(values["final_hits"], values["cases"]),
            "operation_accuracy": values["operation_hits"] / max(1, values["total_steps"]),
            "state_accuracy": values["state_hits"] / max(1, values["total_steps"]),
            "final_accuracy": values["final_hits"] / max(1, values["cases"]),
        }
    return {
        "cases": len(results),
        "operation_exact": _ratio(operation_hits, total_steps),
        "state_exact": _ratio(state_hits, total_steps),
        "final_exact": _ratio(final_hits, len(results)),
        "operation_accuracy": operation_hits / max(1, total_steps),
        "state_accuracy": state_hits / max(1, total_steps),
        "final_accuracy": final_hits / max(1, len(results)),
        "by_family": formatted_by_family,
        "mismatch_examples": mismatch_examples,
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
    results: list[dict[str, Any]] = []
    seq_len = int(max_length or cfg.train.seq_len)
    with torch.no_grad():
        for row in rows:
            enc = tokenizer(
                row["prompt"],
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
            )
            trace_len = len(row["solver_trace"])
            if len(operations) < trace_len:
                operations = operations + ["__missing__"] * (trace_len - len(operations))
            rollout = rollout_solver_trace_from_operations(row, operations)
            results.append(
                {
                    "id": row.get("id"),
                    "source_id": row.get("source_id"),
                    "task_family": row.get("task_family") or row.get("category"),
                    "predicted_operations": operations[:trace_len],
                    "rollout": rollout,
                }
            )
    return {
        "checkpoint": checkpoint_path,
        "config": config_path,
        "disable_core": bool(disable_core),
        "core_steps": int(core_steps),
        "summary": summarize_rollouts(results),
        "results": results,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate QTRM primitive operation logits via solver rollout.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--core-steps", type=int, default=4)
    parser.add_argument("--disable-core", action="store_true")
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
    )
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
