#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch.nn import functional as F

from qtrm_mm.agentic.solver_state_machine import (
    OperationVocab,
    StructuredOperationPolicy,
    execute_solver_transition,
    rollout_trace_rows,
)


class RowTensors(NamedTuple):
    family_id: torch.Tensor
    trace_index_id: torch.Tensor
    depth_id: torch.Tensor
    label: torch.Tensor


def load_trace_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") and row.get("type") != "pure_recursive_solver_trace":
                raise ValueError(f"{path}:{line_no}: expected pure_recursive_solver_trace row")
            if row.get("evidence"):
                raise ValueError(f"{path}:{line_no}: structured operation rows must not include evidence")
            if bool(row.get("retrieval_allowed", False)) or bool(row.get("memoryos_allowed", False)):
                raise ValueError(f"{path}:{line_no}: shortcut paths are not allowed")
            for key in ("task_family", "depth", "trace_index", "operation"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no}: missing {key}")
            rows.append(row)
    if not rows:
        raise ValueError(f"no structured operation rows in {path}")
    return rows


def row_tensors(
    row: dict[str, Any],
    family_vocab: OperationVocab,
    trace_vocab: OperationVocab,
    depth_vocab: OperationVocab,
    operation_vocab: OperationVocab,
    *,
    device: str | torch.device = "cpu",
) -> RowTensors:
    return RowTensors(
        family_id=torch.tensor(
            family_vocab.encode(str(row.get("task_family") or row.get("category"))),
            dtype=torch.long,
            device=device,
        ),
        trace_index_id=torch.tensor(
            trace_vocab.encode(str(int(row["trace_index"]))),
            dtype=torch.long,
            device=device,
        ),
        depth_id=torch.tensor(
            depth_vocab.encode(str(int(row["depth"]))),
            dtype=torch.long,
            device=device,
        ),
        label=torch.tensor(
            operation_vocab.encode(str(row["operation"])),
            dtype=torch.long,
            device=device,
        ),
    )


def batch_tensors(
    rows: list[dict[str, Any]],
    family_vocab: OperationVocab,
    trace_vocab: OperationVocab,
    depth_vocab: OperationVocab,
    operation_vocab: OperationVocab,
    *,
    device: str | torch.device,
) -> RowTensors:
    tensors = [
        row_tensors(
            row,
            family_vocab,
            trace_vocab,
            depth_vocab,
            operation_vocab,
            device=device,
        )
        for row in rows
    ]
    return RowTensors(
        family_id=torch.stack([item.family_id for item in tensors]),
        trace_index_id=torch.stack([item.trace_index_id for item in tensors]),
        depth_id=torch.stack([item.depth_id for item in tensors]),
        label=torch.stack([item.label for item in tensors]),
    )


@torch.no_grad()
def predict_operation(
    model: StructuredOperationPolicy,
    row: dict[str, Any],
    family_vocab: OperationVocab,
    trace_vocab: OperationVocab,
    depth_vocab: OperationVocab,
    operation_vocab: OperationVocab,
    *,
    device: str | torch.device,
) -> str:
    model.eval()
    tensors = row_tensors(
        row,
        family_vocab,
        trace_vocab,
        depth_vocab,
        operation_vocab,
        device=device,
    )
    logits = model(
        family_ids=tensors.family_id.view(1),
        trace_index_ids=tensors.trace_index_id.view(1),
        depth_ids=tensors.depth_id.view(1),
    )
    return operation_vocab.decode(int(torch.argmax(logits[0]).item()))


@torch.no_grad()
def evaluate_policy(
    model: StructuredOperationPolicy,
    rows: list[dict[str, Any]],
    family_vocab: OperationVocab,
    trace_vocab: OperationVocab,
    depth_vocab: OperationVocab,
    operation_vocab: OperationVocab,
    args: argparse.Namespace,
) -> dict[str, Any]:
    op_hits = 0
    for row in rows:
        pred = predict_operation(
            model,
            row,
            family_vocab,
            trace_vocab,
            depth_vocab,
            operation_vocab,
            device=args.device,
        )
        op_hits += int(pred == str(row["operation"]))

    def _predict_state(row: dict[str, Any], previous_state: str) -> str:
        predicted_operation = predict_operation(
            model,
            row,
            family_vocab,
            trace_vocab,
            depth_vocab,
            operation_vocab,
            device=args.device,
        )
        patched = dict(row)
        patched["operation"] = predicted_operation
        try:
            return execute_solver_transition(patched, previous_state)
        except Exception:
            return "<ERROR>"

    records = rollout_trace_rows(rows, _predict_state)
    state_hits = sum(int(bool(record["state_exact_match"])) for record in records)
    last_by_source: dict[str, dict[str, Any]] = {}
    for record in records:
        last_by_source[str(record.get("source_id", ""))] = record
    final_hits = 0
    for record in last_by_source.values():
        aliases = [str(alias) for alias in (record.get("answer_aliases") or [])]
        if not aliases and record.get("final_answer") is not None:
            aliases = [str(record.get("final_answer"))]
        final_hits += int(str(record.get("predicted_state_text", "")) in set(aliases))
    return {
        "rows": len(rows),
        "cases": len(last_by_source),
        "operation_exact": op_hits / max(1, len(rows)),
        "rollout_state_exact": state_hits / max(1, len(records)),
        "rollout_final_exact": final_hits / max(1, len(last_by_source)),
    }


def train_structured_operation_policy(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    train_rows = load_trace_rows(args.train_jsonl)
    eval_rows = load_trace_rows(args.eval_jsonl) if args.eval_jsonl else []
    all_rows = train_rows + eval_rows
    family_vocab = OperationVocab.build(str(row.get("task_family") or row.get("category")) for row in all_rows)
    trace_vocab = OperationVocab.build(str(int(row["trace_index"])) for row in all_rows)
    depth_vocab = OperationVocab.build(str(int(row["depth"])) for row in all_rows)
    operation_vocab = OperationVocab.build(str(row["operation"]) for row in all_rows)
    model = StructuredOperationPolicy(
        num_families=len(family_vocab.id_to_operation),
        num_trace_indices=len(trace_vocab.id_to_operation),
        num_depths=len(depth_vocab.id_to_operation),
        num_operations=len(operation_vocab.id_to_operation),
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    batch_size = int(args.batch_size)
    loss_value = 0.0
    for step in range(int(args.steps)):
        model.train()
        indices = [
            (step * batch_size + offset) % len(train_rows)
            for offset in range(batch_size)
        ]
        batch = batch_tensors(
            [train_rows[index] for index in indices],
            family_vocab,
            trace_vocab,
            depth_vocab,
            operation_vocab,
            device=args.device,
        )
        logits = model(
            family_ids=batch.family_id,
            trace_index_ids=batch.trace_index_id,
            depth_ids=batch.depth_id,
        )
        loss = F.cross_entropy(logits, batch.label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        loss_value = float(loss.item())
        if int(args.log_every) > 0 and (step + 1) % int(args.log_every) == 0:
            print(f"step={step + 1} loss={loss_value:.4f}")

    train_metrics = evaluate_policy(
        model,
        train_rows,
        family_vocab,
        trace_vocab,
        depth_vocab,
        operation_vocab,
        args,
    )
    eval_metrics = (
        evaluate_policy(
            model,
            eval_rows,
            family_vocab,
            trace_vocab,
            depth_vocab,
            operation_vocab,
            args,
        )
        if eval_rows
        else {}
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "family_vocab": {
            "operation_to_id": family_vocab.operation_to_id,
            "id_to_operation": list(family_vocab.id_to_operation),
        },
        "trace_vocab": {
            "operation_to_id": trace_vocab.operation_to_id,
            "id_to_operation": list(trace_vocab.id_to_operation),
        },
        "depth_vocab": {
            "operation_to_id": depth_vocab.operation_to_id,
            "id_to_operation": list(depth_vocab.id_to_operation),
        },
        "operation_vocab": {
            "operation_to_id": operation_vocab.operation_to_id,
            "id_to_operation": list(operation_vocab.id_to_operation),
        },
        "args": vars(args),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "last_loss": loss_value,
    }
    torch.save(checkpoint, out_dir / "last.pt")
    report = {
        "status": "complete",
        "last_loss": loss_value,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "checkpoint": str(out_dir / "last.pt"),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a structured operation policy from task_family, trace_index, "
            "and depth. This tests whether primitive-state QTRM should expose "
            "structured transition metadata instead of reparsing it as text."
        )
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=240)
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = train_structured_operation_policy(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
