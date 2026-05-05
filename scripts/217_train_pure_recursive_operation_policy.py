#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import torch
from torch.nn import functional as F

from qtrm_mm.agentic.solver_state_machine import (
    CharVocab,
    OperationPolicy,
    OperationVocab,
    execute_solver_transition,
    operation_policy_input_text,
    rollout_trace_rows,
)


class RowTensors(NamedTuple):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
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
                raise ValueError(f"{path}:{line_no}: operation policy rows must not include evidence")
            if bool(row.get("retrieval_allowed", False)) or bool(row.get("memoryos_allowed", False)):
                raise ValueError(f"{path}:{line_no}: shortcut paths are not allowed")
            if not row.get("operation"):
                raise ValueError(f"{path}:{line_no}: missing operation")
            if not row.get("prompt") and not row.get("question"):
                raise ValueError(f"{path}:{line_no}: missing prompt/question")
            rows.append(row)
    if not rows:
        raise ValueError(f"no operation policy rows in {path}")
    return rows


def vocab_texts_for_rows(rows: Iterable[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for row in rows:
        texts.append(operation_policy_input_text(row))
        texts.append(str(row.get("previous_state_text", "")))
        texts.append(str(row.get("target_state_text", "")))
        texts.append(str(row.get("final_answer", "")))
    return texts


def row_tensors(
    row: dict[str, Any],
    char_vocab: CharVocab,
    operation_vocab: OperationVocab,
    *,
    max_input_len: int,
    previous_state: str | None = None,
    device: str | torch.device = "cpu",
) -> RowTensors:
    text = operation_policy_input_text(row, previous_state=previous_state)
    input_ids = torch.tensor(
        char_vocab.encode(text, add_eos=True, max_len=int(max_input_len)),
        dtype=torch.long,
        device=device,
    )
    return RowTensors(
        input_ids=input_ids,
        attention_mask=(input_ids != char_vocab.pad_id).to(torch.long),
        label=torch.tensor(operation_vocab.encode(str(row["operation"])), dtype=torch.long, device=device),
    )


def batch_tensors(
    rows: list[dict[str, Any]],
    char_vocab: CharVocab,
    operation_vocab: OperationVocab,
    args: argparse.Namespace,
) -> RowTensors:
    tensors = [
        row_tensors(
            row,
            char_vocab,
            operation_vocab,
            max_input_len=args.max_input_len,
            device=args.device,
        )
        for row in rows
    ]
    return RowTensors(
        input_ids=torch.stack([item.input_ids for item in tensors]),
        attention_mask=torch.stack([item.attention_mask for item in tensors]),
        label=torch.stack([item.label for item in tensors]),
    )


@torch.no_grad()
def predict_operation(
    model: OperationPolicy,
    char_vocab: CharVocab,
    operation_vocab: OperationVocab,
    row: dict[str, Any],
    *,
    previous_state: str,
    max_input_len: int,
    device: str | torch.device,
) -> str:
    model.eval()
    tensors = row_tensors(
        row,
        char_vocab,
        operation_vocab,
        max_input_len=max_input_len,
        previous_state=previous_state,
        device=device,
    )
    logits = model(
        input_ids=tensors.input_ids.unsqueeze(0),
        attention_mask=tensors.attention_mask.unsqueeze(0),
    )
    return operation_vocab.decode(int(torch.argmax(logits[0]).item()))


@torch.no_grad()
def evaluate_policy(
    model: OperationPolicy,
    rows: list[dict[str, Any]],
    char_vocab: CharVocab,
    operation_vocab: OperationVocab,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.eval()
    op_hits = 0
    for row in rows:
        pred = predict_operation(
            model,
            char_vocab,
            operation_vocab,
            row,
            previous_state=str(row.get("previous_state_text", "")),
            max_input_len=args.max_input_len,
            device=args.device,
        )
        op_hits += int(pred == str(row.get("operation")))

    def _predict_state(row: dict[str, Any], previous_state: str) -> str:
        predicted_operation = predict_operation(
            model,
            char_vocab,
            operation_vocab,
            row,
            previous_state=previous_state,
            max_input_len=args.max_input_len,
            device=args.device,
        )
        patched_row = dict(row)
        patched_row["operation"] = predicted_operation
        try:
            return execute_solver_transition(patched_row, previous_state)
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


def train_operation_policy(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    train_rows = load_trace_rows(args.train_jsonl)
    eval_rows = load_trace_rows(args.eval_jsonl) if args.eval_jsonl else []
    char_vocab = CharVocab.build(vocab_texts_for_rows(train_rows + eval_rows))
    operation_vocab = OperationVocab.build(
        [str(row["operation"]) for row in train_rows + eval_rows]
    )
    model = OperationPolicy(
        vocab_size=len(char_vocab.id_to_token),
        num_operations=len(operation_vocab.id_to_operation),
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        pad_id=char_vocab.pad_id,
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
        batch_rows_for_step = [train_rows[index] for index in indices]
        batch = batch_tensors(batch_rows_for_step, char_vocab, operation_vocab, args)
        logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        loss = F.cross_entropy(logits, batch.label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        loss_value = float(loss.item())
        if int(args.log_every) > 0 and (step + 1) % int(args.log_every) == 0:
            print(f"step={step + 1} loss={loss_value:.4f}")

    train_metrics = evaluate_policy(model, train_rows, char_vocab, operation_vocab, args)
    eval_metrics = evaluate_policy(model, eval_rows, char_vocab, operation_vocab, args) if eval_rows else {}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "char_vocab": {
            "token_to_id": char_vocab.token_to_id,
            "id_to_token": list(char_vocab.id_to_token),
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
            "Train a recurrent operation policy for pure recursive solver traces. "
            "The policy predicts the next primitive; a deterministic primitive "
            "transition updates explicit state."
        )
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=240)
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = train_operation_policy(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
