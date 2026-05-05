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
    SolverStateMachine,
    rollout_trace_rows,
    state_machine_input_text,
    target_tensors,
)


class RowTensors(NamedTuple):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    labels: torch.Tensor


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
                raise ValueError(f"{path}:{line_no}: solver trace rows must not include evidence")
            if bool(row.get("retrieval_allowed", False)) or bool(row.get("memoryos_allowed", False)):
                raise ValueError(f"{path}:{line_no}: shortcut paths are not allowed")
            if not row.get("prompt"):
                raise ValueError(f"{path}:{line_no}: missing prompt")
            if not row.get("operation"):
                raise ValueError(f"{path}:{line_no}: missing operation")
            if "target_state_text" not in row:
                raise ValueError(f"{path}:{line_no}: missing target_state_text")
            rows.append(row)
    if not rows:
        raise ValueError(f"no solver trace rows in {path}")
    return rows


def vocab_texts_for_rows(rows: Iterable[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for row in rows:
        texts.append(state_machine_input_text(row))
        texts.append(str(row.get("previous_state_text", "")))
        texts.append(str(row.get("target_state_text", "")))
        texts.append(str(row.get("final_answer", "")))
        for alias in row.get("answer_aliases") or ():
            texts.append(str(alias))
    return texts


def row_tensors(
    row: dict[str, Any],
    vocab: CharVocab,
    *,
    max_input_len: int,
    max_target_len: int,
    previous_state: str | None = None,
    device: str | torch.device = "cpu",
) -> RowTensors:
    text = state_machine_input_text(row, previous_state=previous_state)
    input_ids = torch.tensor(
        vocab.encode(text, add_eos=True, max_len=int(max_input_len)),
        dtype=torch.long,
        device=device,
    )
    attention_mask = (input_ids != vocab.pad_id).to(torch.long)
    decoder_input_ids, labels = target_tensors(
        vocab,
        str(row.get("target_state_text", "")),
        max_target_len=int(max_target_len),
    )
    return RowTensors(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids.to(device),
        labels=labels.to(device),
    )


def batch_tensors(rows: list[dict[str, Any]], vocab: CharVocab, args: argparse.Namespace) -> RowTensors:
    tensors = [
        row_tensors(
            row,
            vocab,
            max_input_len=args.max_input_len,
            max_target_len=args.max_target_len,
            device=args.device,
        )
        for row in rows
    ]
    return RowTensors(
        input_ids=torch.stack([item.input_ids for item in tensors]),
        attention_mask=torch.stack([item.attention_mask for item in tensors]),
        decoder_input_ids=torch.stack([item.decoder_input_ids for item in tensors]),
        labels=torch.stack([item.labels for item in tensors]),
    )


def loss_for_batch(
    model: SolverStateMachine,
    batch: RowTensors,
    *,
    pad_id: int,
) -> torch.Tensor:
    logits = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        decoder_input_ids=batch.decoder_input_ids,
    )
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch.labels.reshape(-1),
        ignore_index=int(pad_id),
    )


@torch.no_grad()
def predict_state(
    model: SolverStateMachine,
    vocab: CharVocab,
    row: dict[str, Any],
    *,
    previous_state: str,
    max_input_len: int,
    max_target_len: int,
    device: str | torch.device,
) -> str:
    model.eval()
    text = state_machine_input_text(row, previous_state=previous_state)
    input_ids = torch.tensor(
        [vocab.encode(text, add_eos=True, max_len=int(max_input_len))],
        dtype=torch.long,
        device=device,
    )
    attention_mask = (input_ids != vocab.pad_id).to(torch.long)
    decoder_input_ids = torch.tensor([[vocab.bos_id]], dtype=torch.long, device=device)
    generated: list[int] = []
    for _ in range(int(max_target_len)):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        next_id = int(torch.argmax(logits[0, -1]).item())
        generated.append(next_id)
        if next_id == vocab.eos_id:
            break
        next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
        decoder_input_ids = torch.cat([decoder_input_ids, next_tensor], dim=1)
    return vocab.decode(generated)


@torch.no_grad()
def evaluate_model(
    model: SolverStateMachine,
    rows: list[dict[str, Any]],
    vocab: CharVocab,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.eval()
    teacher_hits = 0
    for row in rows:
        batch = batch_tensors([row], vocab, args)
        logits = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            decoder_input_ids=batch.decoder_input_ids,
        )
        pred_text = vocab.decode(torch.argmax(logits[0], dim=-1).tolist())
        teacher_hits += int(pred_text == str(row.get("target_state_text", "")))

    def _predict(row: dict[str, Any], previous_state: str) -> str:
        return predict_state(
            model,
            vocab,
            row,
            previous_state=previous_state,
            max_input_len=args.max_input_len,
            max_target_len=args.max_target_len,
            device=args.device,
        )

    rollout_records = rollout_trace_rows(rows, _predict)
    state_hits = sum(int(bool(record["state_exact_match"])) for record in rollout_records)
    last_by_source: dict[str, dict[str, Any]] = {}
    for record in rollout_records:
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
        "teacher_forced_state_exact": teacher_hits / max(1, len(rows)),
        "rollout_state_exact": state_hits / max(1, len(rollout_records)),
        "rollout_final_exact": final_hits / max(1, len(last_by_source)),
    }


def train_solver_state_machine(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    train_rows = load_trace_rows(args.train_jsonl)
    eval_rows = load_trace_rows(args.eval_jsonl) if args.eval_jsonl else []
    vocab = CharVocab.build(vocab_texts_for_rows(train_rows + eval_rows))
    model = SolverStateMachine(
        vocab_size=len(vocab.id_to_token),
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        pad_id=vocab.pad_id,
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
        batch = batch_tensors(batch_rows_for_step, vocab, args)
        loss = loss_for_batch(model, batch, pad_id=vocab.pad_id)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        loss_value = float(loss.item())
        if int(args.log_every) > 0 and (step + 1) % int(args.log_every) == 0:
            print(f"step={step + 1} loss={loss_value:.4f}")

    train_metrics = evaluate_model(model, train_rows, vocab, args)
    eval_metrics = evaluate_model(model, eval_rows, vocab, args) if eval_rows else {}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "vocab": {
            "token_to_id": vocab.token_to_id,
            "id_to_token": list(vocab.id_to_token),
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
            "Train a small explicit recurrent state-machine probe on pure recursive "
            "solver traces. This isolates raw state-transition learning from donor, "
            "retrieval, and MemoryOS shortcuts."
        )
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--max-target-len", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=240)
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = train_solver_state_machine(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
