#!/usr/bin/env python3
"""Train a thin choice verifier on QTRM thought states for Stage59 rows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wgram_lm.eval.general_answer_interface import (
    answer_aliases,
    normalize_answer_text,
    normalized_alias_set,
    summarize_records,
)


def _load_stage523() -> Any:
    path = Path(__file__).resolve().parent / "523_train_state_text_speaker.py"
    spec = importlib.util.spec_from_file_location("qtrm_stage523", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


stage523 = _load_stage523()


def configure_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def row_choices(row: dict[str, Any]) -> list[str]:
    choices = row.get("choices")
    return [str(choice) for choice in choices] if isinstance(choices, list) else []


def target_choice_index(row: dict[str, Any]) -> int:
    aliases = set(normalized_alias_set(answer_aliases(row)))
    for index, choice in enumerate(row_choices(row)):
        if normalize_answer_text(choice) in aliases:
            return int(index)
    return -1


def build_choice_char_vocab(rows: list[dict[str, Any]]) -> list[str]:
    chars = {"<pad>"}
    for row in rows:
        for choice in row_choices(row):
            chars.update(str(choice))
    return ["<pad>"] + sorted(char for char in chars if char != "<pad>")


def encode_choices(
    rows: list[dict[str, Any]],
    *,
    allowed_chars: list[str],
    max_choices: int,
    max_choice_chars: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    char_index = {char: index for index, char in enumerate(allowed_chars)}
    choice_ids: list[list[list[int]]] = []
    choice_mask: list[list[bool]] = []
    targets: list[int] = []
    for row in rows:
        choices = row_choices(row)[: int(max_choices)]
        row_ids: list[list[int]] = []
        row_mask: list[bool] = []
        for choice in choices:
            ids = [char_index[char] for char in str(choice)[: int(max_choice_chars)]]
            row_ids.append(ids + [0] * (int(max_choice_chars) - len(ids)))
            row_mask.append(True)
        while len(row_ids) < int(max_choices):
            row_ids.append([0] * int(max_choice_chars))
            row_mask.append(False)
        choice_ids.append(row_ids)
        choice_mask.append(row_mask)
        target = target_choice_index(row)
        targets.append(target if 0 <= target < int(max_choices) else -100)
    return (
        torch.tensor(choice_ids, dtype=torch.long, device=device),
        torch.tensor(choice_mask, dtype=torch.bool, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


class ChoiceVerifier(nn.Module):
    def __init__(self, *, d_state: int, vocab_size: int, max_choice_chars: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.d_state = int(d_state)
        self.max_choice_chars = int(max_choice_chars)
        hidden = int(hidden_dim or d_state * 2)
        self.char_embed = nn.Embedding(int(vocab_size), self.d_state, padding_idx=0)
        self.pos_embed = nn.Embedding(self.max_choice_chars, self.d_state)
        self.choice_norm = nn.LayerNorm(self.d_state)
        self.choice_proj = nn.Sequential(
            nn.Linear(self.d_state, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.d_state),
        )
        self.scorer = nn.Sequential(
            nn.Linear(self.d_state * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, readout: torch.Tensor, choice_ids: torch.Tensor, choice_mask: torch.Tensor) -> torch.Tensor:
        if readout.ndim != 2:
            raise ValueError("readout must have shape (batch, d_state)")
        bsz, n_choices, n_chars = choice_ids.shape
        positions = torch.arange(n_chars, device=choice_ids.device)
        embeds = self.char_embed(choice_ids) + self.pos_embed(positions).view(1, 1, n_chars, self.d_state)
        char_mask = choice_ids.ne(0).unsqueeze(-1)
        denom = char_mask.sum(dim=2).clamp_min(1)
        choice_vec = (embeds * char_mask).sum(dim=2) / denom
        choice_vec = self.choice_norm(choice_vec + self.choice_proj(choice_vec))
        readout_expanded = readout.unsqueeze(1).expand(-1, n_choices, -1)
        features = torch.cat(
            [
                readout_expanded,
                choice_vec,
                readout_expanded * choice_vec,
                (readout_expanded - choice_vec).abs(),
            ],
            dim=-1,
        )
        logits = self.scorer(features).squeeze(-1)
        return logits.masked_fill(~choice_mask, -1e9)


def collate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if target_choice_index(row) >= 0]


def train_epoch(
    *,
    wgram_model: Any,
    tokenizer: Any,
    verifier: ChoiceVerifier,
    rows: list[dict[str, Any]],
    allowed_chars: list[str],
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    if args.train_qtrm_core:
        wgram_model.train()
        wgram_model.qwen.eval()
    else:
        wgram_model.eval()
    verifier.train()
    loader = DataLoader(rows, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_rows)
    total_loss = 0.0
    total_rows = 0
    started = time.time()
    for batch in loader:
        context = stage523.thought_context_for_batch(
            wgram_model,
            tokenizer,
            batch,
            max_length=args.max_length,
            n_steps=args.n_steps,
            device=device,
            detach=not bool(args.train_qtrm_core),
        )
        choice_ids, choice_mask, targets = encode_choices(
            batch,
            allowed_chars=allowed_chars,
            max_choices=args.max_choices,
            max_choice_chars=args.max_choice_chars,
            device=device,
        )
        logits = verifier(context["readout"], choice_ids, choice_mask)
        loss = F.cross_entropy(logits.float(), targets, ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = list(verifier.parameters())
        if args.train_qtrm_core:
            params.extend(parameter for parameter in wgram_model.parameters() if parameter.requires_grad)
        torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(batch)
        total_rows += len(batch)
    return {"loss": total_loss / max(1, total_rows), "seconds": time.time() - started}


@torch.no_grad()
def evaluate(
    *,
    wgram_model: Any,
    tokenizer: Any,
    verifier: ChoiceVerifier,
    rows: list[dict[str, Any]],
    allowed_chars: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    wgram_model.eval()
    verifier.eval()
    loader = DataLoader(rows, batch_size=int(args.eval_batch_size), shuffle=False, collate_fn=collate_rows)
    records: list[dict[str, Any]] = []
    for batch in loader:
        context = stage523.thought_context_for_batch(
            wgram_model,
            tokenizer,
            batch,
            max_length=args.max_length,
            n_steps=args.n_steps,
            device=device,
        )
        choice_ids, choice_mask, targets = encode_choices(
            batch,
            allowed_chars=allowed_chars,
            max_choices=args.max_choices,
            max_choice_chars=args.max_choice_chars,
            device=device,
        )
        logits = verifier(context["readout"], choice_ids, choice_mask)
        pred = logits.argmax(dim=-1).detach().cpu().tolist()
        target_list = targets.detach().cpu().tolist()
        for row, pred_index, target_index in zip(batch, pred, target_list):
            choices = row_choices(row)[: int(args.max_choices)]
            selected = choices[int(pred_index)] if choices else ""
            exact = int(pred_index) == int(target_index)
            records.append(
                {
                    "id": stage523.row_id(row),
                    "task_family": row.get("task_family") or row.get("category") or "unknown",
                    "answer_kind": "choice",
                    "aliases": list(answer_aliases(row)),
                    "candidates": choices,
                    "selected_index": int(pred_index),
                    "target_index": int(target_index),
                    "selected": selected,
                    "normalized_selected": normalize_answer_text(selected),
                    "exact": bool(exact),
                    "oracle_exact": True,
                    "selection_mode": "learned_choice_verifier",
                }
            )
    summary = summarize_records(records)
    summary.update(
        {
            "stage": "Stage59 QTRM state choice verifier",
            "plain_language_read": (
                "This asks whether the thought state can recognize the right answer among supplied choices. "
                "If this succeeds while text generation fails, the bottleneck is the speaker/renderer."
            ),
        }
    )
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/mnt/sdc1/tripleyoung/qtrm_eval/20260522_092733_LOCAL_STAGE54B_oracle_guard_mixedall_seed42/best_stochastic_oracle.pt")
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--train-jsonl", default="data/filtered/pure_recursive_solver_trace_all_family_train_cases.jsonl")
    parser.add_argument("--eval-jsonl", default="data/eval/pure_recursive_solver_trace_all_family_heldout_cases.jsonl")
    parser.add_argument("--out-dir", default="local_eval/stage59_choice_verifier")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--max-choices", type=int, default=4)
    parser.add_argument("--max-choice-chars", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1524)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--train-qtrm-core", action="store_true")
    parser.add_argument("--core-impl", default="state_transition")
    parser.add_argument("--core-update", default="mlp")
    parser.add_argument("--answer-path", choices=("state_head", "lm_head"), default="lm_head")
    parser.add_argument("--workspace-pooling", default="sequence")
    parser.add_argument("--recurrent-readout-pooling", default="sharp_attention")
    parser.add_argument("--recurrent-readout-temperature", type=float, default=0.25)
    parser.add_argument("--n-steps", type=int, default=14)
    parser.add_argument("--state-update-schedule", default="nested")
    parser.add_argument("--stochastic-high-level-guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stochastic-high-level-scale", type=float, default=1.0)
    parser.add_argument("--stochastic-high-level-min-std", type=float, default=1e-4)
    parser.add_argument("--stochastic-high-level-max-std", type=float, default=1.0)
    parser.add_argument("--stochastic-high-level-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stochastic-posterior-guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stochastic-transition-mode", choices=("delta", "true_gram"), default="true_gram")
    args = parser.parse_args()

    configure_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    train_rows = stage523.load_jsonl(args.train_jsonl, limit=int(args.train_limit))
    eval_rows = stage523.load_jsonl(args.eval_jsonl, limit=int(args.eval_limit))
    allowed_chars = build_choice_char_vocab([*train_rows, *eval_rows])
    wgram_model, tokenizer, load_stats = stage523.build_qtrm(args, device)
    verifier = ChoiceVerifier(
        d_state=int(wgram_model.d_state),
        vocab_size=len(allowed_chars),
        max_choice_chars=int(args.max_choice_chars),
    ).to(device)
    trainable = list(verifier.parameters())
    if args.train_qtrm_core:
        trainable.extend(parameter for parameter in wgram_model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    for epoch in range(1, int(args.epochs) + 1):
        train = train_epoch(
            wgram_model=wgram_model,
            tokenizer=tokenizer,
            verifier=verifier,
            rows=train_rows,
            allowed_chars=allowed_chars,
            optimizer=optimizer,
            args=args,
            device=device,
        )
        eval_summary, eval_records = evaluate(
            wgram_model=wgram_model,
            tokenizer=tokenizer,
            verifier=verifier,
            rows=eval_rows,
            allowed_chars=allowed_chars,
            args=args,
            device=device,
        )
        train_summary, _ = evaluate(
            wgram_model=wgram_model,
            tokenizer=tokenizer,
            verifier=verifier,
            rows=train_rows,
            allowed_chars=allowed_chars,
            args=args,
            device=device,
        )
        best_accuracy = max(best_accuracy, float(eval_summary["accuracy"]))
        record = {"epoch": epoch, "train": train, "eval": eval_summary, "train_eval": train_summary}
        history.append(record)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train["loss"],
                    "train_accuracy": train_summary["accuracy"],
                    "eval_accuracy": eval_summary["accuracy"],
                    "by_family": eval_summary["by_family"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if float(eval_summary["accuracy"]) >= float(best_accuracy):
            torch.save(
                {
                    "verifier": verifier.state_dict(),
                    "args": vars(args),
                    "allowed_chars": allowed_chars,
                    "load_stats": load_stats,
                    "epoch": epoch,
                    "eval": eval_summary,
                },
                out_dir / "best_choice_verifier.pt",
            )
            (out_dir / "best_records.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in eval_records),
                encoding="utf-8",
            )

    summary = {
        "best_accuracy": best_accuracy,
        "best_epoch": max(history, key=lambda item: item["eval"]["accuracy"])["epoch"] if history else 0,
        "history": history,
        "load_stats": load_stats,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
