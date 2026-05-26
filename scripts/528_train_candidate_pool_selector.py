#!/usr/bin/env python3
"""Train a register-style selector over a typed candidate pool.

Deprecated diagnostic scaffold: the pool is hand-built by typed heuristics.
This can reproduce the upper-bound evidence, but final-only Stage59 work must
learn the working table inside the evaluated answer path.
"""

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

from qtrm_mm.eval.general_answer_interface import (
    answer_aliases,
    normalize_answer_text,
    normalized_alias_set,
    summarize_records,
)


def _load_script(name: str, filename: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


stage523 = _load_script("qtrm_stage523_for_528", "523_train_state_text_speaker.py")
stage524 = _load_script("qtrm_stage524_for_528", "524_train_state_choice_verifier.py")
stage525 = _load_script("qtrm_stage525_for_528", "525_eval_qwen_candidate_exposure.py")


def configure_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def row_choices(row: dict[str, Any]) -> list[str]:
    choices = row.get("choices")
    return [str(choice) for choice in choices] if isinstance(choices, list) else []


def candidate_pool(row: dict[str, Any], *, max_pool_candidates: int) -> list[str]:
    return stage525.typed_heuristic_candidates(row, max_candidates=int(max_pool_candidates))


def build_pool_char_vocab(rows: list[dict[str, Any]], *, max_pool_candidates: int) -> list[str]:
    chars = {"<pad>"}
    for row in rows:
        for text in [*row_choices(row), *candidate_pool(row, max_pool_candidates=max_pool_candidates)]:
            chars.update(str(text))
    return ["<pad>"] + sorted(char for char in chars if char != "<pad>")


def encode_string_table(
    strings: list[list[str]],
    *,
    allowed_chars: list[str],
    max_items: int,
    max_chars: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    char_index = {char: index for index, char in enumerate(allowed_chars)}
    rows: list[list[list[int]]] = []
    masks: list[list[bool]] = []
    for row_strings in strings:
        row_ids: list[list[int]] = []
        row_mask: list[bool] = []
        for text in row_strings[: int(max_items)]:
            ids = [char_index.get(char, 0) for char in str(text)[: int(max_chars)]]
            row_ids.append(ids + [0] * (int(max_chars) - len(ids)))
            row_mask.append(True)
        while len(row_ids) < int(max_items):
            row_ids.append([0] * int(max_chars))
            row_mask.append(False)
        rows.append(row_ids)
        masks.append(row_mask)
    return (
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.bool, device=device),
    )


def pool_targets(
    rows: list[dict[str, Any]],
    pools: list[list[str]],
    *,
    max_pool_candidates: int,
    target_mode: str,
    device: torch.device,
) -> torch.Tensor:
    targets: list[list[float]] = []
    for row, pool in zip(rows, pools):
        if target_mode == "answer":
            positives = set(normalized_alias_set(answer_aliases(row)))
        elif target_mode == "choices":
            positives = {normalize_answer_text(choice) for choice in row_choices(row)}
        else:
            raise ValueError(f"unsupported target_mode: {target_mode}")
        row_targets: list[float] = []
        for candidate in pool[: int(max_pool_candidates)]:
            row_targets.append(float(normalize_answer_text(candidate) in positives))
        row_targets += [0.0] * (int(max_pool_candidates) - len(row_targets))
        targets.append(row_targets[: int(max_pool_candidates)])
    return torch.tensor(targets, dtype=torch.float32, device=device)


def topk_candidates(scores: torch.Tensor, pools: list[list[str]], *, k: int) -> list[list[str]]:
    out: list[list[str]] = []
    for row_scores, pool in zip(scores.detach().cpu(), pools):
        if not pool:
            out.append([])
            continue
        values = row_scores[: len(pool)]
        order = torch.argsort(values, descending=True).tolist()
        selected = [pool[index] for index in order[: int(k)]]
        out.append(stage525.dedupe_candidates(selected, max_candidates=int(k)))
    return out


class CandidatePoolSelector(nn.Module):
    """Score typed candidate-pool rows against the QTRM thought readout."""

    def __init__(self, *, d_state: int, vocab_size: int, max_chars: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.d_state = int(d_state)
        self.max_chars = int(max_chars)
        hidden = int(hidden_dim or d_state * 2)
        self.char_embed = nn.Embedding(int(vocab_size), self.d_state, padding_idx=0)
        self.pos_embed = nn.Embedding(self.max_chars, self.d_state)
        self.pool_norm = nn.LayerNorm(self.d_state)
        self.pool_proj = nn.Sequential(
            nn.Linear(self.d_state, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.d_state),
        )
        self.scorer = nn.Sequential(
            nn.Linear(self.d_state * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, readout: torch.Tensor, pool_ids: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        bsz, n_pool, n_chars = pool_ids.shape
        positions = torch.arange(n_chars, device=pool_ids.device)
        embeds = self.char_embed(pool_ids) + self.pos_embed(positions).view(1, 1, n_chars, self.d_state)
        char_mask = pool_ids.ne(0).unsqueeze(-1)
        denom = char_mask.sum(dim=2).clamp_min(1)
        pool_vec = (embeds * char_mask).sum(dim=2) / denom
        pool_vec = self.pool_norm(pool_vec + self.pool_proj(pool_vec))
        readout_expanded = readout.unsqueeze(1).expand(-1, n_pool, -1)
        features = torch.cat(
            [
                readout_expanded,
                pool_vec,
                readout_expanded * pool_vec,
                (readout_expanded - pool_vec).abs(),
            ],
            dim=-1,
        )
        logits = self.scorer(features).squeeze(-1)
        return logits.masked_fill(~pool_mask, -1e9)


def collate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row_choices(row)]


def train_epoch(
    *,
    qtrm_model: Any,
    tokenizer: Any,
    selector: CandidatePoolSelector,
    rows: list[dict[str, Any]],
    allowed_chars: list[str],
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    qtrm_model.eval()
    selector.train()
    loader = DataLoader(rows, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_rows)
    total_loss = 0.0
    total_rows = 0
    started = time.time()
    for batch in loader:
        pools = [candidate_pool(row, max_pool_candidates=args.max_pool_candidates) for row in batch]
        context = stage523.thought_context_for_batch(
            qtrm_model,
            tokenizer,
            batch,
            max_length=args.max_length,
            n_steps=args.n_steps,
            device=device,
            detach=True,
        )
        pool_ids, pool_mask = encode_string_table(
            pools,
            allowed_chars=allowed_chars,
            max_items=args.max_pool_candidates,
            max_chars=args.max_candidate_chars,
            device=device,
        )
        targets = pool_targets(
            batch,
            pools,
            max_pool_candidates=args.max_pool_candidates,
            target_mode=args.target_mode,
            device=device,
        )
        logits = selector(context["readout"], pool_ids, pool_mask)
        raw_loss = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none")
        loss = (raw_loss * pool_mask.to(raw_loss.dtype)).sum() / pool_mask.sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), float(args.grad_clip))
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(batch)
        total_rows += len(batch)
    return {"loss": total_loss / max(1, total_rows), "seconds": time.time() - started}


@torch.no_grad()
def evaluate(
    *,
    qtrm_model: Any,
    tokenizer: Any,
    selector: CandidatePoolSelector,
    verifier: stage524.ChoiceVerifier | None,
    verifier_allowed_chars: list[str] | None,
    rows: list[dict[str, Any]],
    allowed_chars: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    qtrm_model.eval()
    selector.eval()
    if verifier is not None:
        verifier.eval()
    loader = DataLoader(rows, batch_size=int(args.eval_batch_size), shuffle=False, collate_fn=collate_rows)
    records: list[dict[str, Any]] = []
    pool_oracle_hits = 0
    for batch in loader:
        pools = [candidate_pool(row, max_pool_candidates=args.max_pool_candidates) for row in batch]
        context = stage523.thought_context_for_batch(
            qtrm_model,
            tokenizer,
            batch,
            max_length=args.max_length,
            n_steps=args.n_steps,
            device=device,
            detach=True,
        )
        pool_ids, pool_mask = encode_string_table(
            pools,
            allowed_chars=allowed_chars,
            max_items=args.max_pool_candidates,
            max_chars=args.max_candidate_chars,
            device=device,
        )
        pool_scores = selector(context["readout"], pool_ids, pool_mask)
        exposed_candidates = topk_candidates(pool_scores, pools, k=args.max_candidates)

        selected_indices: list[int] = [-1] * len(batch)
        selected_texts: list[str] = [cands[0] if cands else "" for cands in exposed_candidates]
        if verifier is not None and verifier_allowed_chars is not None:
            choice_ids, choice_mask = stage525.encode_candidate_strings(
                exposed_candidates,
                allowed_chars=verifier_allowed_chars,
                max_choices=args.max_candidates,
                max_choice_chars=args.verifier_max_choice_chars,
                device=device,
            )
            verifier_scores = verifier(context["readout"], choice_ids, choice_mask)
            selected_indices = verifier_scores.argmax(dim=-1).detach().cpu().tolist()
            selected_texts = [
                exposed_candidates[row_i][index] if 0 <= index < len(exposed_candidates[row_i]) else ""
                for row_i, index in enumerate(selected_indices)
            ]

        for row, pool, candidates, selected_index, selected in zip(
            batch,
            pools,
            exposed_candidates,
            selected_indices,
            selected_texts,
        ):
            aliases = set(normalized_alias_set(answer_aliases(row)))
            pool_has_gold = any(normalize_answer_text(candidate) in aliases for candidate in pool)
            pool_oracle_hits += int(pool_has_gold)
            oracle_index = next(
                (index for index, candidate in enumerate(candidates) if normalize_answer_text(candidate) in aliases),
                -1,
            )
            exact = normalize_answer_text(selected) in aliases
            records.append(
                {
                    "id": stage523.row_id(row),
                    "task_family": row.get("task_family") or row.get("category") or "unknown",
                    "answer_kind": "pool_selected_candidate",
                    "aliases": list(answer_aliases(row)),
                    "pool": pool,
                    "candidates": candidates,
                    "pool_oracle_exact": bool(pool_has_gold),
                    "oracle_exact": bool(oracle_index >= 0),
                    "oracle_index": int(oracle_index),
                    "selected": selected,
                    "selected_index": int(selected_index),
                    "normalized_selected": normalize_answer_text(selected),
                    "exact": bool(exact),
                    "selection_mode": "typed_pool_selector_plus_verifier"
                    if verifier is not None
                    else "typed_pool_selector_first_candidate",
                }
            )
    summary = summarize_records(records)
    summary.update(
        {
            "pool_oracle_coverage": pool_oracle_hits / max(1, len(records)),
            "exposed_oracle_coverage": sum(1 for row in records if row["oracle_exact"]) / max(1, len(records)),
            "stage": "Stage59 typed candidate pool selector",
            "plain_language_read": (
                "This is a working-table upper-bound: candidates are copied from a typed pool instead of invented as characters."
            ),
        }
    )
    return summary, records


def load_verifier(args: argparse.Namespace, *, d_state: int, device: torch.device):
    if not args.verifier_checkpoint:
        return None, None, 0
    payload = torch.load(str(args.verifier_checkpoint), map_location=device)
    allowed_chars = list(payload["allowed_chars"])
    verifier_args = payload.get("args") or {}
    max_choice_chars = int(verifier_args.get("max_choice_chars", args.verifier_max_choice_chars))
    verifier = stage524.ChoiceVerifier(
        d_state=int(d_state),
        vocab_size=len(allowed_chars),
        max_choice_chars=max_choice_chars,
    ).to(device)
    verifier.load_state_dict(payload["verifier"], strict=True)
    return verifier, allowed_chars, max_choice_chars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/mnt/sdc1/tripleyoung/qtrm_eval/20260522_092733_LOCAL_STAGE54B_oracle_guard_mixedall_seed42/best_stochastic_oracle.pt")
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--verifier-checkpoint", default="")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--max-pool-candidates", type=int, default=16)
    parser.add_argument("--max-candidate-chars", type=int, default=24)
    parser.add_argument("--verifier-max-choice-chars", type=int, default=24)
    parser.add_argument("--target-mode", choices=("answer", "choices"), default="answer")
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1528)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--allow-diagnostic-scaffold",
        action="store_true",
        help="Required because this script trains over a hand-built heuristic candidate pool.",
    )
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

    if not args.allow_diagnostic_scaffold:
        raise SystemExit(
            "scripts/528 is a deprecated diagnostic scaffold. "
            "Use --allow-diagnostic-scaffold only for audit/reproduction, not final-path experiments."
        )

    configure_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    train_rows = stage523.load_jsonl(args.train_jsonl, limit=int(args.train_limit))
    eval_rows = stage523.load_jsonl(args.eval_jsonl, limit=int(args.eval_limit))
    allowed_chars = build_pool_char_vocab(
        [*train_rows, *eval_rows],
        max_pool_candidates=int(args.max_pool_candidates),
    )
    qtrm_model, tokenizer, load_stats = stage523.build_qtrm(args, device)
    selector = CandidatePoolSelector(
        d_state=int(qtrm_model.d_state),
        vocab_size=len(allowed_chars),
        max_chars=int(args.max_candidate_chars),
        hidden_dim=int(args.hidden_dim) if int(args.hidden_dim) > 0 else None,
    ).to(device)
    verifier, verifier_allowed_chars, verifier_max_choice_chars = load_verifier(
        args,
        d_state=int(qtrm_model.d_state),
        device=device,
    )
    if verifier_max_choice_chars:
        args.verifier_max_choice_chars = int(verifier_max_choice_chars)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_exposed = -1.0
    best_selected = -1.0
    for epoch in range(1, int(args.epochs) + 1):
        train = train_epoch(
            qtrm_model=qtrm_model,
            tokenizer=tokenizer,
            selector=selector,
            rows=train_rows,
            allowed_chars=allowed_chars,
            optimizer=optimizer,
            args=args,
            device=device,
        )
        eval_summary, eval_records = evaluate(
            qtrm_model=qtrm_model,
            tokenizer=tokenizer,
            selector=selector,
            verifier=verifier,
            verifier_allowed_chars=verifier_allowed_chars,
            rows=eval_rows,
            allowed_chars=allowed_chars,
            args=args,
            device=device,
        )
        record = {"epoch": epoch, "train": train, "eval": eval_summary}
        history.append(record)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train["loss"],
                    "eval_accuracy": eval_summary["accuracy"],
                    "pool_oracle_coverage": eval_summary["pool_oracle_coverage"],
                    "exposed_oracle_coverage": eval_summary["exposed_oracle_coverage"],
                    "by_family": eval_summary["by_family"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        exposed = float(eval_summary["exposed_oracle_coverage"])
        selected = float(eval_summary["accuracy"])
        if exposed > best_exposed or (exposed == best_exposed and selected >= best_selected):
            best_exposed = exposed
            best_selected = selected
            torch.save(
                {
                    "selector": selector.state_dict(),
                    "args": vars(args),
                    "allowed_chars": allowed_chars,
                    "load_stats": load_stats,
                    "epoch": epoch,
                    "eval": eval_summary,
                },
                out_dir / "best_candidate_pool_selector.pt",
            )
            (out_dir / "best_records.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in eval_records),
                encoding="utf-8",
            )

    summary = {
        "best_exposed_oracle_coverage": best_exposed,
        "best_selected_accuracy": best_selected,
        "best_epoch": max(
            history,
            key=lambda item: (item["eval"]["exposed_oracle_coverage"], item["eval"]["accuracy"]),
        )["epoch"]
        if history
        else 0,
        "history": history,
        "load_stats": load_stats,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "plain_language_read": (
            "Deprecated scaffold: this tests an upper bound with a hand-built typed pool. "
            "Final-only Stage59 work must learn the typed working table in the answer path."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
