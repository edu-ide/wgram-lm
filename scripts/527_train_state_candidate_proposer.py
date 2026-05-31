#!/usr/bin/env python3
"""Train a learned answer-candidate proposer from QTRM thought states."""

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

PAD = "<pad>"
EOS = "<eos>"
IGNORE_INDEX = -100


def _load_script(name: str, filename: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


stage523 = _load_script("qtrm_stage523_for_527", "523_train_state_text_speaker.py")
stage524 = _load_script("qtrm_stage524_for_527", "524_train_state_choice_verifier.py")
stage525 = _load_script("qtrm_stage525_for_527", "525_eval_qwen_candidate_exposure.py")


def configure_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def row_choices(row: dict[str, Any]) -> list[str]:
    choices = row.get("choices")
    return [str(choice) for choice in choices] if isinstance(choices, list) else []


def build_candidate_char_vocab(rows: list[dict[str, Any]]) -> list[str]:
    chars = {PAD, EOS}
    for row in rows:
        for choice in row_choices(row):
            chars.update(str(choice))
    return [PAD, EOS] + sorted(char for char in chars if char not in {PAD, EOS})


def encode_candidate_targets(
    rows: list[dict[str, Any]],
    *,
    allowed_chars: list[str],
    max_candidates: int,
    max_candidate_chars: int,
    device: torch.device,
) -> torch.Tensor:
    char_index = {char: index for index, char in enumerate(allowed_chars)}
    pad_idx = char_index[PAD]
    eos_idx = char_index[EOS]
    encoded_rows: list[list[list[int]]] = []
    for row in rows:
        choices = row_choices(row)[: int(max_candidates)]
        encoded_choices: list[list[int]] = []
        for choice in choices:
            ids = [char_index[char] for char in str(choice)[: max(0, int(max_candidate_chars) - 1)]]
            ids.append(eos_idx)
            ids += [pad_idx] * (int(max_candidate_chars) - len(ids))
            encoded_choices.append(ids[: int(max_candidate_chars)])
        while len(encoded_choices) < int(max_candidates):
            encoded_choices.append([IGNORE_INDEX] * int(max_candidate_chars))
        encoded_rows.append(encoded_choices)
    return torch.tensor(encoded_rows, dtype=torch.long, device=device)


def decode_candidate_ids(
    ids: torch.Tensor,
    *,
    allowed_chars: list[str],
) -> list[list[str]]:
    out: list[list[str]] = []
    for row in ids.detach().cpu().tolist():
        row_texts: list[str] = []
        for candidate in row:
            chars: list[str] = []
            for index in candidate:
                if index < 0 or index >= len(allowed_chars):
                    continue
                char = allowed_chars[int(index)]
                if char in {PAD, EOS}:
                    break
                chars.append(char)
            row_texts.append("".join(chars).strip())
        out.append(stage525.dedupe_candidates(row_texts, max_candidates=len(row_texts)))
    return out


def oracle_coverage(candidates: list[str], row: dict[str, Any]) -> tuple[bool, int]:
    aliases = set(normalized_alias_set(answer_aliases(row)))
    for index, candidate in enumerate(candidates):
        if normalize_answer_text(candidate) in aliases:
            return True, int(index)
    return False, -1


class StateCandidateProposer(nn.Module):
    """Small parallel character proposer over K candidate slots."""

    def __init__(
        self,
        *,
        d_state: int,
        vocab_size: int,
        max_candidates: int,
        max_candidate_chars: int,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_state = int(d_state)
        self.max_candidates = int(max_candidates)
        self.max_candidate_chars = int(max_candidate_chars)
        hidden = int(hidden_dim or d_state * 2)
        self.slot_embed = nn.Embedding(self.max_candidates, self.d_state)
        self.pos_embed = nn.Embedding(self.max_candidate_chars, self.d_state)
        self.norm = nn.LayerNorm(self.d_state)
        self.adapter = nn.Sequential(
            nn.Linear(self.d_state, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.d_state),
            nn.GELU(),
        )
        self.out = nn.Linear(self.d_state, int(vocab_size))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.slot_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        for module in self.adapter:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, readout: torch.Tensor) -> torch.Tensor:
        if readout.ndim != 2:
            raise ValueError("readout must have shape (batch, d_state)")
        batch = int(readout.size(0))
        slot = self.slot_embed(torch.arange(self.max_candidates, device=readout.device)).view(
            1, self.max_candidates, 1, self.d_state
        )
        pos = self.pos_embed(torch.arange(self.max_candidate_chars, device=readout.device)).view(
            1, 1, self.max_candidate_chars, self.d_state
        )
        base = readout.view(batch, 1, 1, self.d_state) + slot + pos
        hidden = self.adapter(self.norm(base))
        return self.out(hidden)


class WorkspaceAwareCandidateProposer(StateCandidateProposer):
    """Candidate proposer that can read the thought path and Qwen workspace."""

    def __init__(
        self,
        *,
        d_state: int,
        vocab_size: int,
        max_candidates: int,
        max_candidate_chars: int,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
        n_heads: int = 4,
    ) -> None:
        super().__init__(
            d_state=d_state,
            vocab_size=vocab_size,
            max_candidates=max_candidates,
            max_candidate_chars=max_candidate_chars,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        if int(d_state) % int(n_heads) != 0:
            raise ValueError("d_state must be divisible by n_heads")
        self.trajectory_attn = nn.MultiheadAttention(
            int(d_state),
            num_heads=int(n_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.workspace_attn = nn.MultiheadAttention(
            int(d_state),
            num_heads=int(n_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.trajectory_norm = nn.LayerNorm(int(d_state))
        self.workspace_norm = nn.LayerNorm(int(d_state))

    @staticmethod
    def _key_padding_mask(attention_mask: torch.Tensor | None, memory: torch.Tensor) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        mask = attention_mask.to(device=memory.device, dtype=torch.bool).logical_not()
        if mask.ndim != 2 or mask.size(0) != memory.size(0) or mask.size(1) != memory.size(1):
            return None
        all_masked = mask.all(dim=1)
        if bool(all_masked.any()):
            mask = mask.clone()
            mask[all_masked] = False
        return mask

    def forward(
        self,
        readout: torch.Tensor,
        *,
        state_trajectory: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
        workspace_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if readout.ndim != 2:
            raise ValueError("readout must have shape (batch, d_state)")
        batch = int(readout.size(0))
        slot = self.slot_embed(torch.arange(self.max_candidates, device=readout.device)).view(
            1, self.max_candidates, 1, self.d_state
        )
        pos = self.pos_embed(torch.arange(self.max_candidate_chars, device=readout.device)).view(
            1, 1, self.max_candidate_chars, self.d_state
        )
        query = self.norm(readout.view(batch, 1, 1, self.d_state) + slot + pos)
        flat_query = query.view(batch, self.max_candidates * self.max_candidate_chars, self.d_state)
        if state_trajectory is not None:
            trajectory_context, _ = self.trajectory_attn(
                flat_query,
                state_trajectory.to(flat_query.dtype),
                state_trajectory.to(flat_query.dtype),
                need_weights=False,
            )
            flat_query = self.trajectory_norm(flat_query + trajectory_context)
        if workspace is not None:
            workspace_context, _ = self.workspace_attn(
                flat_query,
                workspace.to(flat_query.dtype),
                workspace.to(flat_query.dtype),
                key_padding_mask=self._key_padding_mask(workspace_attention_mask, workspace),
                need_weights=False,
            )
            flat_query = self.workspace_norm(flat_query + workspace_context)
        hidden = self.adapter(flat_query.view(batch, self.max_candidates, self.max_candidate_chars, self.d_state))
        return self.out(hidden)


def collate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row_choices(row)]


def train_epoch(
    *,
    wgram_model: Any,
    tokenizer: Any,
    proposer: StateCandidateProposer,
    rows: list[dict[str, Any]],
    allowed_chars: list[str],
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    wgram_model.eval()
    proposer.train()
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
            detach=True,
        )
        targets = encode_candidate_targets(
            batch,
            allowed_chars=allowed_chars,
            max_candidates=args.max_candidates,
            max_candidate_chars=args.max_candidate_chars,
            device=device,
        )
        if isinstance(proposer, WorkspaceAwareCandidateProposer):
            logits = proposer(
                context["readout"],
                state_trajectory=context.get("trajectory"),
                workspace=context.get("workspace"),
                workspace_attention_mask=context.get("workspace_attention_mask"),
            )
        else:
            logits = proposer(context["readout"])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proposer.parameters(), float(args.grad_clip))
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(batch)
        total_rows += len(batch)
    return {"loss": total_loss / max(1, total_rows), "seconds": time.time() - started}


@torch.no_grad()
def evaluate(
    *,
    wgram_model: Any,
    tokenizer: Any,
    proposer: StateCandidateProposer,
    verifier: stage524.ChoiceVerifier | None,
    verifier_allowed_chars: list[str] | None,
    rows: list[dict[str, Any]],
    allowed_chars: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    wgram_model.eval()
    proposer.eval()
    if verifier is not None:
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
            detach=True,
        )
        if isinstance(proposer, WorkspaceAwareCandidateProposer):
            logits = proposer(
                context["readout"],
                state_trajectory=context.get("trajectory"),
                workspace=context.get("workspace"),
                workspace_attention_mask=context.get("workspace_attention_mask"),
            )
        else:
            logits = proposer(context["readout"])
        predicted_ids = logits.argmax(dim=-1)
        batch_candidates = decode_candidate_ids(predicted_ids, allowed_chars=allowed_chars)

        selected_indices: list[int] = [-1] * len(batch)
        selected_texts: list[str] = [cands[0] if cands else "" for cands in batch_candidates]
        if verifier is not None and verifier_allowed_chars is not None:
            choice_ids, choice_mask = stage525.encode_candidate_strings(
                batch_candidates,
                allowed_chars=verifier_allowed_chars,
                max_choices=args.max_candidates,
                max_choice_chars=args.verifier_max_choice_chars,
                device=device,
            )
            scores = verifier(context["readout"], choice_ids, choice_mask)
            selected_indices = scores.argmax(dim=-1).detach().cpu().tolist()
            selected_texts = [
                batch_candidates[row_i][index] if 0 <= index < len(batch_candidates[row_i]) else ""
                for row_i, index in enumerate(selected_indices)
            ]
        for row, candidates, selected_index, selected in zip(batch, batch_candidates, selected_indices, selected_texts):
            oracle_hit, oracle_index = oracle_coverage(candidates, row)
            aliases = set(normalized_alias_set(answer_aliases(row)))
            exact = normalize_answer_text(selected) in aliases
            records.append(
                {
                    "id": stage523.row_id(row),
                    "task_family": row.get("task_family") or row.get("category") or "unknown",
                    "answer_kind": "generated_candidate",
                    "aliases": list(answer_aliases(row)),
                    "candidates": candidates,
                    "oracle_exact": bool(oracle_hit),
                    "oracle_index": int(oracle_index),
                    "selected": selected,
                    "selected_index": int(selected_index),
                    "normalized_selected": normalize_answer_text(selected),
                    "exact": bool(exact),
                    "selection_mode": "learned_candidate_proposer_plus_verifier"
                    if verifier is not None
                    else "learned_candidate_proposer_first_candidate",
                }
            )
    summary = summarize_records(records)
    summary.update(
        {
            "oracle_coverage": sum(1 for row in records if row["oracle_exact"]) / max(1, len(records)),
            "stage": "Stage59 learned candidate proposer",
            "plain_language_read": (
                "This asks whether the thought state can put plausible answer candidates on the table. "
                "Coverage measures imagination; selected accuracy measures the verifier plus copy path."
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--max-candidate-chars", type=int, default=24)
    parser.add_argument("--verifier-max-choice-chars", type=int, default=24)
    parser.add_argument("--proposer-context-mode", choices=("readout", "trajectory_workspace"), default="readout")
    parser.add_argument("--proposer-attn-heads", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1527)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
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
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    train_rows = stage523.load_jsonl(args.train_jsonl, limit=int(args.train_limit))
    eval_rows = stage523.load_jsonl(args.eval_jsonl, limit=int(args.eval_limit))
    allowed_chars = build_candidate_char_vocab([*train_rows, *eval_rows])
    wgram_model, tokenizer, load_stats = stage523.build_qtrm(args, device)
    proposer_cls = WorkspaceAwareCandidateProposer if args.proposer_context_mode == "trajectory_workspace" else StateCandidateProposer
    proposer_kwargs = {
        "d_state": int(wgram_model.d_state),
        "vocab_size": len(allowed_chars),
        "max_candidates": int(args.max_candidates),
        "max_candidate_chars": int(args.max_candidate_chars),
        "hidden_dim": int(args.hidden_dim) if int(args.hidden_dim) > 0 else None,
        "dropout": float(args.dropout),
    }
    if proposer_cls is WorkspaceAwareCandidateProposer:
        proposer_kwargs["n_heads"] = int(args.proposer_attn_heads)
    proposer = proposer_cls(**proposer_kwargs).to(device)
    verifier, verifier_allowed_chars, verifier_max_choice_chars = load_verifier(
        args,
        d_state=int(wgram_model.d_state),
        device=device,
    )
    if verifier_max_choice_chars:
        args.verifier_max_choice_chars = int(verifier_max_choice_chars)
    optimizer = torch.optim.AdamW(proposer.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_coverage = -1.0
    best_selected = -1.0
    for epoch in range(1, int(args.epochs) + 1):
        train = train_epoch(
            wgram_model=wgram_model,
            tokenizer=tokenizer,
            proposer=proposer,
            rows=train_rows,
            allowed_chars=allowed_chars,
            optimizer=optimizer,
            args=args,
            device=device,
        )
        eval_summary, eval_records = evaluate(
            wgram_model=wgram_model,
            tokenizer=tokenizer,
            proposer=proposer,
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
                    "oracle_coverage": eval_summary["oracle_coverage"],
                    "by_family": eval_summary["by_family"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        coverage = float(eval_summary["oracle_coverage"])
        selected = float(eval_summary["accuracy"])
        if coverage > best_coverage or (coverage == best_coverage and selected >= best_selected):
            best_coverage = coverage
            best_selected = selected
            torch.save(
                {
                    "proposer": proposer.state_dict(),
                    "args": vars(args),
                    "allowed_chars": allowed_chars,
                    "load_stats": load_stats,
                    "epoch": epoch,
                    "eval": eval_summary,
                },
                out_dir / "best_candidate_proposer.pt",
            )
            (out_dir / "best_records.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in eval_records),
                encoding="utf-8",
            )

    summary = {
        "best_oracle_coverage": best_coverage,
        "best_selected_accuracy": best_selected,
        "best_epoch": max(
            history,
            key=lambda item: (item["eval"]["oracle_coverage"], item["eval"]["accuracy"]),
        )["epoch"]
        if history
        else 0,
        "history": history,
        "load_stats": load_stats,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "plain_language_read": (
            "The candidate proposer is the model's imagination. If coverage stays low, "
            "the thought state still cannot expose answer candidates without a hand-coded helper."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
