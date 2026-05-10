#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn

from qtrm_mm.algorithmic_value_state import (
    parse_int_list_state,
    relative_source_slot_parity_ids,
    row_input_list,
    row_list_state_values,
    source_position_list_classes,
    token_numeric_source_slot_ids,
    token_numeric_source_slot_token_spans,
    token_numeric_source_slot_token_ids,
    token_numeric_value_ids,
)


def load_rows(path: str | Path, *, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(max_rows) > 0 and len(rows) >= int(max_rows):
                break
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def source_position_targets(
    row: dict[str, Any],
    *,
    max_slots: int,
    position_vocab_size: int,
    target_depth: int = 1,
) -> tuple[int, ...]:
    depth_targets = row.get("depth_targets")
    if not isinstance(depth_targets, dict):
        raise ValueError("row is missing depth_targets")
    raw = depth_targets.get(str(int(target_depth)))
    values = row_list_state_values(row, raw) if raw is not None else None
    if values is None:
        raise ValueError(f"target depth {target_depth} is not a list state")
    classes = source_position_list_classes(
        row,
        values,
        doubled=int(target_depth) > 1,
        max_slots=int(max_slots),
        slot_vocab_size=int(position_vocab_size),
    )
    if classes is None:
        raise ValueError("could not encode source-position targets")
    return tuple(int(value) for value in classes)


def numeric_value_ids(
    row: dict[str, Any],
    *,
    max_list_len: int,
    value_vocab_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    values = row_input_list(row)
    if values is None:
        raise ValueError("row has no input_list")
    ids: list[int] = []
    mask: list[int] = []
    for value in values[: int(max_list_len)]:
        class_id = int(value) + 1
        if class_id <= 0 or class_id >= int(value_vocab_size):
            raise ValueError(f"numeric value class out of range: {class_id}")
        ids.append(class_id)
        mask.append(1)
    while len(ids) < int(max_list_len):
        ids.append(0)
        mask.append(0)
    return tuple(ids), tuple(mask)


class PromptSourcePositionBinder(nn.Module):
    """Slot-query prompt binder for ordered source-position targets.

    This is a probe, not a production QTRM component. It checks whether the
    prompt token stream carries enough signal to bind list positions before the
    recursive state update is trained.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        max_slots: int,
        position_vocab_size: int,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        dropout: float = 0.0,
        max_positions: int = 512,
    ) -> None:
        super().__init__()
        self.max_slots = int(max_slots)
        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        self.position_embed = nn.Embedding(int(max_positions), int(hidden_dim))
        self.slot_queries = nn.Parameter(
            torch.randn(int(max_slots), int(hidden_dim)) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(transformer_heads),
            dim_feedforward=int(hidden_dim) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(transformer_layers))
        self.head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(position_vocab_size)),
        )

    def forward(self, states, attention_mask=None):
        hidden = self.input_proj(states.float())
        positions = torch.arange(
            hidden.shape[1],
            dtype=torch.long,
            device=hidden.device,
        ).clamp(max=self.position_embed.num_embeddings - 1)
        hidden = hidden + self.position_embed(positions).unsqueeze(0)
        batch = hidden.shape[0]
        queries = self.slot_queries.unsqueeze(0).expand(batch, -1, -1)
        sequence = torch.cat([queries, hidden], dim=1)
        padding_mask = None
        if attention_mask is not None:
            prompt_mask = attention_mask.to(device=sequence.device).bool()
            query_mask = torch.ones(
                batch,
                self.max_slots,
                dtype=torch.bool,
                device=sequence.device,
            )
            padding_mask = ~torch.cat([query_mask, prompt_mask], dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)
        return self.head(encoded[:, : self.max_slots, :])


def batch_targets(
    rows: list[dict[str, Any]],
    *,
    max_slots: int,
    position_vocab_size: int,
    target_depth: int,
    device: str,
):
    values = [
        source_position_targets(
            row,
            max_slots=int(max_slots),
            position_vocab_size=int(position_vocab_size),
            target_depth=int(target_depth),
        )
        for row in rows
    ]
    return torch.tensor(values, dtype=torch.long, device=device)


def score_logits(logits, targets) -> dict[str, float]:
    pred = logits.detach().argmax(dim=-1)
    hits = pred == targets
    content = targets > 0
    exact = hits.all(dim=-1)
    return {
        "slot_acc": float(hits.float().mean().item()),
        "content_slot_acc": float(hits[content].float().mean().item())
        if bool(content.any())
        else 0.0,
        "exact_acc": float(exact.float().mean().item()),
    }


def _states_from_batch(
    *,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    token_embed: Any,
    donor: Any,
    input_source: str,
    max_length: int,
    device: str,
    numeric_max_list_len: int = 5,
    numeric_value_vocab_size: int = 128,
):
    if str(input_source) == "numeric_value_embedding":
        ids_and_masks = [
            numeric_value_ids(
                row,
                max_list_len=int(numeric_max_list_len),
                value_vocab_size=int(numeric_value_vocab_size),
            )
            for row in rows
        ]
        ids = torch.tensor(
            [item[0] for item in ids_and_masks],
            dtype=torch.long,
            device=device,
        )
        mask = torch.tensor(
            [item[1] for item in ids_and_masks],
            dtype=torch.long,
            device=device,
        )
        return token_embed(ids), mask
    if str(input_source) == "relative_source_slot_parity":
        ids_and_masks = [
            relative_source_slot_parity_ids(
                row,
                max_list_len=int(numeric_max_list_len),
            )
            for row in rows
        ]
        ids = torch.tensor(
            [item[0] for item in ids_and_masks],
            dtype=torch.long,
            device=device,
        )
        mask = torch.tensor(
            [item[1] for item in ids_and_masks],
            dtype=torch.long,
            device=device,
        )
        return token_embed(ids), mask
    if str(input_source) == "token_numeric_source_slots":
        enc = tokenizer(
            [str(row["prompt"]) for row in rows],
            return_tensors="pt",
            truncation=True,
            max_length=int(max_length),
            padding=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offset_mapping = enc.get("offset_mapping")
        if offset_mapping is None:
            raise ValueError("tokenizer did not return offset_mapping")
        ids_and_masks = [
            token_numeric_source_slot_ids(
                row,
                offsets=offset_mapping[index].tolist(),
                max_list_len=int(numeric_max_list_len),
                value_vocab_size=int(numeric_value_vocab_size),
            )
            for index, row in enumerate(rows)
        ]
        ids = torch.tensor(
            [item[0] for item in ids_and_masks],
            dtype=torch.long,
            device=device,
        )
        mask = torch.tensor(
            [item[1] for item in ids_and_masks],
            dtype=torch.long,
            device=device,
        )
        return token_embed(ids), mask
    use_token_numeric = str(input_source) == "token_plus_numeric_value"
    enc = tokenizer(
        [str(row["prompt"]) for row in rows],
        return_tensors="pt",
        truncation=True,
        max_length=int(max_length),
        padding=True,
        add_special_tokens=True,
        return_offsets_mapping=use_token_numeric,
    )
    input_ids = enc["input_ids"].to(device)
    mask = enc.get("attention_mask").to(device)
    if str(input_source) == "token_embedding":
        return token_embed(input_ids), mask
    if use_token_numeric:
        offset_mapping = enc.get("offset_mapping")
        if offset_mapping is None:
            raise ValueError("tokenizer did not return offset_mapping")
        numeric_ids = torch.tensor(
            [
                token_numeric_value_ids(
                    row,
                    offsets=offset_mapping[index].tolist(),
                    value_vocab_size=int(numeric_value_vocab_size),
                )
                for index, row in enumerate(rows)
            ],
            dtype=torch.long,
            device=device,
        )
        return token_embed["token"](input_ids) + token_embed["numeric"](numeric_ids), mask
    with torch.no_grad():
        donor_out = donor.encode_inputs(
            enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            return_logits=False,
        )
    return (
        donor_out["text_states"].to(device=device),
        donor_out["attention_mask"].to(device=device),
    )


@torch.no_grad()
def evaluate(
    *,
    model,
    donor,
    token_embed,
    input_source: str,
    tokenizer,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    max_slots: int,
    position_vocab_size: int,
    target_depth: int,
    device: str,
    numeric_max_list_len: int = 5,
    numeric_value_vocab_size: int = 128,
) -> dict[str, float]:
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    total_rows = 0
    slot_hits = 0.0
    content_hits = 0.0
    exact_hits = 0.0
    for start in range(0, len(rows), int(batch_size)):
        batch = rows[start : start + int(batch_size)]
        states, mask = _states_from_batch(
            rows=batch,
            tokenizer=tokenizer,
            token_embed=token_embed,
            donor=donor,
            input_source=str(input_source),
            max_length=int(max_length),
            device=device,
            numeric_max_list_len=int(numeric_max_list_len),
            numeric_value_vocab_size=int(numeric_value_vocab_size),
        )
        targets = batch_targets(
            batch,
            max_slots=int(max_slots),
            position_vocab_size=int(position_vocab_size),
            target_depth=int(target_depth),
            device=device,
        )
        logits = model(states, mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        metrics = score_logits(logits, targets)
        total_loss += float(loss.item()) * len(batch)
        total_rows += len(batch)
        slot_hits += float(metrics["slot_acc"]) * len(batch)
        content_hits += float(metrics["content_slot_acc"]) * len(batch)
        exact_hits += float(metrics["exact_acc"]) * len(batch)
    denom = float(max(1, total_rows))
    return {
        "rows": float(total_rows),
        "loss": total_loss / denom,
        "slot_acc": slot_hits / denom,
        "content_slot_acc": content_hits / denom,
        "exact_acc": exact_hits / denom,
    }


def checkpoint_payload(
    *,
    model,
    token_embed,
    input_dim: int,
    hidden_dim: int,
    max_slots: int,
    position_vocab_size: int,
    input_source: str,
    step: int,
    eval_report: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "max_slots": int(max_slots),
        "position_vocab_size": int(position_vocab_size),
        "input_source": str(input_source),
        "step": int(step),
        "eval": eval_report,
    }
    if token_embed is not None:
        payload["token_embed"] = token_embed.state_dict()
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a prompt-token source-position binder probe. This isolates "
            "whether list source positions can be bound before recurrent QTRM "
            "state updates."
        )
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument(
        "--input-source",
        choices=[
            "token_embedding",
            "donor_hidden",
            "numeric_value_embedding",
            "relative_source_slot_parity",
            "token_plus_numeric_value",
            "token_numeric_source_slots",
        ],
        default="token_embedding",
    )
    parser.add_argument("--token-embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-positions", type=int, default=512)
    parser.add_argument("--max-slots", type=int, default=4)
    parser.add_argument("--position-vocab-size", type=int, default=8)
    parser.add_argument("--numeric-value-vocab-size", type=int, default=128)
    parser.add_argument("--numeric-max-list-len", type=int, default=5)
    parser.add_argument("--target-depth", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=320)
    return parser


def main() -> None:
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    args = build_arg_parser().parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_rows = load_rows(args.train_jsonl, max_rows=int(args.max_train_rows))
    eval_rows = load_rows(args.eval_jsonl, max_rows=int(args.max_eval_rows))
    tokenizer = None
    if str(args.input_source) != "numeric_value_embedding":
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_model_id,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    donor = None
    token_embed = None
    if str(args.input_source) in {
        "numeric_value_embedding",
        "relative_source_slot_parity",
        "token_numeric_source_slots",
    }:
        input_dim = int(args.token_embedding_dim)
        token_embed = nn.Embedding(int(args.numeric_value_vocab_size), input_dim).to(device)
    elif str(args.input_source) == "token_embedding":
        input_dim = int(args.token_embedding_dim)
        assert tokenizer is not None
        token_embed = nn.Embedding(len(tokenizer), input_dim).to(device)
    elif str(args.input_source) == "token_plus_numeric_value":
        input_dim = int(args.token_embedding_dim)
        assert tokenizer is not None
        token_embed = nn.ModuleDict(
            {
                "token": nn.Embedding(len(tokenizer), input_dim),
                "numeric": nn.Embedding(
                    int(args.numeric_value_vocab_size),
                    input_dim,
                ),
            }
        ).to(device)
    else:
        from qtrm_mm.config import load_config
        from qtrm_mm.qwen_donor import QwenDonorAdapter

        if not args.config:
            raise ValueError("--config is required for donor_hidden input")
        cfg = load_config(args.config)
        donor = QwenDonorAdapter(cfg.donor)
        sample_states, _ = _states_from_batch(
            rows=[train_rows[0]],
            tokenizer=tokenizer,
            token_embed=None,
            donor=donor,
            input_source="donor_hidden",
            max_length=int(args.max_length),
            device=device,
        )
        input_dim = int(sample_states.shape[-1])

    model = PromptSourcePositionBinder(
        input_dim=input_dim,
        hidden_dim=int(args.hidden_dim),
        max_slots=int(args.max_slots),
        position_vocab_size=int(args.position_vocab_size),
        transformer_layers=int(args.transformer_layers),
        transformer_heads=int(args.transformer_heads),
        dropout=float(args.dropout),
        max_positions=int(args.max_positions),
    ).to(device)
    opt_params = list(model.parameters())
    if token_embed is not None:
        opt_params.extend(token_embed.parameters())
    opt = torch.optim.AdamW(opt_params, lr=float(args.lr), weight_decay=0.01)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    best_exact = -1.0

    for step in range(1, int(args.steps) + 1):
        model.train()
        batch = random.choices(train_rows, k=int(args.batch_size))
        states, mask = _states_from_batch(
            rows=batch,
            tokenizer=tokenizer,
            token_embed=token_embed,
            donor=donor,
            input_source=str(args.input_source),
            max_length=int(args.max_length),
            device=device,
            numeric_max_list_len=int(args.numeric_max_list_len),
            numeric_value_vocab_size=int(args.numeric_value_vocab_size),
        )
        targets = batch_targets(
            batch,
            max_slots=int(args.max_slots),
            position_vocab_size=int(args.position_vocab_size),
            target_depth=int(args.target_depth),
            device=device,
        )
        logits = model(states, mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % int(args.log_every) == 0:
            metrics = score_logits(logits, targets)
            print(
                json.dumps(
                    {"step": step, "loss": float(loss.item()), **metrics},
                    ensure_ascii=False,
                )
            )
        if step == int(args.steps) or step % int(args.eval_every) == 0:
            eval_report = evaluate(
                model=model,
                donor=donor,
                token_embed=token_embed,
                input_source=str(args.input_source),
                tokenizer=tokenizer,
                rows=eval_rows,
                batch_size=int(args.eval_batch_size),
                max_length=int(args.max_length),
                max_slots=int(args.max_slots),
                position_vocab_size=int(args.position_vocab_size),
                target_depth=int(args.target_depth),
                device=device,
                numeric_max_list_len=int(args.numeric_max_list_len),
                numeric_value_vocab_size=int(args.numeric_value_vocab_size),
            )
            eval_report["step"] = step
            reports.append(eval_report)
            print(json.dumps({"eval": eval_report}, ensure_ascii=False))
            if float(eval_report["exact_acc"]) > best_exact:
                best_exact = float(eval_report["exact_acc"])
                torch.save(
                    checkpoint_payload(
                        model=model,
                        token_embed=token_embed,
                        input_dim=input_dim,
                        hidden_dim=int(args.hidden_dim),
                        max_slots=int(args.max_slots),
                        position_vocab_size=int(args.position_vocab_size),
                        input_source=str(args.input_source),
                        step=step,
                        eval_report=eval_report,
                    ),
                    out_dir / "best.pt",
                )

    decision = "accepted_l1" if best_exact >= 0.90 else "rejected"
    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "decision": decision,
                "accepted": decision == "accepted_l1",
                "target_level": "L1 scaffold",
                "major_bottleneck": "prompt-token source-position binding",
                "input_source": str(args.input_source),
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "best_exact_acc": best_exact,
                "reports": reports,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
