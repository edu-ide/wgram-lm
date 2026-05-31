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


FIELD_NAMES = ("scalar_coeff", "scalar_offset", "scalar_residual")


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


def operand_target_classes(
    row: dict[str, Any],
    *,
    scalar_vocab_size: int,
) -> tuple[int, int, int]:
    coeff = int(row["scalar_coeff"]) + 1
    offset = int(row["subtract_offset"]) + 1
    residual = int(row["scalar_initial_residual"]) + 1
    values = (coeff, offset, residual)
    if any(value <= 0 or value >= int(scalar_vocab_size) for value in values):
        raise ValueError(f"operand class out of range: {values}")
    return values


class PromptOperandBinder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        scalar_vocab_size: int,
        fields: tuple[str, ...] = FIELD_NAMES,
    ):
        super().__init__()
        self.fields = tuple(fields)
        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        self.queries = nn.Parameter(
            torch.randn(len(self.fields), int(hidden_dim)) * 0.02
        )
        self.heads = nn.ModuleDict(
            {
                field: nn.Linear(int(hidden_dim), int(scalar_vocab_size))
                for field in self.fields
            }
        )

    def forward(self, states, attention_mask=None):
        hidden = torch.tanh(self.input_proj(states.float()))
        scores = torch.einsum("bsh,fh->bfs", hidden, self.queries)
        scores = scores / math.sqrt(max(1, hidden.shape[-1]))
        if attention_mask is not None:
            mask = attention_mask.to(device=scores.device).bool()
            scores = scores.masked_fill(~mask[:, None, :], -1.0e4)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bfs,bsh->bfh", weights, hidden)
        return {
            field: self.heads[field](pooled[:, index, :])
            for index, field in enumerate(self.fields)
        }


class PromptOperandTransformerBinder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        scalar_vocab_size: int,
        fields: tuple[str, ...] = FIELD_NAMES,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
        max_positions: int = 512,
    ):
        super().__init__()
        self.fields = tuple(fields)
        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        self.position_embed = nn.Embedding(int(max_positions), int(hidden_dim))
        self.field_queries = nn.Parameter(
            torch.randn(len(self.fields), int(hidden_dim)) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(heads),
            dim_feedforward=int(hidden_dim) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(layers))
        self.heads = nn.ModuleDict(
            {
                field: nn.Sequential(
                    nn.LayerNorm(int(hidden_dim)),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.GELU(),
                    nn.Linear(int(hidden_dim), int(scalar_vocab_size)),
                )
                for field in self.fields
            }
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
        queries = self.field_queries.unsqueeze(0).expand(batch, -1, -1)
        sequence = torch.cat([queries, hidden], dim=1)
        padding_mask = None
        if attention_mask is not None:
            prompt_mask = attention_mask.to(device=sequence.device).bool()
            query_mask = torch.ones(
                batch,
                len(self.fields),
                dtype=torch.bool,
                device=sequence.device,
            )
            full_mask = torch.cat([query_mask, prompt_mask], dim=1)
            padding_mask = ~full_mask
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)
        field_states = encoded[:, : len(self.fields), :]
        return {
            field: self.heads[field](field_states[:, index, :])
            for index, field in enumerate(self.fields)
        }


def checkpoint_payload(
    *,
    model,
    token_embed,
    input_dim: int,
    hidden_dim: int,
    scalar_vocab_size: int,
    input_source: str,
    binder_kind: str,
    step: int,
    eval_report: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "scalar_vocab_size": int(scalar_vocab_size),
        "input_source": str(input_source),
        "binder_kind": str(binder_kind),
        "step": int(step),
        "eval": eval_report,
    }
    if token_embed is not None:
        payload["token_embed"] = token_embed.state_dict()
    return payload


def batch_targets(rows: list[dict[str, Any]], *, scalar_vocab_size: int, device: str):
    values = [
        operand_target_classes(row, scalar_vocab_size=int(scalar_vocab_size))
        for row in rows
    ]
    return torch.tensor(values, dtype=torch.long, device=device)


def score_logits(logits: dict[str, Any], targets) -> dict[str, float]:
    hits = []
    result: dict[str, float] = {}
    for index, field in enumerate(FIELD_NAMES):
        pred = logits[field].detach().argmax(dim=-1)
        field_hits = pred == targets[:, index]
        hits.append(field_hits)
        result[f"{field}_acc"] = float(field_hits.float().mean().item())
    exact = torch.stack(hits, dim=1).all(dim=1)
    result["exact_acc"] = float(exact.float().mean().item())
    return result


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
    scalar_vocab_size: int,
    device: str,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    total_rows = 0
    correct = {field: 0 for field in FIELD_NAMES}
    exact_correct = 0
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            batch = rows[start : start + int(batch_size)]
            enc = tokenizer(
                [str(row["prompt"]) for row in batch],
                return_tensors="pt",
                truncation=True,
                max_length=int(max_length),
                padding=True,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(device)
            mask = enc.get("attention_mask").to(device)
            if str(input_source) == "token_embedding":
                states = token_embed(input_ids)
            else:
                donor_out = donor.encode_inputs(
                    enc["input_ids"],
                    attention_mask=enc.get("attention_mask"),
                    return_logits=False,
                )
                states = donor_out["text_states"].to(device=device)
                mask = donor_out["attention_mask"].to(device=device)
            targets = batch_targets(
                batch,
                scalar_vocab_size=int(scalar_vocab_size),
                device=device,
            )
            logits = model(states, mask)
            loss = sum(
                F.cross_entropy(logits[field], targets[:, index])
                for index, field in enumerate(FIELD_NAMES)
            )
            metrics = score_logits(logits, targets)
            total_loss += float(loss.item()) * len(batch)
            total_rows += len(batch)
            for field in FIELD_NAMES:
                correct[field] += int(round(metrics[f"{field}_acc"] * len(batch)))
            exact_correct += int(round(metrics["exact_acc"] * len(batch)))
    report = {
        "rows": float(total_rows),
        "loss": total_loss / max(1, total_rows),
        "exact_acc": float(exact_correct) / float(max(1, total_rows)),
    }
    for field in FIELD_NAMES:
        report[f"{field}_acc"] = float(correct[field]) / float(max(1, total_rows))
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a learned prompt operand binder over frozen donor hidden states."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--binder-kind",
        choices=["attention", "transformer"],
        default="attention",
    )
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--transformer-max-positions", type=int, default=512)
    parser.add_argument(
        "--input-source",
        choices=["donor_hidden", "token_embedding"],
        default="donor_hidden",
    )
    parser.add_argument("--token-embedding-dim", type=int, default=256)
    parser.add_argument("--scalar-vocab-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from wgram_lm.config import load_config
    from wgram_lm.qwen_donor import QwenDonorAdapter

    args = build_arg_parser().parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_rows = load_rows(args.train_jsonl, max_rows=int(args.max_train_rows))
    eval_rows = load_rows(args.eval_jsonl, max_rows=int(args.max_eval_rows))
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    donor = None
    token_embed = None
    if str(args.input_source) == "token_embedding":
        input_dim = int(args.token_embedding_dim)
        token_embed = torch.nn.Embedding(len(tokenizer), input_dim).to(device)
    else:
        donor = QwenDonorAdapter(cfg.donor)
        sample_enc = tokenizer(
            str(train_rows[0]["prompt"]),
            return_tensors="pt",
            truncation=True,
            max_length=int(args.max_length),
            padding=True,
            add_special_tokens=True,
        )
        sample_out = donor.encode_inputs(
            sample_enc["input_ids"],
            attention_mask=sample_enc.get("attention_mask"),
            return_logits=False,
        )
        input_dim = int(sample_out["text_states"].shape[-1])
    if str(args.binder_kind) == "transformer":
        model = PromptOperandTransformerBinder(
            input_dim=input_dim,
            hidden_dim=int(args.hidden_dim),
            scalar_vocab_size=int(args.scalar_vocab_size),
            layers=int(args.transformer_layers),
            heads=int(args.transformer_heads),
            dropout=float(args.transformer_dropout),
            max_positions=int(args.transformer_max_positions),
        ).to(device)
    else:
        model = PromptOperandBinder(
            input_dim=input_dim,
            hidden_dim=int(args.hidden_dim),
            scalar_vocab_size=int(args.scalar_vocab_size),
        ).to(device)
    opt_params = list(model.parameters())
    if token_embed is not None:
        opt_params.extend(token_embed.parameters())
    opt = torch.optim.AdamW(opt_params, lr=float(args.lr), weight_decay=0.01)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_exact = -1.0
    reports: list[dict[str, Any]] = []

    for step in range(1, int(args.steps) + 1):
        model.train()
        batch = random.choices(train_rows, k=int(args.batch_size))
        enc = tokenizer(
            [str(row["prompt"]) for row in batch],
            return_tensors="pt",
            truncation=True,
            max_length=int(args.max_length),
            padding=True,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)
        mask = enc.get("attention_mask").to(device)
        if str(args.input_source) == "token_embedding":
            states = token_embed(input_ids)
        else:
            with torch.no_grad():
                donor_out = donor.encode_inputs(
                    enc["input_ids"],
                    attention_mask=enc.get("attention_mask"),
                    return_logits=False,
                )
            states = donor_out["text_states"].to(device=device)
            mask = donor_out["attention_mask"].to(device=device)
        targets = batch_targets(
            batch,
            scalar_vocab_size=int(args.scalar_vocab_size),
            device=device,
        )
        logits = model(states, mask)
        loss = sum(
            F.cross_entropy(logits[field], targets[:, index])
            for index, field in enumerate(FIELD_NAMES)
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
                scalar_vocab_size=int(args.scalar_vocab_size),
                device=device,
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
                        scalar_vocab_size=int(args.scalar_vocab_size),
                        input_source=str(args.input_source),
                        binder_kind=str(args.binder_kind),
                        step=step,
                        eval_report=eval_report,
                    ),
                    out_dir / "best.pt",
                )
    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "input_source": str(args.input_source),
                "binder_kind": str(args.binder_kind),
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "best_exact_acc": best_exact,
                "reports": reports,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
