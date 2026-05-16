#!/usr/bin/env python3
"""Minimal recurrent latent-state to next-token readout reproduction.

This is a diagnostic reset for the QTRM LM-readout bottleneck. It does not
claim reasoning progress. It isolates whether a recurrent decoder can turn a
latent per-token state into stable autoregressive numeric tokens under greedy
rollout, with scheduled-sampling style on-policy pressure available.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


EOS_TOKEN = 10
BOS_TOKEN = 11
VOCAB_SIZE = 12


class ReadoutCase(NamedTuple):
    value: int
    answer: str
    token_ids: list[int]


def build_case(value: int, *, answer_len: int) -> ReadoutCase:
    modulus = 10**answer_len
    answer = f"{int(value) % modulus:0{answer_len}d}"
    token_ids = [int(ch) for ch in answer] + [EOS_TOKEN]
    return ReadoutCase(value=int(value), answer=answer, token_ids=token_ids)


def build_cases(
    *,
    count: int,
    answer_len: int,
    start: int,
    seed: int,
    randomize: bool,
) -> list[ReadoutCase]:
    rng = random.Random(seed)
    cases = []
    for index in range(count):
        if randomize:
            value = rng.randrange(10**answer_len)
        else:
            value = start + index
        cases.append(build_case(value, answer_len=answer_len))
    return cases


def make_latent_table(
    *,
    answer_len: int,
    latent_dim: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    table = torch.randn(answer_len + 1, VOCAB_SIZE, latent_dim, generator=generator)
    table = F.normalize(table, dim=-1)
    return table.to(device)


def cases_to_tensors(
    cases: list[ReadoutCase],
    *,
    latent_table: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.tensor([case.token_ids for case in cases], dtype=torch.long, device=device)
    positions = torch.arange(targets.size(1), device=device).unsqueeze(0).expand_as(targets)
    latents = latent_table[positions, targets]
    return latents, targets


class LatentReadoutDecoder(nn.Module):
    def __init__(self, *, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.token_embed = nn.Embedding(VOCAB_SIZE, hidden_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.cell = nn.GRUCell(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, VOCAB_SIZE)

    def forward(
        self,
        latents: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        scheduled_sampling_prob: float = 0.0,
        greedy: bool = False,
    ) -> torch.Tensor:
        batch, steps, _ = latents.shape
        h = latents.new_zeros(batch, self.cell.hidden_size)
        prev = torch.full((batch,), BOS_TOKEN, dtype=torch.long, device=latents.device)
        logits_by_step = []
        for step in range(steps):
            x = torch.cat(
                [self.token_embed(prev), self.latent_proj(latents[:, step])],
                dim=-1,
            )
            h = self.cell(x, h)
            logits = self.out(h)
            logits_by_step.append(logits)
            pred = logits.argmax(dim=-1)
            if greedy or targets is None:
                prev = pred
            elif scheduled_sampling_prob > 0.0:
                mask = torch.rand(batch, device=latents.device) < scheduled_sampling_prob
                prev = torch.where(mask, pred.detach(), targets[:, step])
            else:
                prev = targets[:, step]
        return torch.stack(logits_by_step, dim=1)


def sequence_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    token_acc = (pred == targets).float().mean().item()
    exact = (pred == targets).all(dim=1).float().mean().item()
    return {"token_acc": float(token_acc), "exact": float(exact)}


def decode_tokens(token_ids: list[int]) -> str:
    chars = []
    for token_id in token_ids:
        if token_id == EOS_TOKEN:
            break
        if 0 <= token_id <= 9:
            chars.append(str(token_id))
        elif token_id == BOS_TOKEN:
            chars.append("<bos>")
        else:
            chars.append("?")
    return "".join(chars)


def evaluate(
    model: LatentReadoutDecoder,
    cases: list[ReadoutCase],
    *,
    latent_table: torch.Tensor,
    device: torch.device,
    max_examples: int = 4,
) -> dict:
    model.eval()
    with torch.no_grad():
        latents, targets = cases_to_tensors(cases, latent_table=latent_table, device=device)
        teacher_logits = model(latents, targets, scheduled_sampling_prob=0.0)
        greedy_logits = model(latents, greedy=True)
    teacher = sequence_metrics(teacher_logits, targets)
    greedy = sequence_metrics(greedy_logits, targets)
    greedy_pred = greedy_logits.argmax(dim=-1).detach().cpu().tolist()
    target_cpu = targets.detach().cpu().tolist()
    examples = []
    for case, pred_ids, target_ids in zip(cases[:max_examples], greedy_pred, target_cpu):
        examples.append(
            {
                "answer": case.answer,
                "target_tokens": target_ids,
                "prediction": decode_tokens(pred_ids),
                "pred_tokens": pred_ids,
                "exact": pred_ids == target_ids,
            }
        )
    return {
        "teacher_forced_token_acc": teacher["token_acc"],
        "teacher_forced_exact": teacher["exact"],
        "greedy_token_acc": greedy["token_acc"],
        "greedy_exact": greedy["exact"],
        "greedy_examples": examples,
    }


def make_decision(metrics: dict[str, float], *, accept_greedy_exact: float) -> dict:
    reject_reasons = []
    if float(metrics.get("greedy_exact", 0.0)) < accept_greedy_exact:
        reject_reasons.append("greedy_exact_below_threshold")
    if (
        float(metrics.get("teacher_forced_token_acc", 0.0)) >= 0.95
        and float(metrics.get("greedy_exact", 0.0)) < accept_greedy_exact
    ):
        reject_reasons.append("teacher_forced_greedy_gap")
    return {
        "accepted": not reject_reasons,
        "reject_reasons": reject_reasons,
        "thresholds": {"accept_greedy_exact": float(accept_greedy_exact)},
    }


def run_experiment(
    *,
    train_steps: int,
    train_cases: int,
    eval_cases: int,
    answer_len: int,
    latent_dim: int,
    hidden_dim: int,
    scheduled_sampling_prob: float,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    accept_greedy_exact: float,
    log_every: int,
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    torch_device = torch.device(device)
    latent_table = make_latent_table(
        answer_len=answer_len,
        latent_dim=latent_dim,
        seed=seed + 17,
        device=torch_device,
    )
    train = build_cases(
        count=train_cases,
        answer_len=answer_len,
        start=400000,
        seed=seed + 23,
        randomize=True,
    )
    eval_rows = build_cases(
        count=eval_cases,
        answer_len=answer_len,
        start=600000,
        seed=seed + 29,
        randomize=False,
    )
    model = LatentReadoutDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim).to(torch_device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    last_loss = 0.0
    for step in range(1, train_steps + 1):
        model.train()
        batch = random.sample(train, k=min(batch_size, len(train)))
        latents, targets = cases_to_tensors(batch, latent_table=latent_table, device=torch_device)
        logits = model(
            latents,
            targets,
            scheduled_sampling_prob=scheduled_sampling_prob,
        )
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
        if log_every and (step == 1 or step % log_every == 0 or step == train_steps):
            print(json.dumps({"step": step, "loss": last_loss}, ensure_ascii=False))
    eval_report = evaluate(model, eval_rows, latent_table=latent_table, device=torch_device)
    metrics = {
        "last_loss": last_loss,
        "teacher_forced_token_acc": eval_report["teacher_forced_token_acc"],
        "teacher_forced_exact": eval_report["teacher_forced_exact"],
        "greedy_token_acc": eval_report["greedy_token_acc"],
        "greedy_exact": eval_report["greedy_exact"],
    }
    decision = make_decision(metrics, accept_greedy_exact=accept_greedy_exact)
    return {
        "target_level": "L1 minimal latent-readout reproduction",
        "major_bottleneck": "latent-state-to-autoregressive next-token synthesis",
        "prior_family": [
            "scheduled_sampling",
            "looped_language_models",
            "latent_thought_to_lm_logits",
        ],
        "train": {
            "train_steps": train_steps,
            "train_cases": train_cases,
            "eval_cases": eval_cases,
            "answer_len": answer_len,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "scheduled_sampling_prob": scheduled_sampling_prob,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
        },
        "metrics": metrics,
        "greedy_examples": eval_report["greedy_examples"],
        **decision,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal recurrent latent-state to next-token readout reproduction."
    )
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--train-cases", type=int, default=1024)
    parser.add_argument("--eval-cases", type=int, default=128)
    parser.add_argument("--answer-len", type=int, default=6)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--scheduled-sampling-prob", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--accept-greedy-exact", type=float, default=0.90)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_experiment(
        train_steps=args.train_steps,
        train_cases=args.train_cases,
        eval_cases=args.eval_cases,
        answer_len=args.answer_len,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        scheduled_sampling_prob=args.scheduled_sampling_prob,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        accept_greedy_exact=args.accept_greedy_exact,
        log_every=args.log_every,
    )
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
