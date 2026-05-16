#!/usr/bin/env python3
"""Reduced operation-order transition diagnostic for QTRM-native L6.

This probe is intentionally smaller than the full L6 model. It tests one
question only:

    Can a learned recurrent transition compose prompt operations in the
    requested order and emit the answer through normal LM logits?

It is diagnostic, not a final architecture claim. No donor, MemoryOS, RAG,
renderer, or symbolic answer path is used at inference.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


PAD = 0
BOS = 1
EOS = 2
FWD = 3
REV = 4
START = 5
ANS = 6
OP_BASE = 7

OP_SPECS: tuple[tuple[str, int], ...] = (
    ("noop", 0),
    ("add", 1),
    ("add", 3),
    ("add", 5),
    ("mul", 2),
    ("mul", 3),
    ("affine", 2),
    ("affine", 3),
)
FAMILY_TO_TOKEN = {"fwd": FWD, "rev": REV}


@dataclass(frozen=True)
class OrderCase:
    case_id: str
    family: str
    start: int
    op_ids: tuple[int, ...]
    answer: int
    modulus: int


@dataclass(frozen=True)
class OrderBatch:
    input_ids: torch.Tensor
    answer_targets: torch.Tensor
    trace_targets: torch.Tensor
    op_positions: torch.Tensor
    answer_positions: torch.Tensor


def value_base() -> int:
    return OP_BASE + len(OP_SPECS)


def vocab_size(modulus: int) -> int:
    return value_base() + int(modulus)


def op_token(op_id: int) -> int:
    return OP_BASE + int(op_id)


def value_token(value: int) -> int:
    return value_base() + int(value)


def token_value(token_id: int) -> int | None:
    value = int(token_id) - value_base()
    return value if value >= 0 else None


def apply_op(value: int, op_id: int, modulus: int) -> int:
    name, param = OP_SPECS[int(op_id)]
    if name == "noop":
        return int(value) % int(modulus)
    if name == "add":
        return (int(value) + int(param)) % int(modulus)
    if name == "mul":
        return (int(value) * int(param)) % int(modulus)
    if name == "affine":
        return ((int(value) * int(param)) + int(param + 1)) % int(modulus)
    raise ValueError(f"unknown op: {name}")


def apply_ops_for_family(
    start: int,
    op_ids: Sequence[int],
    *,
    family: str,
    modulus: int,
) -> int:
    if str(family) not in FAMILY_TO_TOKEN:
        raise ValueError(f"unsupported family: {family!r}")
    ordered_ops = tuple(op_ids) if str(family) == "fwd" else tuple(reversed(op_ids))
    value = int(start)
    for op_id in ordered_ops:
        value = apply_op(value, int(op_id), int(modulus))
    return int(value)


def make_case(
    *,
    case_id: str,
    family: str,
    start: int,
    op_ids: Sequence[int],
    modulus: int,
) -> OrderCase:
    return OrderCase(
        case_id=str(case_id),
        family=str(family),
        start=int(start),
        op_ids=tuple(int(op_id) for op_id in op_ids),
        answer=apply_ops_for_family(
            int(start),
            tuple(int(op_id) for op_id in op_ids),
            family=str(family),
            modulus=int(modulus),
        ),
        modulus=int(modulus),
    )


def parse_families(value: str) -> tuple[str, ...]:
    rows = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not rows:
        raise ValueError("at least one family is required")
    for family in rows:
        if family not in FAMILY_TO_TOKEN:
            raise ValueError(f"unsupported family: {family!r}")
    return rows


def build_cases(
    *,
    count: int,
    seed: int,
    program_len: int,
    modulus: int,
    families: Sequence[str] = ("fwd", "rev"),
) -> list[OrderCase]:
    rng = random.Random(int(seed))
    family_rows = tuple(str(family) for family in families)
    rows: list[OrderCase] = []
    for index in range(int(count)):
        family = family_rows[index % len(family_rows)]
        start = rng.randrange(int(modulus))
        op_ids = tuple(rng.randrange(1, len(OP_SPECS)) for _ in range(int(program_len)))
        rows.append(
            make_case(
                case_id=f"order-{seed}-{index:06d}",
                family=family,
                start=start,
                op_ids=op_ids,
                modulus=int(modulus),
            )
        )
    return rows


def case_prompt_tokens(case: OrderCase) -> list[int]:
    return (
        [BOS, FAMILY_TO_TOKEN[str(case.family)], START, value_token(case.start)]
        + [op_token(op_id) for op_id in case.op_ids]
        + [ANS]
    )


def case_full_tokens(case: OrderCase) -> list[int]:
    return case_prompt_tokens(case) + [value_token(case.answer), EOS]


def case_trace_tokens(case: OrderCase, *, modulus: int) -> list[int]:
    ordered_ops = case.op_ids if str(case.family) == "fwd" else tuple(reversed(case.op_ids))
    value = int(case.start)
    rows: list[int] = []
    for op_id in ordered_ops:
        value = apply_op(value, int(op_id), int(modulus))
        rows.append(value_token(value))
    return rows


def cases_to_batch(cases: Sequence[OrderCase], *, device: torch.device) -> OrderBatch:
    if not cases:
        raise ValueError("cases must not be empty")
    prompts = [case_prompt_tokens(case) for case in cases]
    prompt_len = len(prompts[0])
    for prompt in prompts:
        if len(prompt) != prompt_len:
            raise ValueError("all prompts must share a fixed length")
    input_ids = torch.tensor(prompts, dtype=torch.long, device=device)
    answer_targets = torch.tensor(
        [value_token(case.answer) for case in cases],
        dtype=torch.long,
        device=device,
    )
    trace_targets = torch.tensor(
        [
            case_trace_tokens(case, modulus=int(case.modulus))
            for case in cases
        ],
        dtype=torch.long,
        device=device,
    )
    op_positions = torch.arange(
        4,
        4 + len(cases[0].op_ids),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0).expand(len(cases), -1)
    answer_positions = torch.full(
        (len(cases),),
        prompt_len - 1,
        dtype=torch.long,
        device=device,
    )
    return OrderBatch(
        input_ids=input_ids,
        answer_targets=answer_targets,
        trace_targets=trace_targets,
        op_positions=op_positions,
        answer_positions=answer_positions,
    )


class OperationOrderTransitionLM(nn.Module):
    def __init__(
        self,
        *,
        vocab: int,
        max_seq_len: int,
        d_model: int,
        program_len: int,
        modulus: int | None = None,
        value_codec: str = "learned",
    ) -> None:
        super().__init__()
        self.vocab = int(vocab)
        self.max_seq_len = int(max_seq_len)
        self.program_len = int(program_len)
        self.modulus = int(modulus) if modulus is not None else max(1, int(vocab) - value_base())
        self.value_codec = str(value_codec)
        if self.value_codec not in {"learned", "circular"}:
            raise ValueError(f"unsupported value_codec: {value_codec!r}")
        self.token_embed = nn.Embedding(int(vocab), int(d_model))
        self.pos_embed = nn.Embedding(int(max_seq_len), int(d_model))
        self.input_norm = nn.LayerNorm(int(d_model))
        self.start_proj = nn.Linear(int(d_model), int(d_model))
        self.transition_cell = nn.GRUCell(2 * int(d_model), int(d_model))
        self.readout_norm = nn.LayerNorm(int(d_model))
        self.lm_head = nn.Linear(int(d_model), int(vocab), bias=False)
        self.value_feature_dim = 6
        if self.value_codec == "circular":
            self.value_code_proj = nn.Linear(self.value_feature_dim, int(d_model), bias=False)
            self.value_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.start_proj.weight)
        nn.init.zeros_(self.start_proj.bias)
        for name, parameter in self.transition_cell.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
        nn.init.xavier_uniform_(self.lm_head.weight)
        if hasattr(self, "value_code_proj"):
            nn.init.xavier_uniform_(self.value_code_proj.weight)

    def _value_features(self, device: torch.device) -> torch.Tensor:
        values = torch.arange(int(self.modulus), dtype=torch.float32, device=device)
        angle = (2.0 * math.pi * values) / float(max(1, int(self.modulus)))
        rows = []
        for frequency in (1.0, 2.0, 4.0):
            rows.append(torch.sin(angle * frequency))
            rows.append(torch.cos(angle * frequency))
        return torch.stack(rows, dim=-1)

    def _circular_value_embeddings(self, device: torch.device) -> torch.Tensor:
        if not hasattr(self, "value_code_proj"):
            raise RuntimeError("circular value embeddings requested for learned codec")
        return self.value_code_proj(self._value_features(device))

    def _token_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.token_embed(input_ids)
        if self.value_codec != "circular":
            return embeddings
        value_mask = (input_ids >= value_base()) & (
            input_ids < value_base() + int(self.modulus)
        )
        if not bool(value_mask.any()):
            return embeddings
        value_ids = (input_ids[value_mask] - value_base()).clamp(0, int(self.modulus) - 1)
        value_embeddings = self._circular_value_embeddings(input_ids.device)
        embeddings = embeddings.clone()
        embeddings[value_mask] = value_embeddings[value_ids]
        return embeddings

    def _lm_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden)
        if self.value_codec != "circular":
            return logits
        value_embeddings = self._circular_value_embeddings(hidden.device)
        value_embeddings = F.normalize(value_embeddings, dim=-1)
        value_hidden = F.normalize(hidden, dim=-1)
        value_logits = torch.matmul(value_hidden, value_embeddings.t())
        value_logits = value_logits * self.value_logit_scale.exp().clamp(max=100.0)
        logits = logits.clone()
        logits[..., value_base() : value_base() + int(self.modulus)] = value_logits
        return logits

    def _embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = int(input_ids.shape[1])
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        return self.input_norm(self._token_embeddings(input_ids) + self.pos_embed(positions))

    def transition_trace(
        self,
        input_ids: torch.Tensor,
        op_positions: torch.Tensor,
        *,
        ablation: str = "none",
    ) -> torch.Tensor:
        encoded = self._embeddings(input_ids)
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        family_tokens = input_ids[:, 1]
        family_embedding = encoded[:, 1, :]
        if str(ablation) == "family_zero":
            family_embedding = torch.zeros_like(family_embedding)
        start_embedding = encoded[:, 3, :]
        state = torch.tanh(self.start_proj(start_embedding))
        if str(ablation) == "state_reset":
            return torch.zeros(
                input_ids.shape[0],
                op_positions.shape[1],
                state.shape[-1],
                dtype=state.dtype,
                device=state.device,
            )
        if str(ablation) == "transition_off":
            return state.unsqueeze(1).expand(-1, op_positions.shape[1], -1)

        op_states = encoded[
            batch_indices.unsqueeze(1),
            op_positions.to(device=input_ids.device),
        ]
        is_reverse = (family_tokens == REV).view(-1, 1, 1)
        if str(ablation) == "order_shuffle":
            is_reverse = torch.zeros_like(is_reverse)
        reversed_ops = torch.flip(op_states, dims=(1,))
        ordered_ops = torch.where(is_reverse, reversed_ops, op_states)
        states: list[torch.Tensor] = []
        for step in range(int(ordered_ops.shape[1])):
            transition_input = torch.cat([ordered_ops[:, step, :], family_embedding], dim=-1)
            state = self.transition_cell(transition_input, state)
            states.append(state)
        return torch.stack(states, dim=1)

    def transition_state(
        self,
        input_ids: torch.Tensor,
        op_positions: torch.Tensor,
        *,
        ablation: str = "none",
    ) -> torch.Tensor:
        trace = self.transition_trace(input_ids, op_positions, ablation=str(ablation))
        return trace[:, -1, :]

    def trace_logits(
        self,
        input_ids: torch.Tensor,
        op_positions: torch.Tensor,
        *,
        ablation: str = "none",
    ) -> torch.Tensor:
        states = self.transition_trace(input_ids, op_positions, ablation=str(ablation))
        return self._lm_logits(self.readout_norm(states))

    def forward(
        self,
        input_ids: torch.Tensor,
        op_positions: torch.Tensor,
        answer_positions: torch.Tensor,
        *,
        ablation: str = "none",
    ) -> torch.Tensor:
        encoded = self._embeddings(input_ids)
        state = self.transition_state(input_ids, op_positions, ablation=str(ablation))
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        answer_hidden = encoded[batch_indices, answer_positions.to(device=input_ids.device)]
        answer_hidden = self.readout_norm(answer_hidden + state)
        logits = self._lm_logits(encoded)
        logits = logits.clone()
        logits[batch_indices, answer_positions.to(device=input_ids.device)] = self._lm_logits(
            answer_hidden
        )
        return logits


def answer_logits(model: OperationOrderTransitionLM, batch: OrderBatch, *, ablation: str = "none") -> torch.Tensor:
    logits = model(
        batch.input_ids,
        batch.op_positions,
        batch.answer_positions,
        ablation=str(ablation),
    )
    rows = torch.arange(batch.input_ids.shape[0], device=batch.input_ids.device)
    return logits[rows, batch.answer_positions]


def train_step(
    model: OperationOrderTransitionLM,
    cases: Sequence[OrderCase],
    *,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
    trace_loss_weight: float = 0.0,
) -> float:
    batch_cases = [cases[rng.randrange(len(cases))] for _ in range(int(batch_size))]
    batch = cases_to_batch(batch_cases, device=device)
    logits = answer_logits(model, batch)
    loss = F.cross_entropy(logits, batch.answer_targets)
    if float(trace_loss_weight) > 0.0:
        trace_logits = model.trace_logits(batch.input_ids, batch.op_positions)
        trace_loss = F.cross_entropy(
            trace_logits.reshape(-1, trace_logits.shape[-1]),
            batch.trace_targets.reshape(-1),
        )
        loss = loss + float(trace_loss_weight) * trace_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().cpu().item())


@torch.no_grad()
def evaluate(
    model: OperationOrderTransitionLM,
    cases: Sequence[OrderCase],
    *,
    device: torch.device,
    batch_size: int,
    ablation: str = "none",
) -> dict[str, object]:
    model.eval()
    correct = 0
    total = 0
    valid = 0
    by_family: dict[str, dict[str, int | float]] = {}
    for offset in range(0, len(cases), int(batch_size)):
        batch_cases = list(cases[offset : offset + int(batch_size)])
        batch = cases_to_batch(batch_cases, device=device)
        predictions = answer_logits(model, batch, ablation=str(ablation)).argmax(dim=-1)
        for pred, target, case in zip(
            predictions.detach().cpu().tolist(),
            batch.answer_targets.detach().cpu().tolist(),
            batch_cases,
        ):
            family = str(case.family)
            if family not in by_family:
                by_family[family] = {"correct": 0, "valid": 0, "total": 0}
            is_valid = token_value(int(pred)) is not None
            is_correct = int(pred) == int(target)
            correct += int(is_correct)
            valid += int(is_valid)
            total += 1
            by_family[family]["correct"] = int(by_family[family]["correct"]) + int(is_correct)
            by_family[family]["valid"] = int(by_family[family]["valid"]) + int(is_valid)
            by_family[family]["total"] = int(by_family[family]["total"]) + 1
    family_metrics: dict[str, dict[str, float | int]] = {}
    for family, row in by_family.items():
        row_total = max(1, int(row["total"]))
        family_metrics[family] = {
            "correct": int(row["correct"]),
            "valid": int(row["valid"]),
            "total": int(row["total"]),
            "generation_exact": float(row["correct"]) / row_total,
            "generation_format_valid": float(row["valid"]) / row_total,
        }
    return {
        "ablation": str(ablation),
        "cases": int(total),
        "generation_exact": float(correct) / max(1, total),
        "generation_format_valid": float(valid) / max(1, total),
        "by_family": family_metrics,
    }


def make_decision(
    eval_metrics: dict[str, dict[str, object]],
    *,
    accept_min_exact: float,
    accept_min_transition_drop: float,
    accept_min_order_drop: float,
) -> tuple[bool, list[str], dict[str, float]]:
    full = float(eval_metrics["full"]["generation_exact"])
    transition_off = float(eval_metrics["transition_off"]["generation_exact"])
    order_shuffle = float(eval_metrics["order_shuffle"]["generation_exact"])
    transition_drop = full - transition_off
    order_drop = full - order_shuffle
    decisive = {
        "full_generation_exact": full,
        "transition_off_generation_exact": transition_off,
        "order_shuffle_generation_exact": order_shuffle,
        "state_reset_generation_exact": float(eval_metrics["state_reset"]["generation_exact"]),
        "family_zero_generation_exact": float(eval_metrics["family_zero"]["generation_exact"]),
        "full_minus_transition_off": transition_drop,
        "full_minus_order_shuffle": order_drop,
    }
    reasons: list[str] = []
    if full < float(accept_min_exact):
        reasons.append("full_exact_below_threshold")
    if transition_drop < float(accept_min_transition_drop):
        reasons.append("transition_drop_below_threshold")
    if order_drop < float(accept_min_order_drop):
        reasons.append("order_drop_below_threshold")
    return not reasons, reasons, decisive


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--train-cases", type=int, default=4096)
    parser.add_argument("--eval-cases", type=int, default=512)
    parser.add_argument("--program-len", type=int, default=6)
    parser.add_argument("--modulus", type=int, default=32)
    parser.add_argument("--families", default="fwd,rev")
    parser.add_argument("--eval-families", default="")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--value-codec", choices=("learned", "circular"), default="learned")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--trace-loss-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=337)
    parser.add_argument("--eval-seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--accept-min-exact", type=float, default=0.70)
    parser.add_argument("--accept-min-transition-drop", type=float, default=0.10)
    parser.add_argument("--accept-min-order-drop", type=float, default=0.10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(str(args.device))
    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    families = parse_families(str(args.families))
    eval_families = parse_families(str(args.eval_families)) if str(args.eval_families) else families
    train_cases = build_cases(
        count=int(args.train_cases),
        seed=int(args.seed),
        program_len=int(args.program_len),
        modulus=int(args.modulus),
        families=families,
    )
    eval_cases = build_cases(
        count=int(args.eval_cases),
        seed=int(args.eval_seed),
        program_len=int(args.program_len),
        modulus=int(args.modulus),
        families=eval_families,
    )
    model = OperationOrderTransitionLM(
        vocab=vocab_size(int(args.modulus)),
        max_seq_len=len(case_prompt_tokens(train_cases[0])),
        d_model=int(args.d_model),
        program_len=int(args.program_len),
        modulus=int(args.modulus),
        value_codec=str(args.value_codec),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    rng = random.Random(int(args.seed) + 17)
    last_loss = 0.0
    model.train()
    for step in range(1, int(args.steps) + 1):
        model.train()
        last_loss = train_step(
            model,
            train_cases,
            batch_size=int(args.batch_size),
            device=device,
            optimizer=optimizer,
            rng=rng,
            trace_loss_weight=float(args.trace_loss_weight),
        )
        if int(args.log_every) > 0 and (step == 1 or step % int(args.log_every) == 0):
            print(json.dumps({"step": step, "loss": last_loss, "lr": float(args.lr)}))
    eval_metrics = {
        "full": evaluate(
            model,
            eval_cases,
            device=device,
            batch_size=int(args.batch_size),
            ablation="none",
        ),
        "transition_off": evaluate(
            model,
            eval_cases,
            device=device,
            batch_size=int(args.batch_size),
            ablation="transition_off",
        ),
        "order_shuffle": evaluate(
            model,
            eval_cases,
            device=device,
            batch_size=int(args.batch_size),
            ablation="order_shuffle",
        ),
        "state_reset": evaluate(
            model,
            eval_cases,
            device=device,
            batch_size=int(args.batch_size),
            ablation="state_reset",
        ),
        "family_zero": evaluate(
            model,
            eval_cases,
            device=device,
            batch_size=int(args.batch_size),
            ablation="family_zero",
        ),
    }
    accepted, reject_reasons, decisive_metrics = make_decision(
        eval_metrics,
        accept_min_exact=float(args.accept_min_exact),
        accept_min_transition_drop=float(args.accept_min_transition_drop),
        accept_min_order_drop=float(args.accept_min_order_drop),
    )
    report = {
        "status": "complete",
        "target_level": "reduced operation-order recurrent transition diagnostic",
        "train": vars(args),
        "families": list(families),
        "eval_families": list(eval_families),
        "vocab_size": vocab_size(int(args.modulus)),
        "prompt_len": len(case_prompt_tokens(train_cases[0])),
        "last_loss": float(last_loss),
        "eval_metrics": eval_metrics,
        "accepted": bool(accepted),
        "decision": (
            "accepted_operation_order_transition_diagnostic"
            if bool(accepted)
            else "rejected"
        ),
        "reject_reasons": reject_reasons,
        "decisive_metrics": decisive_metrics,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "train": vars(args),
            "report": report,
        },
        out_dir / "last.pt",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if bool(accepted) else 1


if __name__ == "__main__":
    raise SystemExit(main())
