#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


PAD_VALUE = -1
FILTER_OP = 1
DOUBLE_OP = 2
HOLD_OP = 3


@dataclass(frozen=True)
class OrderedListCase:
    case_id: str
    values: tuple[int, ...]
    depth_targets: dict[int, tuple[int, ...]]


def _pad_values(values: Iterable[int], *, max_output_len: int) -> tuple[int, ...]:
    clipped = list(values)[: int(max_output_len)]
    clipped.extend([PAD_VALUE] * (int(max_output_len) - len(clipped)))
    return tuple(int(value) for value in clipped)


def case_from_values(
    case_id: str,
    values: Iterable[int],
    *,
    max_output_len: int,
    max_depth: int = 4,
) -> OrderedListCase:
    raw = tuple(int(value) for value in values)
    filtered = tuple(value for value in raw if value % 2 == 0)
    doubled = tuple(2 * value for value in filtered)
    depth_targets = {
        0: _pad_values(raw, max_output_len=int(max_output_len)),
        1: _pad_values(filtered, max_output_len=int(max_output_len)),
        2: _pad_values(doubled, max_output_len=int(max_output_len)),
    }
    for depth in range(3, int(max_depth) + 1):
        depth_targets[depth] = depth_targets[2]
    return OrderedListCase(
        case_id=str(case_id),
        values=raw,
        depth_targets=depth_targets,
    )


def build_cases(
    *,
    count: int,
    seed: int,
    list_len: int,
    max_output_len: int,
    value_modulus: int,
    max_depth: int,
) -> list[OrderedListCase]:
    rng = random.Random(int(seed))
    cases: list[OrderedListCase] = []
    for index in range(int(count)):
        values = tuple(rng.randrange(int(value_modulus)) for _ in range(int(list_len)))
        cases.append(
            case_from_values(
                f"ordered-list-{seed}-{index:06d}",
                values,
                max_output_len=int(max_output_len),
                max_depth=int(max_depth),
            )
        )
    return cases


def _value_to_class(value: int) -> int:
    if int(value) == PAD_VALUE:
        return 0
    return int(value) + 1


def cases_to_batch(
    cases: list[OrderedListCase],
    *,
    max_input_len: int,
    max_output_len: int,
    max_depth: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.zeros(
        (len(cases), int(max_input_len)),
        dtype=torch.long,
        device=device,
    )
    targets = torch.zeros(
        (len(cases), int(max_depth) + 1, int(max_output_len)),
        dtype=torch.long,
        device=device,
    )
    for row_idx, case in enumerate(cases):
        input_values = list(case.values)[: int(max_input_len)]
        for col_idx, value in enumerate(input_values):
            values[row_idx, col_idx] = _value_to_class(int(value))
        for depth in range(int(max_depth) + 1):
            target_values = case.depth_targets[min(int(depth), 2)]
            if depth in case.depth_targets:
                target_values = case.depth_targets[depth]
            for slot_idx, value in enumerate(target_values[: int(max_output_len)]):
                targets[row_idx, depth, slot_idx] = _value_to_class(int(value))
    return values, targets


class OrderedListStateProbe(nn.Module):
    """Tiny learned ordered-slot recurrence for list select/map/hold.

    This is an L1 scaffold. It is intentionally donorless and does not claim to
    be a canonical LLM path; it tests whether a learned recurrent slot state can
    preserve order and compose filter -> double before QTRM integration.
    """

    def __init__(
        self,
        *,
        value_vocab_size: int,
        max_output_len: int,
        d_model: int = 96,
        num_heads: int = 4,
        ff_dim: int = 192,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.value_vocab_size = int(value_vocab_size)
        self.max_output_len = int(max_output_len)
        self.value_embed = nn.Embedding(self.value_vocab_size, int(d_model))
        self.position_embed = nn.Embedding(self.max_output_len, int(d_model))
        self.op_embed = nn.Embedding(HOLD_OP + 1, int(d_model))
        self.input_norm = nn.LayerNorm(int(d_model))
        self.update = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.output_norm = nn.LayerNorm(int(d_model))
        self.output = nn.Linear(int(d_model), self.value_vocab_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.value_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.op_embed.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _op_id_for_step(self, step_index: int) -> int:
        if int(step_index) == 0:
            return FILTER_OP
        if int(step_index) == 1:
            return DOUBLE_OP
        return HOLD_OP

    def forward(
        self,
        value_ids: torch.Tensor,
        *,
        max_depth: int,
        state_reset_each_step: bool = False,
        op_zero: bool = False,
        order_shuffle: bool = False,
    ) -> torch.Tensor:
        if value_ids.ndim != 2:
            raise ValueError("value_ids must have shape [batch, slots]")
        b, slots = value_ids.shape
        if int(slots) < self.max_output_len:
            pad = value_ids.new_zeros((b, self.max_output_len - int(slots)))
            value_ids = torch.cat([value_ids, pad], dim=1)
        value_ids = value_ids[:, : self.max_output_len]
        positions = torch.arange(
            self.max_output_len,
            device=value_ids.device,
            dtype=torch.long,
        )
        initial = self.value_embed(value_ids) + self.position_embed(positions).unsqueeze(0)
        state = self.input_norm(initial)
        logits = [self.output(self.output_norm(state))]
        generator = None
        if order_shuffle:
            generator = torch.Generator(device=value_ids.device)
            generator.manual_seed(3915)
        for step_index in range(int(max_depth)):
            base_state = initial if bool(state_reset_each_step) else state
            if bool(order_shuffle):
                shuffled = []
                for _ in range(b):
                    assert generator is not None
                    perm = torch.randperm(
                        self.max_output_len,
                        generator=generator,
                        device=value_ids.device,
                    )
                    shuffled.append(base_state[_, perm, :])
                base_state = torch.stack(shuffled, dim=0)
            op_id = HOLD_OP if bool(op_zero) else self._op_id_for_step(step_index)
            op = torch.full(
                (b, self.max_output_len),
                int(op_id),
                dtype=torch.long,
                device=value_ids.device,
            )
            update_input = base_state + self.op_embed(op)
            state = self.update(update_input)
            state = self.input_norm(base_state + state)
            logits.append(self.output(self.output_norm(state)))
        return torch.stack(logits, dim=1)


def _sequence_exact(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred == target).all(dim=-1)


@torch.no_grad()
def evaluate_cases(
    model: OrderedListStateProbe,
    cases: list[OrderedListCase],
    args: argparse.Namespace,
    *,
    ablation: str = "none",
) -> dict[str, float | int | str]:
    model.eval()
    values, targets = cases_to_batch(
        cases,
        max_input_len=int(args.list_len),
        max_output_len=int(args.max_output_len),
        max_depth=int(args.max_depth),
        device=args.device,
    )
    logits = model(
        values,
        max_depth=int(args.max_depth),
        state_reset_each_step=(ablation == "state_reset"),
        op_zero=(ablation == "op_zero"),
        order_shuffle=(ablation == "order_shuffle"),
    )
    pred = logits.argmax(dim=-1)
    final_target = targets[:, min(2, int(args.max_depth)), :]
    metrics: dict[str, float | int | str] = {
        "cases": len(cases),
        "ablation": ablation,
    }
    for depth in parse_depths(args.depths):
        clipped = min(int(depth), int(args.max_depth))
        metrics[f"depth{depth}_state_exact"] = float(
            _sequence_exact(pred[:, clipped, :], targets[:, clipped, :])
            .float()
            .mean()
            .item()
        )
        metrics[f"depth{depth}_final_exact"] = float(
            _sequence_exact(pred[:, clipped, :], final_target)
            .float()
            .mean()
            .item()
        )
    return metrics


def loss_for_batch(
    model: OrderedListStateProbe,
    values: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_depth: int,
) -> torch.Tensor:
    logits = model(values, max_depth=int(max_depth))
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )


def parse_depths(text: str | Iterable[int]) -> list[int]:
    if isinstance(text, str):
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    return [int(value) for value in text]


def decide_gate(
    eval_metrics: dict[str, float | int | str],
    ablations: dict[str, dict[str, float | int | str]],
    args: argparse.Namespace,
) -> str:
    depth_values = parse_depths(args.depths)
    max_depth = max(depth_values)
    shallow_depth = min(depth for depth in depth_values if depth > 0)
    full_exact = float(eval_metrics[f"depth{max_depth}_final_exact"])
    shallow_exact = float(eval_metrics[f"depth{shallow_depth}_final_exact"])
    ablation_names = ["state_reset", "order_shuffle"]
    if bool(getattr(args, "require_op_ablation", False)):
        ablation_names.append("op_zero")
    worst_ablation = max(
        float(ablations[name][f"depth{max_depth}_final_exact"])
        for name in ablation_names
    )
    if (
        full_exact >= float(args.accept_min_final_exact)
        and (full_exact - shallow_exact) >= float(args.accept_min_depth_gain)
        and (full_exact - worst_ablation) >= float(args.accept_min_ablation_drop)
    ):
        return "accepted_l1"
    return "rejected"


def train_probe(args: argparse.Namespace) -> dict[str, object]:
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    train_cases = build_cases(
        count=int(args.train_cases),
        seed=int(args.seed),
        list_len=int(args.list_len),
        max_output_len=int(args.max_output_len),
        value_modulus=int(args.value_modulus),
        max_depth=int(args.max_depth),
    )
    eval_cases = build_cases(
        count=int(args.eval_cases),
        seed=int(args.eval_seed),
        list_len=int(args.list_len),
        max_output_len=int(args.max_output_len),
        value_modulus=int(args.value_modulus),
        max_depth=int(args.max_depth),
    )
    value_vocab_size = 2 * int(args.value_modulus) + 1
    model = OrderedListStateProbe(
        value_vocab_size=value_vocab_size,
        max_output_len=int(args.max_output_len),
        d_model=int(args.d_model),
        num_heads=int(args.num_heads),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    last_loss = 0.0
    for step in range(int(args.steps)):
        model.train()
        indices = [
            ((step * int(args.batch_size)) + offset) % len(train_cases)
            for offset in range(int(args.batch_size))
        ]
        batch_cases = [train_cases[index] for index in indices]
        values, targets = cases_to_batch(
            batch_cases,
            max_input_len=int(args.list_len),
            max_output_len=int(args.max_output_len),
            max_depth=int(args.max_depth),
            device=args.device,
        )
        loss = loss_for_batch(
            model,
            values,
            targets,
            max_depth=int(args.max_depth),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        last_loss = float(loss.item())
        if int(args.log_every) > 0 and (step + 1) % int(args.log_every) == 0:
            print(f"step={step + 1} loss={last_loss:.4f}")

    train_metrics = evaluate_cases(model, train_cases, args)
    eval_metrics = evaluate_cases(model, eval_cases, args)
    ablations = {
        name: evaluate_cases(model, eval_cases, args, ablation=name)
        for name in ("state_reset", "op_zero", "order_shuffle")
    }
    decision = decide_gate(eval_metrics, ablations, args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "args": vars(args),
        "last_loss": last_loss,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "ablations": ablations,
        "decision": decision,
        "target_level": "L1 scaffold",
    }
    torch.save(checkpoint, out_dir / "last.pt")
    report: dict[str, object] = {
        "status": "complete",
        "target_level": "L1 scaffold",
        "major_bottleneck": "ordered recurrent list state before LM renderer",
        "decision": decision,
        "last_loss": last_loss,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "ablations": ablations,
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
            "Train a donorless ordered-list state probe. This is an L1 scaffold "
            "for testing whether a learned recurrent slot state can preserve "
            "order and compose filter -> double before canonical QTRM LM work."
        )
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-cases", type=int, default=4096)
    parser.add_argument("--eval-cases", type=int, default=512)
    parser.add_argument("--list-len", type=int, default=5)
    parser.add_argument("--max-output-len", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--value-modulus", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--depths", default="1,2,4")
    parser.add_argument("--seed", type=int, default=315)
    parser.add_argument("--eval-seed", type=int, default=9315)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--accept-min-final-exact", type=float, default=0.90)
    parser.add_argument("--accept-min-depth-gain", type=float, default=0.25)
    parser.add_argument("--accept-min-ablation-drop", type=float, default=0.20)
    parser.add_argument(
        "--require-op-ablation",
        action="store_true",
        help=(
            "Also require op_zero to drop. Disabled by default because this "
            "fixed-operation L1 probe may legitimately use recurrent step count "
            "instead of a variable operation token."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = train_probe(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
