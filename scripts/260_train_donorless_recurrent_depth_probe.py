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


OP_SPECS: tuple[tuple[str, int], ...] = (
    ("add", -5),
    ("add", -3),
    ("add", -1),
    ("add", 1),
    ("add", 3),
    ("add", 5),
    ("mul", 2),
    ("mul", 3),
    ("affine", 2),
    ("affine", 3),
)
NOOP_ID = len(OP_SPECS)


@dataclass(frozen=True)
class ProgramCase:
    case_id: str
    start: int
    op_ids: tuple[int, ...]
    targets: tuple[int, ...]


def apply_op(value: int, op_id: int, modulus: int) -> int:
    if int(op_id) == NOOP_ID:
        return int(value) % int(modulus)
    name, param = OP_SPECS[int(op_id)]
    if name == "add":
        return (int(value) + int(param)) % int(modulus)
    if name == "mul":
        return (int(value) * int(param)) % int(modulus)
    if name == "affine":
        return ((int(value) * int(param)) + int(param + 1)) % int(modulus)
    raise ValueError(f"unknown op: {name}")


def build_cases(
    *,
    count: int,
    seed: int,
    max_program_len: int,
    modulus: int,
) -> list[ProgramCase]:
    rng = random.Random(int(seed))
    cases: list[ProgramCase] = []
    for index in range(int(count)):
        start = rng.randrange(int(modulus))
        op_ids = tuple(rng.randrange(len(OP_SPECS)) for _ in range(int(max_program_len)))
        value = start
        targets: list[int] = [value]
        for op_id in op_ids:
            value = apply_op(value, op_id, int(modulus))
            targets.append(value)
        cases.append(
            ProgramCase(
                case_id=f"prog-{seed}-{index:06d}",
                start=start,
                op_ids=op_ids,
                targets=tuple(targets),
            )
        )
    return cases


def cases_to_batch(
    cases: list[ProgramCase],
    *,
    max_program_len: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start = torch.tensor([case.start for case in cases], dtype=torch.long, device=device)
    op_ids = torch.full(
        (len(cases), int(max_program_len)),
        fill_value=NOOP_ID,
        dtype=torch.long,
        device=device,
    )
    targets = torch.empty(
        (len(cases), int(max_program_len) + 1),
        dtype=torch.long,
        device=device,
    )
    for row_idx, case in enumerate(cases):
        op_ids[row_idx, : len(case.op_ids)] = torch.tensor(case.op_ids, dtype=torch.long, device=device)
        targets[row_idx, : len(case.targets)] = torch.tensor(case.targets, dtype=torch.long, device=device)
    return start, op_ids, targets


class DonorlessRecurrentDepthProbe(nn.Module):
    """Tiny learned recurrent state update, with no donor, retrieval, or solver."""

    def __init__(
        self,
        *,
        modulus: int,
        num_ops: int,
        d_model: int = 96,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.modulus = int(modulus)
        self.num_ops = int(num_ops)
        self.start_embedding = nn.Embedding(self.modulus, int(d_model))
        self.op_embedding = nn.Embedding(self.num_ops, int(d_model))
        self.init = nn.Linear(int(d_model), int(hidden_dim))
        self.cell = nn.GRUCell(int(d_model), int(hidden_dim))
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.output = nn.Linear(int(hidden_dim), self.modulus)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.start_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.op_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.init.weight)
        nn.init.zeros_(self.init.bias)
        for name, param in self.cell.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        *,
        start_ids: torch.Tensor,
        op_ids: torch.Tensor,
        max_steps: int | None = None,
        state_reset_each_step: bool = False,
    ) -> torch.Tensor:
        if start_ids.ndim != 1:
            raise ValueError("start_ids must have shape [batch]")
        if op_ids.ndim != 2:
            raise ValueError("op_ids must have shape [batch, steps]")
        steps = int(op_ids.shape[1] if max_steps is None else max_steps)
        init_state = torch.tanh(self.init(self.start_embedding(start_ids)))
        state = init_state
        logits = [self.output(self.norm(state))]
        for step in range(steps):
            op_emb = self.op_embedding(op_ids[:, step])
            base_state = init_state if bool(state_reset_each_step) else state
            state = self.cell(op_emb, base_state)
            logits.append(self.output(self.norm(state)))
        return torch.stack(logits, dim=1)


class DonorlessTransitionTableDepthProbe(nn.Module):
    """Learned finite latent-state recurrence over modular values."""

    def __init__(
        self,
        *,
        modulus: int,
        num_ops: int,
        transition_init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.modulus = int(modulus)
        self.num_ops = int(num_ops)
        self.transition_logits = nn.Parameter(
            torch.empty(self.num_ops, self.modulus, self.modulus)
        )
        self.transition_init_scale = float(transition_init_scale)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.transition_init_scale > 0.0:
            nn.init.normal_(self.transition_logits, mean=0.0, std=self.transition_init_scale)
        else:
            nn.init.zeros_(self.transition_logits)

    def forward(
        self,
        *,
        start_ids: torch.Tensor,
        op_ids: torch.Tensor,
        max_steps: int | None = None,
        state_reset_each_step: bool = False,
    ) -> torch.Tensor:
        if start_ids.ndim != 1:
            raise ValueError("start_ids must have shape [batch]")
        if op_ids.ndim != 2:
            raise ValueError("op_ids must have shape [batch, steps]")
        steps = int(op_ids.shape[1] if max_steps is None else max_steps)
        init_state = F.one_hot(start_ids, num_classes=self.modulus).to(torch.float32)
        state = init_state
        logits = [torch.log(state.clamp_min(1.0e-9))]
        transition_probs = F.softmax(self.transition_logits, dim=-1)
        for step in range(steps):
            transition = transition_probs[op_ids[:, step]]
            base_state = init_state if bool(state_reset_each_step) else state
            state = torch.bmm(base_state.unsqueeze(1), transition).squeeze(1)
            logits.append(torch.log(state.clamp_min(1.0e-9)))
        return torch.stack(logits, dim=1)


def loss_for_batch(
    model: nn.Module,
    start: torch.Tensor,
    op_ids: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    logits = model(start_ids=start, op_ids=op_ids)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )


def direct_transition_loss_for_batch(
    model: nn.Module,
    op_ids: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(model, DonorlessTransitionTableDepthProbe):
        return torch.zeros((), dtype=torch.float32, device=op_ids.device)
    previous_targets = targets[:, :-1]
    next_targets = targets[:, 1:]
    logits = model.transition_logits[op_ids, previous_targets]
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        next_targets.reshape(-1),
    )


@torch.no_grad()
def evaluate_cases(
    model: nn.Module,
    cases: list[ProgramCase],
    args: argparse.Namespace,
    *,
    ablation: str = "none",
) -> dict[str, float | int | str]:
    model.eval()
    start, op_ids, targets = cases_to_batch(
        cases,
        max_program_len=args.max_program_len,
        device=args.device,
    )
    if ablation == "op_zero":
        op_ids = torch.full_like(op_ids, NOOP_ID)
    elif ablation == "op_shuffle":
        generator = torch.Generator(device=op_ids.device)
        generator.manual_seed(int(args.seed) + 999)
        shuffled = op_ids.clone()
        for row_idx in range(shuffled.shape[0]):
            perm = torch.randperm(shuffled.shape[1], generator=generator, device=shuffled.device)
            shuffled[row_idx] = shuffled[row_idx, perm]
        op_ids = shuffled
    elif ablation not in {"none", "state_reset"}:
        raise ValueError(f"unknown ablation: {ablation}")

    logits = model(
        start_ids=start,
        op_ids=op_ids,
        state_reset_each_step=(ablation == "state_reset"),
    )
    pred = logits.argmax(dim=-1)
    final_target = targets[:, -1]
    metrics: dict[str, float | int | str] = {
        "cases": len(cases),
        "ablation": ablation,
    }
    for depth in parse_depths(args.depths):
        clipped = min(int(depth), int(args.max_program_len))
        metrics[f"depth{depth}_final_exact"] = float((pred[:, clipped] == final_target).float().mean().item())
        metrics[f"depth{depth}_prefix_exact"] = float((pred[:, clipped] == targets[:, clipped]).float().mean().item())
    return metrics


def parse_depths(text: str | Iterable[int]) -> list[int]:
    if isinstance(text, str):
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    return [int(value) for value in text]


def train_probe(args: argparse.Namespace) -> dict[str, object]:
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    train_cases = build_cases(
        count=args.train_cases,
        seed=args.seed,
        max_program_len=args.max_program_len,
        modulus=args.modulus,
    )
    eval_cases = build_cases(
        count=args.eval_cases,
        seed=args.eval_seed,
        max_program_len=args.max_program_len,
        modulus=args.modulus,
    )
    if args.model_kind == "gru":
        model: nn.Module = DonorlessRecurrentDepthProbe(
            modulus=args.modulus,
            num_ops=NOOP_ID + 1,
            d_model=args.d_model,
            hidden_dim=args.hidden_dim,
        ).to(args.device)
    elif args.model_kind == "transition_table":
        model = DonorlessTransitionTableDepthProbe(
            modulus=args.modulus,
            num_ops=NOOP_ID + 1,
            transition_init_scale=args.transition_init_scale,
        ).to(args.device)
    else:
        raise ValueError(f"unknown model_kind: {args.model_kind}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_value = 0.0
    for step in range(int(args.steps)):
        model.train()
        indices = [
            ((step * int(args.batch_size)) + offset) % len(train_cases)
            for offset in range(int(args.batch_size))
        ]
        batch_cases = [train_cases[index] for index in indices]
        start, op_ids, targets = cases_to_batch(
            batch_cases,
            max_program_len=args.max_program_len,
            device=args.device,
        )
        recurrent_loss = loss_for_batch(model, start, op_ids, targets)
        transition_loss = direct_transition_loss_for_batch(model, op_ids, targets)
        loss = (
            float(args.recurrent_ce_weight) * recurrent_loss
            + float(args.direct_transition_ce_weight) * transition_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        loss_value = float(loss.item())
        if int(args.log_every) > 0 and (step + 1) % int(args.log_every) == 0:
            print(f"step={step + 1} loss={loss_value:.4f}")

    train_metrics = evaluate_cases(model, train_cases, args)
    eval_metrics = evaluate_cases(model, eval_cases, args)
    ablations = {
        name: evaluate_cases(model, eval_cases, args, ablation=name)
        for name in ("state_reset", "op_zero", "op_shuffle")
    }
    depth_values = parse_depths(args.depths)
    max_depth = max(depth_values)
    shallow_depth = min(depth for depth in depth_values if depth > 0)
    full_exact = float(eval_metrics[f"depth{max_depth}_final_exact"])
    shallow_exact = float(eval_metrics[f"depth{shallow_depth}_final_exact"])
    worst_ablation = max(float(metrics[f"depth{max_depth}_final_exact"]) for metrics in ablations.values())
    decision = (
        "accepted_l1"
        if full_exact >= float(args.accept_min_final_exact)
        and (full_exact - shallow_exact) >= float(args.accept_min_depth_gain)
        and (full_exact - worst_ablation) >= float(args.accept_min_ablation_drop)
        else "rejected"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "op_specs": list(OP_SPECS),
        "noop_id": NOOP_ID,
        "model_kind": args.model_kind,
        "args": vars(args),
        "last_loss": loss_value,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "ablations": ablations,
        "decision": decision,
    }
    torch.save(checkpoint, out_dir / "last.pt")
    report: dict[str, object] = {
        "status": "complete",
        "target_level": "L1 scaffold",
        "decision": decision,
        "model_kind": args.model_kind,
        "last_loss": loss_value,
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
            "Train a donorless recurrent depth probe on synthetic modular programs. "
            "This is an L1 scaffold for proving learned latent state updates can "
            "benefit from more recurrent steps before integrated QTRM tuning."
        )
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-cases", type=int, default=4096)
    parser.add_argument("--eval-cases", type=int, default=512)
    parser.add_argument("--max-program-len", type=int, default=8)
    parser.add_argument("--modulus", type=int, default=97)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--model-kind",
        choices=["transition_table", "gru"],
        default="transition_table",
        help=(
            "transition_table learns an op-conditioned finite latent-state "
            "transition; gru is the previous free hidden-state baseline."
        ),
    )
    parser.add_argument("--transition-init-scale", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--recurrent-ce-weight", type=float, default=1.0)
    parser.add_argument("--direct-transition-ce-weight", type=float, default=1.0)
    parser.add_argument("--depths", default="0,1,2,4,8")
    parser.add_argument("--seed", type=int, default=260)
    parser.add_argument("--eval-seed", type=int, default=9260)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--accept-min-final-exact", type=float, default=0.95)
    parser.add_argument("--accept-min-depth-gain", type=float, default=0.50)
    parser.add_argument("--accept-min-ablation-drop", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = train_probe(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
