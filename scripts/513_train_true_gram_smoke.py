"""
Stage30A minimum-faithful GRAM smoke test.

This is deliberately smaller than the Qwen-backed trainer. Its job is to test
whether the GRAM mechanism itself is alive before we graft it back into QTRM:

- stochastic transition prior p(z_t | z_{t-1}, op_t, arg_t)
- target-aware variational posterior q(z_t | z_{t-1}, op_t, arg_t, y_t)
- reparameterized z_t is the recurrent state, not a tiny post-hoc delta
- ELBO-style task loss plus KL with free-bits and beta warmup
- inference-time width scaling by sampling K prior trajectories
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


OP_TO_ID = {"add": 0, "mul": 1, "sub": 2, "copy": 3}
ID_TO_OP = {value: key for key, value in OP_TO_ID.items()}


@dataclass(frozen=True)
class ModuloCase:
    initial_label: int
    operation_ids: List[int]
    operation_arg_ids: List[int]
    state_labels: List[int]
    answer_label: int
    depth: int
    family: str


def _pad(values: List[int], length: int, pad_value: int) -> List[int]:
    if len(values) >= length:
        return values[:length]
    return values + [pad_value] * (length - len(values))


def _apply_op(value: int, op_id: int, arg: int) -> int:
    op = ID_TO_OP[int(op_id)]
    if op == "add":
        return (value + arg) % 10
    if op == "mul":
        return (value * arg) % 10
    if op == "sub":
        return (value - arg) % 10
    if op == "copy":
        return value
    raise ValueError(f"unknown op_id: {op_id}")


def build_cases(
    *,
    count: int,
    depths: Sequence[int],
    max_steps: int,
    seed: int,
    family_mix: str,
) -> List[ModuloCase]:
    if not depths:
        raise ValueError("depths must not be empty")
    if max(depths) > max_steps:
        raise ValueError("max depth must be <= max_steps")
    family_mixes = {
        "chain": ("chain",),
        "checksum": ("checksum",),
        "chain2_checksum1": ("chain", "chain", "checksum"),
        "balanced": ("chain", "checksum"),
    }
    if family_mix not in family_mixes:
        raise ValueError(f"unknown family_mix: {family_mix}")

    rng = random.Random(seed)
    cases: List[ModuloCase] = []
    families = family_mixes[family_mix]
    for index in range(count):
        depth = depths[index % len(depths)]
        family = families[(index // len(depths)) % len(families)]
        if family == "checksum":
            value = 0
            digits = [rng.randint(0, 9) for _ in range(depth)]
            states: List[int] = []
            for digit in digits:
                value = (value + digit) % 10
                states.append(value)
            ops = [OP_TO_ID["add"]] * depth
            args = digits
            initial = 0
        else:
            value = rng.randint(0, 9)
            initial = value
            ops = []
            args = []
            states = []
            for _ in range(depth):
                op_name = rng.choice(("add", "mul", "sub"))
                arg = rng.randint(0, 9)
                op_id = OP_TO_ID[op_name]
                value = _apply_op(value, op_id, arg)
                ops.append(op_id)
                args.append(arg)
                states.append(value)

        answer = states[-1]
        cases.append(
            ModuloCase(
                initial_label=initial,
                operation_ids=_pad(ops, max_steps, OP_TO_ID["copy"]),
                operation_arg_ids=_pad(args, max_steps, 0),
                state_labels=_pad(states, max_steps, answer),
                answer_label=answer,
                depth=depth,
                family=family,
            )
        )
    return cases


class ModuloDataset(Dataset):
    def __init__(self, cases: Sequence[ModuloCase]) -> None:
        self.cases = list(cases)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        case = self.cases[index]
        return {
            "initial_labels": torch.tensor(case.initial_label, dtype=torch.long),
            "operation_ids": torch.tensor(case.operation_ids, dtype=torch.long),
            "operation_arg_ids": torch.tensor(case.operation_arg_ids, dtype=torch.long),
            "state_labels": torch.tensor(case.state_labels, dtype=torch.long),
            "answer_labels": torch.tensor(case.answer_label, dtype=torch.long),
            "depths": torch.tensor(case.depth, dtype=torch.long),
        }


class FeedForwardBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, zero_last: bool = False) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        if zero_last:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeterministicModuloReasoner(nn.Module):
    def __init__(self, *, d_state: int, hidden_dim: int, n_steps: int) -> None:
        super().__init__()
        self.n_steps = int(n_steps)
        self.initial_embed = nn.Embedding(10, d_state)
        self.op_embed = nn.Embedding(len(OP_TO_ID), d_state)
        self.arg_embed = nn.Embedding(10, d_state)
        self.transition = FeedForwardBlock(d_state * 3, hidden_dim, d_state)
        self.state_norm = nn.LayerNorm(d_state)
        self.state_head = nn.Linear(d_state, 10)
        self.answer_head = nn.Linear(d_state, 10)

    def forward(
        self,
        *,
        initial_labels: torch.Tensor,
        operation_ids: torch.Tensor,
        operation_arg_ids: torch.Tensor,
        state_labels: Optional[torch.Tensor] = None,
        sample_prior: bool = False,
    ) -> Dict[str, torch.Tensor]:
        del state_labels, sample_prior
        z = self.initial_embed(initial_labels)
        states = []
        n_steps = min(self.n_steps, operation_ids.size(1))
        for step in range(n_steps):
            op_vec = self.op_embed(operation_ids[:, step])
            arg_vec = self.arg_embed(operation_arg_ids[:, step].clamp(0, 9))
            delta = self.transition(torch.cat([z, op_vec, arg_vec], dim=-1))
            z = self.state_norm(z + delta)
            states.append(z)
        trajectory = torch.stack(states, dim=1)
        return {
            "answer_logits": self.answer_head(trajectory[:, -1]),
            "state_logits": self.state_head(trajectory),
            "trajectory": trajectory,
            "kl_per_step": None,
            "prior_std_per_step": None,
            "reward_logits": None,
        }


class TrueGRAMModuloReasoner(nn.Module):
    def __init__(
        self,
        *,
        d_state: int,
        hidden_dim: int,
        n_steps: int,
        min_logvar: float,
        max_logvar: float,
    ) -> None:
        super().__init__()
        self.n_steps = int(n_steps)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)
        self.initial_embed = nn.Embedding(10, d_state)
        self.op_embed = nn.Embedding(len(OP_TO_ID), d_state)
        self.arg_embed = nn.Embedding(10, d_state)
        self.label_embed = nn.Embedding(10, d_state)
        self.prior = FeedForwardBlock(d_state * 3, hidden_dim, d_state * 2)
        self.posterior = FeedForwardBlock(d_state * 4, hidden_dim, d_state * 2)
        self.refine = FeedForwardBlock(d_state * 3, hidden_dim, d_state, zero_last=True)
        self.state_norm = nn.LayerNorm(d_state)
        self.state_head = nn.Linear(d_state, 10)
        self.answer_head = nn.Linear(d_state, 10)
        self.reward_head = nn.Linear(d_state, 1)

    def _split_distribution(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar.clamp(self.min_logvar, self.max_logvar)

    @staticmethod
    def _sample(mu: torch.Tensor, logvar: torch.Tensor, *, sample: bool) -> torch.Tensor:
        if not sample:
            return mu
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    @staticmethod
    def _kl_diag_gaussian(
        q_mu: torch.Tensor,
        q_logvar: torch.Tensor,
        p_mu: torch.Tensor,
        p_logvar: torch.Tensor,
    ) -> torch.Tensor:
        q_var = q_logvar.float().exp()
        p_var = p_logvar.float().exp().clamp_min(1e-8)
        mean_delta = (q_mu.float() - p_mu.float()).pow(2)
        return 0.5 * ((q_var + mean_delta) / p_var - 1.0 + p_logvar.float() - q_logvar.float()).sum(dim=-1)

    def forward(
        self,
        *,
        initial_labels: torch.Tensor,
        operation_ids: torch.Tensor,
        operation_arg_ids: torch.Tensor,
        state_labels: Optional[torch.Tensor] = None,
        sample_prior: bool = False,
    ) -> Dict[str, torch.Tensor]:
        z = self.initial_embed(initial_labels)
        states = []
        rewards = []
        kls = []
        prior_stds = []
        n_steps = min(self.n_steps, operation_ids.size(1))
        for step in range(n_steps):
            prev_z = z
            op_vec = self.op_embed(operation_ids[:, step])
            arg_vec = self.arg_embed(operation_arg_ids[:, step].clamp(0, 9))
            prior_input = torch.cat([prev_z, op_vec, arg_vec], dim=-1)
            prior_mu, prior_logvar = self._split_distribution(self.prior(prior_input))
            prior_stds.append(torch.exp(0.5 * prior_logvar).mean(dim=-1))

            if self.training and state_labels is not None:
                label_vec = self.label_embed(state_labels[:, step].clamp(0, 9))
                post_input = torch.cat([prev_z, op_vec, arg_vec, label_vec], dim=-1)
                post_mu, post_logvar = self._split_distribution(self.posterior(post_input))
                z_sample = self._sample(post_mu, post_logvar, sample=True)
                kls.append(self._kl_diag_gaussian(post_mu, post_logvar, prior_mu, prior_logvar))
            else:
                z_sample = self._sample(prior_mu, prior_logvar, sample=sample_prior)
                kls.append(torch.zeros_like(initial_labels, dtype=z_sample.dtype))

            refine_delta = self.refine(torch.cat([z_sample, op_vec, arg_vec], dim=-1))
            z = self.state_norm(z_sample + refine_delta)
            states.append(z)
            rewards.append(self.reward_head(z).squeeze(-1))

        trajectory = torch.stack(states, dim=1)
        return {
            "answer_logits": self.answer_head(trajectory[:, -1]),
            "state_logits": self.state_head(trajectory),
            "trajectory": trajectory,
            "kl_per_step": torch.stack(kls, dim=1),
            "prior_std_per_step": torch.stack(prior_stds, dim=1),
            "reward_logits": torch.stack(rewards, dim=1),
        }


def init_aim_run(args: argparse.Namespace, *, variant: str) -> Optional[Any]:
    if not args.aim_repo:
        return None
    try:
        from aim import Run
    except ImportError as exc:
        print(f"[warn] Aim logging disabled; package is not installed: {exc}", flush=True)
        return None
    run = Run(repo=args.aim_repo, experiment=args.aim_experiment)
    run.name = f"{args.run_name}_{variant}"
    run.description = args.aim_description or "Stage30A minimum-faithful GRAM smoke"
    run["hparams"] = {**vars(args), "variant": variant}
    run["paths"] = {
        "out_dir": os.path.join(args.out_dir, variant),
        "tensorboard_logdir": os.path.join(args.out_dir, variant, "logs"),
    }
    return run


def track_aim(
    aim_run: Optional[Any],
    value: float,
    *,
    name: str,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    context: Optional[Dict[str, str]] = None,
) -> None:
    if aim_run is None:
        return
    aim_run.track(float(value), name=name, step=step, epoch=epoch, context=context or {})


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_task_losses(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    *,
    state_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    answer_loss = F.cross_entropy(out["answer_logits"], batch["answer_labels"])
    state_loss = F.cross_entropy(
        out["state_logits"].reshape(-1, 10),
        batch["state_labels"][:, : out["state_logits"].size(1)].reshape(-1),
    )
    loss = answer_loss + state_loss * float(state_weight)
    preds = out["answer_logits"].argmax(dim=-1)
    accuracy = (preds == batch["answer_labels"]).float().mean()
    return loss, {
        "answer_loss": float(answer_loss.detach().item()),
        "state_loss": float(state_loss.detach().item()),
        "accuracy": float(accuracy.detach().item()),
    }


def kl_free_bits_loss(kl_per_step: torch.Tensor, *, free_bits: float) -> torch.Tensor:
    return torch.clamp(kl_per_step - float(free_bits), min=0.0).mean()


def reward_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, float]:
    reward_logits = out["reward_logits"]
    if reward_logits is None:
        return out["answer_logits"].new_zeros(()), 0.0
    with torch.no_grad():
        correct = (out["answer_logits"].argmax(dim=-1) == batch["answer_labels"]).float()
        targets = correct[:, None].expand_as(reward_logits)
    pred = torch.sigmoid(reward_logits)
    loss = F.mse_loss(pred, targets)
    calibration_error = (pred[:, -1] - correct).abs().mean()
    return loss, float(calibration_error.detach().item())


@torch.no_grad()
def trajectory_cosine_similarity(trajectory: torch.Tensor, *, samples: int) -> float:
    if samples <= 1:
        return 1.0
    total = trajectory.size(0)
    if total % samples != 0:
        return 1.0
    batch = total // samples
    z = trajectory[:, -1, :].reshape(batch, samples, -1)
    z = F.normalize(z.float(), dim=-1)
    sims = torch.matmul(z, z.transpose(1, 2))
    mask = ~torch.eye(samples, dtype=torch.bool, device=z.device)[None, :, :]
    return float(sims[mask.expand_as(sims)].mean().item())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    variant: str,
    samples: int,
    selection: str,
) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    depth_correct: Dict[int, int] = {}
    depth_total: Dict[int, int] = {}
    diversity_values = []
    calibration_values = []

    for raw_batch in loader:
        batch = batch_to_device(raw_batch, device)
        if variant == "true_gram" and samples > 1:
            repeated = {
                key: value.repeat_interleave(samples, dim=0)
                for key, value in batch.items()
                if key != "depths"
            }
            out = model(
                initial_labels=repeated["initial_labels"],
                operation_ids=repeated["operation_ids"],
                operation_arg_ids=repeated["operation_arg_ids"],
                state_labels=None,
                sample_prior=True,
            )
            logits = out["answer_logits"].reshape(batch["answer_labels"].size(0), samples, 10)
            if selection == "lprm" and out["reward_logits"] is not None:
                scores = out["reward_logits"][:, -1].reshape(batch["answer_labels"].size(0), samples)
                selected = scores.argmax(dim=-1)
                chosen_logits = logits[torch.arange(logits.size(0), device=device), selected]
            elif selection == "vote":
                preds = logits.argmax(dim=-1)
                chosen = []
                for row in preds:
                    counts = torch.bincount(row, minlength=10)
                    chosen.append(counts.argmax())
                chosen_pred = torch.stack(chosen)
                chosen_logits = F.one_hot(chosen_pred, num_classes=10).float()
            else:
                chosen_logits = logits.mean(dim=1)
            diversity_values.append(trajectory_cosine_similarity(out["trajectory"], samples=samples))
            if out["reward_logits"] is not None:
                final_reward = torch.sigmoid(out["reward_logits"][:, -1]).reshape(batch["answer_labels"].size(0), samples)
                best_reward = final_reward.max(dim=-1).values
                pred = chosen_logits.argmax(dim=-1)
                calibration_values.append(float((best_reward - (pred == batch["answer_labels"]).float()).abs().mean().item()))
        else:
            out = model(
                initial_labels=batch["initial_labels"],
                operation_ids=batch["operation_ids"],
                operation_arg_ids=batch["operation_arg_ids"],
                state_labels=None,
                sample_prior=False,
            )
            chosen_logits = out["answer_logits"]
            diversity_values.append(1.0)
        pred = chosen_logits.argmax(dim=-1)
        batch_correct = (pred == batch["answer_labels"])
        correct += int(batch_correct.sum().item())
        total += int(batch_correct.numel())
        for depth in torch.unique(batch["depths"]).tolist():
            mask = batch["depths"] == depth
            depth_correct[int(depth)] = depth_correct.get(int(depth), 0) + int(batch_correct[mask].sum().item())
            depth_total[int(depth)] = depth_total.get(int(depth), 0) + int(mask.sum().item())

    metrics = {
        "accuracy": correct / max(total, 1),
        "trajectory_cosine": sum(diversity_values) / max(len(diversity_values), 1),
        "lprm_calibration_error": sum(calibration_values) / max(len(calibration_values), 1) if calibration_values else 0.0,
    }
    for depth, count in sorted(depth_total.items()):
        metrics[f"depth{depth}_accuracy"] = depth_correct.get(depth, 0) / max(count, 1)
    return metrics


def make_model(args: argparse.Namespace, *, variant: str) -> nn.Module:
    if variant == "deterministic":
        return DeterministicModuloReasoner(
            d_state=args.d_state,
            hidden_dim=args.hidden_dim,
            n_steps=args.n_steps,
        )
    if variant == "true_gram":
        return TrueGRAMModuloReasoner(
            d_state=args.d_state,
            hidden_dim=args.hidden_dim,
            n_steps=args.n_steps,
            min_logvar=args.min_logvar,
            max_logvar=args.max_logvar,
        )
    raise ValueError(f"unknown variant: {variant}")


def train_variant(
    args: argparse.Namespace,
    *,
    variant: str,
    train_cases: Sequence[ModuloCase],
    eval_cases: Sequence[ModuloCase],
    device: torch.device,
) -> Dict[str, Any]:
    variant_dir = os.path.join(args.out_dir, variant)
    os.makedirs(variant_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(variant_dir, "logs"))
    aim_run = init_aim_run(args, variant=variant)
    model = make_model(args, variant=variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        ModuloDataset(train_cases),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    eval_loader = DataLoader(
        ModuloDataset(eval_cases),
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
    )
    global_step = 0
    best_metric = float("-inf")
    best_summary: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        totals = {
            "loss": 0.0,
            "answer_loss": 0.0,
            "state_loss": 0.0,
            "accuracy": 0.0,
            "kl_raw": 0.0,
            "kl_loss": 0.0,
            "prior_std": 0.0,
            "lprm_loss": 0.0,
            "lprm_calibration_error": 0.0,
        }
        seen = 0
        for raw_batch in train_loader:
            batch = batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(
                initial_labels=batch["initial_labels"],
                operation_ids=batch["operation_ids"],
                operation_arg_ids=batch["operation_arg_ids"],
                state_labels=batch["state_labels"] if variant == "true_gram" else None,
                sample_prior=False,
            )
            task_loss, task_metrics = compute_task_losses(out, batch, state_weight=args.state_loss_weight)
            kl_raw_value = 0.0
            kl_loss_value = torch.zeros((), device=device)
            prior_std_value = 0.0
            if variant == "true_gram":
                kl_per_step = out["kl_per_step"]
                kl_raw_value = float(kl_per_step.detach().mean().item())
                kl_beta = min(1.0, global_step / max(args.kl_warmup_steps, 1)) * args.kl_weight
                kl_loss_value = kl_free_bits_loss(kl_per_step, free_bits=args.kl_free_bits) * kl_beta
                prior_std_value = float(out["prior_std_per_step"].detach().mean().item())
            lprm_loss, lprm_cal_error = reward_loss(out, batch)
            loss = task_loss + kl_loss_value + lprm_loss * float(args.lprm_weight)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch_size = batch["answer_labels"].size(0)
            seen += batch_size
            totals["loss"] += float(loss.detach().item()) * batch_size
            totals["answer_loss"] += task_metrics["answer_loss"] * batch_size
            totals["state_loss"] += task_metrics["state_loss"] * batch_size
            totals["accuracy"] += task_metrics["accuracy"] * batch_size
            totals["kl_raw"] += kl_raw_value * batch_size
            totals["kl_loss"] += float(kl_loss_value.detach().item()) * batch_size
            totals["prior_std"] += prior_std_value * batch_size
            totals["lprm_loss"] += float(lprm_loss.detach().item()) * batch_size
            totals["lprm_calibration_error"] += lprm_cal_error * batch_size

            if global_step % args.log_every == 0:
                context = {"variant": variant, "split": "train"}
                writer.add_scalar("Train/Step/Loss", float(loss.detach().item()), global_step)
                writer.add_scalar("Train/Step/Accuracy", task_metrics["accuracy"], global_step)
                writer.add_scalar("Train/Step/KLRaw", kl_raw_value, global_step)
                writer.add_scalar("Train/Step/KLLoss", float(kl_loss_value.detach().item()), global_step)
                writer.add_scalar("Train/Step/PriorStdMean", prior_std_value, global_step)
                writer.add_scalar("Train/Step/LPRMLoss", float(lprm_loss.detach().item()), global_step)
                writer.add_scalar("Train/Step/GradNorm", float(grad_norm), global_step)
                track_aim(aim_run, float(loss.detach().item()), name="loss", step=global_step, epoch=epoch, context=context)
                track_aim(aim_run, task_metrics["accuracy"], name="accuracy", step=global_step, epoch=epoch, context=context)
                track_aim(aim_run, kl_raw_value, name="kl_raw", step=global_step, epoch=epoch, context=context)
                track_aim(aim_run, prior_std_value, name="prior_std_mean", step=global_step, epoch=epoch, context=context)
            global_step += 1

        epoch_metrics = {key: value / max(seen, 1) for key, value in totals.items()}
        for key, value in epoch_metrics.items():
            writer.add_scalar(f"Train/Epoch/{key}", value, epoch)
            track_aim(aim_run, value, name=key, epoch=epoch, context={"variant": variant, "split": "train"})
        writer.add_scalar("Train/Epoch/LearningRate", optimizer.param_groups[0]["lr"], epoch)

        eval_results: Dict[str, Dict[str, float]] = {}
        eval_samples = [1] if variant == "deterministic" else args.eval_samples
        for samples in eval_samples:
            metrics = evaluate(
                model,
                eval_loader,
                device=device,
                variant=variant,
                samples=samples,
                selection=args.selection,
            )
            eval_results[f"k{samples}"] = metrics
            context = {"variant": variant, "split": "heldout", "samples": str(samples)}
            for key, value in metrics.items():
                writer.add_scalar(f"Heldout/K{samples}/{key}", value, epoch)
                track_aim(aim_run, value, name=key, epoch=epoch, context=context)

        primary_key = f"k{max(eval_samples)}"
        primary_metric = eval_results[primary_key]["accuracy"]
        if primary_metric > best_metric:
            best_metric = primary_metric
            best_summary = {
                "epoch": epoch,
                "primary_samples": max(eval_samples),
                "primary_accuracy": primary_metric,
                "train": epoch_metrics,
                "heldout": eval_results,
                "checkpoint": os.path.join(variant_dir, "best.pt"),
            }
            torch.save({"model": model.state_dict(), "args": vars(args), "variant": variant, "best": best_summary}, best_summary["checkpoint"])
            with open(os.path.join(variant_dir, "best_info.json"), "w", encoding="utf-8") as handle:
                json.dump(best_summary, handle, indent=2, sort_keys=True)
            if aim_run is not None:
                aim_run["best"] = best_summary

        epoch_summary = {
            "epoch": epoch,
            "time_seconds": time.time() - start,
            "train": epoch_metrics,
            "heldout": eval_results,
            "best_primary_accuracy": best_metric,
        }
        history.append(epoch_summary)
        with open(os.path.join(variant_dir, "history.json"), "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, sort_keys=True)
        print(
            f"{variant} epoch {epoch:03d} | "
            f"train_acc={epoch_metrics['accuracy']:.4f} "
            f"kl={epoch_metrics['kl_raw']:.4f} "
            f"std={epoch_metrics['prior_std']:.4f} "
            f"heldout_{primary_key}={primary_metric:.4f} "
            f"time={epoch_summary['time_seconds']:.1f}s",
            flush=True,
        )

    writer.close()
    if aim_run is not None:
        aim_run.close()
    return {
        "variant": variant,
        "best": best_summary,
        "history": history,
    }


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--variant", choices=("deterministic", "true_gram", "both"), default="both")
    parser.add_argument("--train-count", type=int, default=1024)
    parser.add_argument("--eval-count", type=int, default=512)
    parser.add_argument("--train-depths", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--eval-depths", type=int, nargs="+", default=[4, 6, 8, 10])
    parser.add_argument("--family-mix", choices=("chain", "checksum", "chain2_checksum1", "balanced"), default="chain2_checksum1")
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--d-state", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--state-loss-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--kl-free-bits", type=float, default=0.2)
    parser.add_argument("--kl-warmup-steps", type=int, default=120)
    parser.add_argument("--lprm-weight", type=float, default=0.05)
    parser.add_argument("--min-logvar", type=float, default=-10.0)
    parser.add_argument("--max-logvar", type=float, default=2.0)
    parser.add_argument("--eval-samples", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--selection", choices=("mean", "vote", "lprm"), default="lprm")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=130)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--aim-repo", default=os.environ.get("QTRM_AIM_REPO"))
    parser.add_argument("--aim-experiment", default="qtrm_stage30_true_gram_smoke")
    parser.add_argument("--aim-description", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_name is None:
        args.run_name = os.path.basename(os.path.normpath(args.out_dir))
    os.makedirs(args.out_dir, exist_ok=True)
    configure_reproducibility(args.seed)
    device = torch.device(args.device)
    train_cases = build_cases(
        count=args.train_count,
        depths=args.train_depths,
        max_steps=args.n_steps,
        seed=args.seed,
        family_mix=args.family_mix,
    )
    eval_cases = build_cases(
        count=args.eval_count,
        depths=args.eval_depths,
        max_steps=args.n_steps,
        seed=args.seed + 10042,
        family_mix=args.family_mix,
    )
    with open(os.path.join(args.out_dir, "run_info.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": vars(args),
                "train_case_example": asdict(train_cases[0]),
                "eval_case_example": asdict(eval_cases[0]),
                "sources": {
                    "gram_arxiv": "https://arxiv.org/abs/2605.19376",
                    "gram_project": "https://ahn-ml.github.io/gram-website/",
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    variants = ["deterministic", "true_gram"] if args.variant == "both" else [args.variant]
    summaries = []
    for variant in variants:
        summaries.append(
            train_variant(
                args,
                variant=variant,
                train_cases=train_cases,
                eval_cases=eval_cases,
                device=device,
            )
        )
    result = {
        "run_name": args.run_name,
        "out_dir": args.out_dir,
        "summaries": summaries,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
