"""Train a local candidate-conditioned trace verifier.

Stage51 asks the verifier to do more than score a latent vector. For each
sampled candidate answer, it rereads the prompt and predicts the stepwise
modulo trace. Candidate selection is then based on whether the predicted final
trace supports that candidate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from qtrm_mm.norm import RMSNorm
from qtrm_mm.qwen_backbone_state_transition import build_qwen_state_transition_model


def _load_train511() -> Any:
    path = Path(__file__).resolve().parent / "511_train_qwen_state_transition_hrmtext.py"
    spec = importlib.util.spec_from_file_location("qtrm_stage511", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


train511 = _load_train511()


class CandidateTraceVerifier(nn.Module):
    """Predict per-step modulo states for a prompt/candidate pair."""

    def __init__(self, hidden_size: int, max_steps: int, hidden_dim: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.step_embed = nn.Embedding(self.max_steps, int(embed_dim))
        self.candidate_embed = nn.Embedding(10, int(embed_dim))
        in_dim = int(hidden_size) + int(embed_dim) * 2
        self.net = nn.Sequential(
            RMSNorm(in_dim),
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 10),
        )
        nn.init.normal_(self.step_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.candidate_embed.weight, mean=0.0, std=0.02)
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, reader_hidden: torch.Tensor, candidate_digits: torch.Tensor) -> torch.Tensor:
        batch = int(reader_hidden.size(0))
        steps = torch.arange(self.max_steps, device=reader_hidden.device)
        step_features = self.step_embed(steps).unsqueeze(0).expand(batch, -1, -1)
        candidate_features = self.candidate_embed(candidate_digits.to(torch.long)).unsqueeze(1).expand(-1, self.max_steps, -1)
        reader_features = reader_hidden.unsqueeze(1).expand(-1, self.max_steps, -1)
        features = torch.cat([reader_features, candidate_features, step_features], dim=-1)
        return self.net(features)


class CandidateTraceSequenceVerifier(nn.Module):
    """Predict per-step states by attending each step query over Qwen tokens."""

    def __init__(self, hidden_size: int, max_steps: int, hidden_dim: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.hidden_size = int(hidden_size)
        self.step_embed = nn.Embedding(self.max_steps, int(embed_dim))
        self.candidate_embed = nn.Embedding(10, int(embed_dim))
        self.query_proj = nn.Linear(int(embed_dim) * 2, self.hidden_size)
        self.net = nn.Sequential(
            RMSNorm(self.hidden_size + int(embed_dim) * 2),
            nn.Linear(self.hidden_size + int(embed_dim) * 2, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 10),
        )
        nn.init.normal_(self.step_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.candidate_embed.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        reader_states: torch.Tensor,
        reader_mask: torch.Tensor,
        candidate_digits: torch.Tensor,
    ) -> torch.Tensor:
        batch = int(reader_states.size(0))
        steps = torch.arange(self.max_steps, device=reader_states.device)
        step_features = self.step_embed(steps).unsqueeze(0).expand(batch, -1, -1)
        candidate_features = self.candidate_embed(candidate_digits.to(torch.long)).unsqueeze(1).expand(-1, self.max_steps, -1)
        query_input = torch.cat([candidate_features, step_features], dim=-1)
        queries = self.query_proj(query_input)
        scores = torch.einsum("bth,bsh->bts", queries.float(), reader_states.float()) / (self.hidden_size ** 0.5)
        scores = scores.masked_fill(reader_mask.to(torch.bool).unsqueeze(1).logical_not(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).to(reader_states.dtype)
        context = torch.einsum("bts,bsh->bth", weights, reader_states)
        features = torch.cat([context, candidate_features.to(context.dtype), step_features.to(context.dtype)], dim=-1)
        return self.net(features)


def configure_reproducibility(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def encode_batch(tokenizer: Any, texts: Sequence[str], max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(texts),
        truncation=True,
        max_length=int(max_length),
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }


def apply_recurrent_overrides(model: Any, args: argparse.Namespace) -> Dict[str, int]:
    stats = {"transition_scale": 0, "injection_gate": 0, "step_embeddings_zeroed": 0, "step_embeddings_frozen": 0}
    if args.override_transition_scale is not None:
        for module in model.modules():
            scale = getattr(module, "transition_scale", None)
            if isinstance(scale, torch.nn.Parameter):
                with torch.no_grad():
                    scale.fill_(float(args.override_transition_scale))
                stats["transition_scale"] += 1
    if args.override_injection_gate_logit is not None:
        for module in model.modules():
            gate = getattr(module, "injection_gate", None)
            if isinstance(gate, torch.nn.Parameter):
                with torch.no_grad():
                    gate.fill_(float(args.override_injection_gate_logit))
                stats["injection_gate"] += 1
    if args.zero_step_embeddings or args.freeze_step_embeddings:
        for module in model.modules():
            step_embed = getattr(module, "step_embed", None)
            weight = getattr(step_embed, "weight", None)
            if isinstance(weight, torch.nn.Parameter):
                if args.zero_step_embeddings:
                    with torch.no_grad():
                        weight.zero_()
                    stats["step_embeddings_zeroed"] += 1
                if args.freeze_step_embeddings:
                    weight.requires_grad_(False)
                    stats["step_embeddings_frozen"] += 1
    return stats


def verifier_text(prompt_text: str, candidate_digit: int) -> str:
    return (
        f"{prompt_text.rstrip()} {int(candidate_digit)}\n"
        "Verifier: compute the modulo-10 state after every step. "
        f"The proposed final digit is {int(candidate_digit)}."
    )


@torch.inference_mode()
def qwen_last_hidden(
    qwen: nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    encoded = encode_batch(tokenizer, texts, max_length, device)
    outputs = qwen(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states[-1] if getattr(outputs, "hidden_states", None) else outputs[0]
    lengths = encoded["attention_mask"].to(torch.long).sum(dim=1).clamp(min=1)
    rows = torch.arange(hidden_states.size(0), device=device)
    return hidden_states[rows, lengths.to(device) - 1].float().clone()


@torch.inference_mode()
def qwen_sequence_hidden(
    qwen: nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = encode_batch(tokenizer, texts, max_length, device)
    outputs = qwen(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states[-1] if getattr(outputs, "hidden_states", None) else outputs[0]
    return hidden_states.float().clone(), encoded["attention_mask"].clone()


def cases_to_tensors(cases: Sequence[Any], n_steps: int, max_trace_steps: int, device: torch.device) -> Dict[str, torch.Tensor]:
    operation_ids = torch.tensor([case.operation_ids for case in cases], dtype=torch.long, device=device)
    operation_args = torch.tensor(
        [case.operation_args or [0] * len(case.operation_ids) for case in cases],
        dtype=torch.long,
        device=device,
    )
    initial_labels = torch.tensor([case.initial_label for case in cases], dtype=torch.long, device=device)
    answer_labels = torch.tensor([case.answer_label for case in cases], dtype=torch.long, device=device)
    state_rows: List[List[int]] = []
    mask_rows: List[List[bool]] = []
    final_indices: List[int] = []
    for case in cases:
        labels = [int(value) for value in case.state_labels[:max_trace_steps]]
        if not labels:
            labels = [int(case.answer_label)]
        mask = [True] * len(labels)
        final_indices.append(min(len(labels), max_trace_steps) - 1)
        if len(labels) < max_trace_steps:
            labels = labels + [labels[-1]] * (max_trace_steps - len(labels))
            mask = mask + [False] * (max_trace_steps - len(mask))
        state_rows.append(labels[:max_trace_steps])
        mask_rows.append(mask[:max_trace_steps])
    return {
        "operation_ids": train511.fit_step_sequence(operation_ids, int(n_steps)),
        "operation_arg_ids": train511.fit_step_sequence(operation_args, int(n_steps)),
        "initial_labels": initial_labels,
        "answer_labels": answer_labels,
        "state_labels": torch.tensor(state_rows, dtype=torch.long, device=device),
        "state_mask": torch.tensor(mask_rows, dtype=torch.bool, device=device),
        "final_indices": torch.tensor(final_indices, dtype=torch.long, device=device),
    }


@torch.no_grad()
def sample_candidates(
    model: Any,
    tokenizer: Any,
    cases: Sequence[Any],
    *,
    samples: int,
    n_steps: int,
    max_length: int,
    max_trace_steps: int,
    condition_on_operation_ids: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
    prompts = [case.prompt_text for case in cases]
    encoded = encode_batch(tokenizer, prompts, max_length, device)
    tensors = cases_to_tensors(cases, n_steps, max_trace_steps, device)
    batch = len(cases)
    repeated_out = model(
        input_ids=encoded["input_ids"].repeat_interleave(samples, dim=0),
        attention_mask=encoded["attention_mask"].repeat_interleave(samples, dim=0),
        operation_ids=(
            tensors["operation_ids"].repeat_interleave(samples, dim=0)
            if condition_on_operation_ids
            else None
        ),
        operation_arg_ids=(
            tensors["operation_arg_ids"].repeat_interleave(samples, dim=0)
            if condition_on_operation_ids
            else None
        ),
        initial_labels=tensors["initial_labels"].repeat_interleave(samples, dim=0),
        n_steps=int(n_steps),
        posterior_labels=None,
    )
    answer_logits = repeated_out["answer_logits"].detach().float().reshape(batch, samples, -1)
    candidate_digits = answer_logits.argmax(dim=-1)
    correct = candidate_digits.eq(tensors["answer_labels"].unsqueeze(1))
    metrics = {
        "oracle_accuracy": float(correct.any(dim=1).float().mean().item()),
        "mean_candidate_accuracy": float(correct.float().mean().item()),
    }
    return candidate_digits, tensors, metrics


def build_reader_inputs(
    model: Any,
    tokenizer: Any,
    cases: Sequence[Any],
    candidate_digits: torch.Tensor,
    *,
    reader_mode: str,
    verifier_max_length: int,
    device: torch.device,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    texts = [
        verifier_text(case.prompt_text, digit)
        for case, row in zip(cases, candidate_digits.tolist())
        for digit in row
    ]
    if reader_mode == "sequence_attention":
        states, mask = qwen_sequence_hidden(
            model.qwen,
            tokenizer,
            texts,
            max_length=verifier_max_length,
            device=device,
        )
        return states.detach().clone(), mask.detach().clone()
    return qwen_last_hidden(
        model.qwen,
        tokenizer,
        texts,
        max_length=verifier_max_length,
        device=device,
    ).detach().clone()


def run_verifier(
    verifier: nn.Module,
    reader_inputs: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    candidate_digits: torch.Tensor,
) -> torch.Tensor:
    flat_candidates = candidate_digits.reshape(-1)
    if isinstance(reader_inputs, tuple):
        return verifier(reader_inputs[0], reader_inputs[1], flat_candidates)
    return verifier(reader_inputs, flat_candidates)


def trace_verifier_loss(
    trace_logits: torch.Tensor,
    candidate_digits: torch.Tensor,
    tensors: Dict[str, torch.Tensor],
    samples: int,
    *,
    selection_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    batch = int(candidate_digits.size(0))
    max_steps = int(trace_logits.size(1))
    labels = tensors["state_labels"].repeat_interleave(samples, dim=0)
    mask = tensors["state_mask"].repeat_interleave(samples, dim=0)
    if not bool(mask.any().item()):
        raise ValueError("state mask is empty")
    trace_ce = F.cross_entropy(trace_logits[mask], labels[mask])

    flat_candidates = candidate_digits.reshape(-1)
    final_indices = tensors["final_indices"]
    rows = torch.arange(batch * samples, device=trace_logits.device)
    repeated_final_indices = final_indices.repeat_interleave(samples)
    final_log_probs = F.log_softmax(trace_logits[rows, repeated_final_indices, :].float(), dim=-1)
    candidate_scores = final_log_probs.gather(1, flat_candidates.unsqueeze(1)).squeeze(1).reshape(batch, samples)
    correct = candidate_digits.eq(tensors["answer_labels"].unsqueeze(1))
    oracle = correct.any(dim=1)
    valid = oracle
    if bool(valid.any().item()):
        best_indices = correct[valid].float().argmax(dim=1)
        selection_loss = F.cross_entropy(candidate_scores[valid], best_indices)
    else:
        selection_loss = trace_logits.new_zeros(())

    loss = trace_ce + float(selection_weight) * selection_loss
    with torch.no_grad():
        selected = candidate_scores.argmax(dim=1)
        selected_correct = correct.gather(1, selected.unsqueeze(1)).squeeze(1)
        unrestricted_final_pred = final_log_probs.reshape(batch, samples, -1)[:, 0, :].argmax(dim=-1)
        unrestricted_final_acc = unrestricted_final_pred.eq(tensors["answer_labels"]).float().mean()
        step_pred = trace_logits.argmax(dim=-1)
        step_acc = step_pred[mask].eq(labels[mask]).float().mean()
    return loss, {
        "trace_ce": float(trace_ce.item()),
        "selection_loss": float(selection_loss.item()),
        "selected_accuracy": float(selected_correct.float().mean().item()),
        "oracle_accuracy": float(oracle.float().mean().item()),
        "step_accuracy": float(step_acc.item()),
        "unrestricted_final_accuracy": float(unrestricted_final_acc.item()),
    }


def build_cases(count: int, seed: int, depths: Sequence[int], args: argparse.Namespace) -> List[Any]:
    return train511.build_generalized_synthetic_cases(
        count=int(count),
        seed=int(seed),
        depths=[int(depth) for depth in depths],
        max_steps=max(int(depth) for depth in depths),
        condition_prefix=args.reasoning_condition_prefix,
        family_mix=args.synthetic_family_mix,
        sampling_strategy=args.synthetic_sampling_strategy,
    )


def iter_batches(items: Sequence[Any], batch_size: int) -> Sequence[Sequence[Any]]:
    for start in range(0, len(items), int(batch_size)):
        yield items[start : start + int(batch_size)]


def evaluate(
    model: Any,
    tokenizer: Any,
    verifier: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    *,
    epoch: int,
    writer: SummaryWriter,
) -> Dict[str, Any]:
    verifier.eval()
    summary: Dict[str, Any] = {"depths": {}, "mean_selected_accuracy": 0.0, "mean_oracle_accuracy": 0.0}
    selected_values: List[float] = []
    oracle_values: List[float] = []
    trace_values: List[float] = []
    with torch.no_grad():
        for depth in args.eval_depths:
            cases = build_cases(args.eval_count, args.eval_seed + int(depth), [int(depth)], args)
            selected_total = 0.0
            oracle_total = 0.0
            trace_final_total = 0.0
            step_acc_total = 0.0
            total = 0.0
            for batch_cases in iter_batches(cases, args.batch_size):
                candidate_digits, tensors, _ = sample_candidates(
                    model,
                    tokenizer,
                    batch_cases,
                    samples=args.samples,
                    n_steps=args.model_n_steps,
                    max_length=args.max_length,
                    max_trace_steps=args.max_trace_steps,
                    condition_on_operation_ids=args.condition_on_operation_ids,
                    device=device,
                )
                reader_inputs = build_reader_inputs(
                    model,
                    tokenizer,
                    batch_cases,
                    candidate_digits,
                    reader_mode=args.reader_mode,
                    verifier_max_length=args.verifier_max_length,
                    device=device,
                )
                trace_logits = run_verifier(verifier, reader_inputs, candidate_digits)
                _, metrics = trace_verifier_loss(
                    trace_logits,
                    candidate_digits,
                    tensors,
                    args.samples,
                    selection_weight=args.selection_loss_weight,
                )
                bsz = len(batch_cases)
                selected_total += metrics["selected_accuracy"] * bsz
                oracle_total += metrics["oracle_accuracy"] * bsz
                trace_final_total += metrics["unrestricted_final_accuracy"] * bsz
                step_acc_total += metrics["step_accuracy"] * bsz
                total += float(bsz)
            selected_acc = selected_total / total if total else 0.0
            oracle_acc = oracle_total / total if total else 0.0
            trace_final_acc = trace_final_total / total if total else 0.0
            step_acc = step_acc_total / total if total else 0.0
            selected_values.append(selected_acc)
            oracle_values.append(oracle_acc)
            trace_values.append(trace_final_acc)
            summary["depths"][str(depth)] = {
                "selected_accuracy": selected_acc,
                "oracle_accuracy": oracle_acc,
                "trace_final_accuracy": trace_final_acc,
                "step_accuracy": step_acc,
                "total": int(total),
            }
            writer.add_scalar("Stage51/Eval/SelectedAccuracy", selected_acc, epoch * 100 + int(depth))
            writer.add_scalar("Stage51/Eval/OracleAccuracy", oracle_acc, epoch * 100 + int(depth))
            writer.add_scalar("Stage51/Eval/TraceFinalAccuracy", trace_final_acc, epoch * 100 + int(depth))
            writer.add_scalar(f"Stage51/Eval/Depth_{depth}/SelectedAccuracy", selected_acc, epoch)
            writer.add_scalar(f"Stage51/Eval/Depth_{depth}/OracleAccuracy", oracle_acc, epoch)
            writer.add_scalar(f"Stage51/Eval/Depth_{depth}/TraceFinalAccuracy", trace_final_acc, epoch)
            print(
                f"Eval epoch {epoch:3d} depth={int(depth):2d} "
                f"selected={selected_acc:.4f} oracle={oracle_acc:.4f} "
                f"trace_final={trace_final_acc:.4f} step={step_acc:.4f}",
                flush=True,
            )
    summary["mean_selected_accuracy"] = sum(selected_values) / len(selected_values) if selected_values else 0.0
    summary["mean_oracle_accuracy"] = sum(oracle_values) / len(oracle_values) if oracle_values else 0.0
    summary["mean_trace_final_accuracy"] = sum(trace_values) / len(trace_values) if trace_values else 0.0
    writer.add_scalar("Stage51/Eval/MeanSelectedAccuracy", summary["mean_selected_accuracy"], epoch)
    writer.add_scalar("Stage51/Eval/MeanOracleAccuracy", summary["mean_oracle_accuracy"], epoch)
    writer.add_scalar("Stage51/Eval/MeanTraceFinalAccuracy", summary["mean_trace_final_accuracy"], epoch)
    writer.flush()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--answer-path", choices=("state_head", "lm_head"), default="lm_head")
    parser.add_argument("--workspace-pooling", choices=("mean", "last", "attention", "sequence", "none"), default="sequence")
    parser.add_argument("--core-impl", choices=("state_transition", "hybrid_state_transition"), default="state_transition")
    parser.add_argument("--core-update", choices=("mlp", "mini_gated_delta"), default="mlp")
    parser.add_argument("--state-update-schedule", choices=("nested", "two_stream"), default="nested")
    parser.add_argument("--recurrent-readout-pooling", choices=("final", "mean", "attention", "sharp_attention", "hybrid_gate"), default="sharp_attention")
    parser.add_argument("--recurrent-readout-temperature", type=float, default=0.25)
    parser.add_argument("--model-n-steps", type=int, default=10)
    parser.add_argument("--max-trace-steps", type=int, default=14)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--eval-count", type=int, default=128)
    parser.add_argument("--train-depths", type=int, nargs="+", default=[4, 6, 8, 10])
    parser.add_argument("--eval-depths", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14])
    parser.add_argument("--reasoning-condition-prefix", default="synth")
    parser.add_argument("--synthetic-family-mix", default="balanced")
    parser.add_argument("--synthetic-sampling-strategy", default="random")
    parser.add_argument("--condition-on-operation-ids", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--verifier-max-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--reader-mode", choices=("last_hidden", "sequence_attention"), default="last_hidden")
    parser.add_argument("--selection-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument("--eval-seed", type=int, default=10042)
    parser.add_argument("--override-transition-scale", type=float, default=0.05)
    parser.add_argument("--override-injection-gate-logit", type=float, default=3.0)
    parser.add_argument("--zero-step-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-step-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stochastic-high-level-guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stochastic-transition-mode", choices=("delta", "true_gram"), default="true_gram")
    parser.add_argument("--stochastic-high-level-scale", type=float, default=1.0)
    parser.add_argument("--stochastic-high-level-min-std", type=float, default=1e-4)
    parser.add_argument("--stochastic-high-level-max-std", type=float, default=1.0)
    parser.add_argument("--stochastic-high-level-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stochastic-posterior-guidance", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.samples <= 1:
        raise ValueError("--samples must be > 1")
    if max([*args.train_depths, *args.eval_depths]) > args.max_trace_steps:
        raise ValueError("--max-trace-steps must cover train/eval depths")

    configure_reproducibility(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "logs"))

    model, tokenizer = build_qwen_state_transition_model(
        args.qwen_model_id,
        freeze_qwen=True,
        device=device,
        core_impl=args.core_impl,
        core_update=args.core_update,
        answer_path=args.answer_path,
        workspace_pooling=args.workspace_pooling,
        recurrent_readout_pooling=args.recurrent_readout_pooling,
        recurrent_readout_temperature=args.recurrent_readout_temperature,
        n_steps=args.model_n_steps,
        state_update_schedule=args.state_update_schedule,
        stochastic_high_level_guidance=args.stochastic_high_level_guidance,
        stochastic_high_level_scale=args.stochastic_high_level_scale,
        stochastic_high_level_min_std=args.stochastic_high_level_min_std,
        stochastic_high_level_max_std=args.stochastic_high_level_max_std,
        stochastic_high_level_eval=args.stochastic_high_level_eval,
        stochastic_posterior_guidance=args.stochastic_posterior_guidance,
        stochastic_transition_mode=args.stochastic_transition_mode,
    )
    load_stats = train511.load_flexible_checkpoint(model, args.checkpoint, device)
    override_stats = apply_recurrent_overrides(model, args)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    verifier_cls = CandidateTraceSequenceVerifier if args.reader_mode == "sequence_attention" else CandidateTraceVerifier
    verifier = verifier_cls(
        hidden_size=int(model.hidden_size),
        max_steps=int(args.max_trace_steps),
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(verifier.parameters(), lr=float(args.lr))

    train_cases = build_cases(args.train_count, args.seed, args.train_depths, args)
    with open(os.path.join(args.out_dir, "run_info.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "args": vars(args),
                "load_stats": load_stats,
                "override_stats": override_stats,
                "train_cases": [asdict(case) for case in train_cases[:3]],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    best_eval = float("-inf")
    best_summary: Optional[Dict[str, Any]] = None
    for epoch in range(1, int(args.epochs) + 1):
        random.Random(args.seed + epoch).shuffle(train_cases)
        verifier.train()
        total_loss = 0.0
        total_selected = 0.0
        total_oracle = 0.0
        total_step_acc = 0.0
        total_trace_final = 0.0
        total = 0.0
        for batch_cases in iter_batches(train_cases, args.batch_size):
            candidate_digits, tensors, _ = sample_candidates(
                model,
                tokenizer,
                batch_cases,
                samples=args.samples,
                n_steps=args.model_n_steps,
                max_length=args.max_length,
                max_trace_steps=args.max_trace_steps,
                condition_on_operation_ids=args.condition_on_operation_ids,
                device=device,
            )
            reader_inputs = build_reader_inputs(
                model,
                tokenizer,
                batch_cases,
                candidate_digits,
                reader_mode=args.reader_mode,
                verifier_max_length=args.verifier_max_length,
                device=device,
            )
            trace_logits = run_verifier(verifier, reader_inputs, candidate_digits)
            loss, metrics = trace_verifier_loss(
                trace_logits,
                candidate_digits,
                tensors,
                args.samples,
                selection_weight=args.selection_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(verifier.parameters(), 1.0)
            optimizer.step()

            bsz = len(batch_cases)
            total_loss += float(loss.item()) * bsz
            total_selected += metrics["selected_accuracy"] * bsz
            total_oracle += metrics["oracle_accuracy"] * bsz
            total_step_acc += metrics["step_accuracy"] * bsz
            total_trace_final += metrics["unrestricted_final_accuracy"] * bsz
            total += float(bsz)

        train_loss = total_loss / total if total else 0.0
        train_selected = total_selected / total if total else 0.0
        train_oracle = total_oracle / total if total else 0.0
        train_step_acc = total_step_acc / total if total else 0.0
        train_trace_final = total_trace_final / total if total else 0.0
        writer.add_scalar("Stage51/Train/Loss", train_loss, epoch)
        writer.add_scalar("Stage51/Train/SelectedAccuracy", train_selected, epoch)
        writer.add_scalar("Stage51/Train/OracleAccuracy", train_oracle, epoch)
        writer.add_scalar("Stage51/Train/StepAccuracy", train_step_acc, epoch)
        writer.add_scalar("Stage51/Train/TraceFinalAccuracy", train_trace_final, epoch)
        print(
            f"Epoch {epoch:3d} | loss={train_loss:.4f} | selected={train_selected:.4f} "
            f"| oracle={train_oracle:.4f} | trace_final={train_trace_final:.4f} "
            f"| step={train_step_acc:.4f}",
            flush=True,
        )

        eval_summary = evaluate(model, tokenizer, verifier, args, device, epoch=epoch, writer=writer)
        if float(eval_summary["mean_selected_accuracy"]) > best_eval:
            best_eval = float(eval_summary["mean_selected_accuracy"])
            best_summary = eval_summary
            torch.save(
                {
                    "verifier": verifier.state_dict(),
                    "args": vars(args),
                    "eval_summary": eval_summary,
                },
                os.path.join(args.out_dir, "best_trace_verifier.pt"),
            )
        with open(os.path.join(args.out_dir, "latest_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_selected_accuracy": train_selected,
                    "train_oracle_accuracy": train_oracle,
                    "train_step_accuracy": train_step_acc,
                    "train_trace_final_accuracy": train_trace_final,
                    "eval": eval_summary,
                    "best_eval_mean_accuracy": best_eval,
                    "best_summary": best_summary,
                },
                handle,
                indent=2,
            )
    writer.close()


if __name__ == "__main__":
    main()
