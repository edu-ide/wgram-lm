"""Train a local reader-verifier for GRAM sampled candidate answers.

Stage50 is intentionally separated from the main training script. The recurrent
thinker is frozen; this script tests whether a verifier that rereads the source
prompt plus each sampled candidate answer can select the correct trajectory.
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

from wgram_lm.norm import RMSNorm
from wgram_lm.qwen_backbone_state_transition import build_qwen_state_transition_model


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


class CandidateReaderVerifier(nn.Module):
    """Small verifier over reread prompt/candidate hidden plus trajectory summary."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


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
        "Verifier: reread the task and candidate answer above. "
        f"Is candidate final digit {int(candidate_digit)} correct? Answer yes or no:"
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
    return hidden_states[rows, lengths.to(device) - 1].float()


def cases_to_tensors(cases: Sequence[Any], n_steps: int, device: torch.device) -> Dict[str, torch.Tensor]:
    operation_ids = torch.tensor([case.operation_ids for case in cases], dtype=torch.long, device=device)
    operation_args = torch.tensor(
        [case.operation_args or [0] * len(case.operation_ids) for case in cases],
        dtype=torch.long,
        device=device,
    )
    initial_labels = torch.tensor([case.initial_label for case in cases], dtype=torch.long, device=device)
    answer_labels = torch.tensor([case.answer_label for case in cases], dtype=torch.long, device=device)
    return {
        "operation_ids": train511.fit_step_sequence(operation_ids, int(n_steps)),
        "operation_arg_ids": train511.fit_step_sequence(operation_args, int(n_steps)),
        "initial_labels": initial_labels,
        "answer_labels": answer_labels,
    }


def build_candidate_features(
    model: Any,
    tokenizer: Any,
    cases: Sequence[Any],
    *,
    samples: int,
    n_steps: int,
    max_length: int,
    verifier_max_length: int,
    condition_on_operation_ids: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    prompts = [case.prompt_text for case in cases]
    encoded = encode_batch(tokenizer, prompts, max_length, device)
    tensors = cases_to_tensors(cases, n_steps, device)
    batch = len(cases)

    with torch.no_grad():
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
    labels = tensors["answer_labels"].to(torch.long)
    correct = candidate_digits.eq(labels.unsqueeze(1)).float()

    flat_digits = candidate_digits.reshape(-1).tolist()
    verifier_texts = [
        verifier_text(case.prompt_text, digit)
        for case, row in zip(cases, candidate_digits.tolist())
        for digit in row
    ]
    reader_hidden = qwen_last_hidden(
        model.qwen,
        tokenizer,
        verifier_texts,
        max_length=verifier_max_length,
        device=device,
    ).clone()

    trajectory = repeated_out["qtrm_core_step_states"].detach().float()
    readout = repeated_out["qtrm_readout_state"].detach().float()
    mean_state = trajectory[:, 1:, :].mean(dim=1)
    final_state = trajectory[:, -1, :]
    probs = torch.softmax(answer_logits.reshape(batch * samples, -1), dim=-1)
    one_hot = F.one_hot(torch.tensor(flat_digits, device=device), num_classes=answer_logits.size(-1)).float()
    margins = probs.topk(k=2, dim=-1).values
    margin = (margins[:, 0] - margins[:, 1]).unsqueeze(-1)
    entropy = (-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)).unsqueeze(-1)

    features = torch.cat(
        [
            reader_hidden,
            readout,
            mean_state,
            final_state - trajectory[:, 0, :],
            answer_logits.reshape(batch * samples, -1),
            one_hot,
            margin,
            entropy,
        ],
        dim=-1,
    )
    metrics = {
        "oracle_accuracy": float(correct.any(dim=1).float().mean().item()),
        "mean_candidate_accuracy": float(correct.mean().item()),
    }
    return features.detach().clone(), correct.reshape(-1).detach().clone(), metrics


def verifier_loss(scores: torch.Tensor, targets: torch.Tensor, samples: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    batch = scores.numel() // int(samples)
    score_matrix = scores.reshape(batch, samples)
    target_matrix = targets.reshape(batch, samples)
    bce = F.binary_cross_entropy_with_logits(scores, targets)
    spread = target_matrix.max(dim=1).values - target_matrix.min(dim=1).values
    valid = spread > 0.0
    if bool(valid.any().item()):
        best_indices = target_matrix[valid].argmax(dim=1)
        listwise = F.cross_entropy(score_matrix[valid], best_indices)
    else:
        listwise = scores.new_zeros(())
    target_delta = target_matrix.unsqueeze(2) - target_matrix.unsqueeze(1)
    score_delta = score_matrix.unsqueeze(2) - score_matrix.unsqueeze(1)
    pair_mask = target_delta > 0.0
    if bool(pair_mask.any().item()):
        pairwise = F.softplus(-score_delta[pair_mask]).mean()
    else:
        pairwise = scores.new_zeros(())
    loss = 0.2 * bce + listwise + pairwise
    with torch.no_grad():
        selected = score_matrix.argmax(dim=1)
        selected_correct = target_matrix.gather(1, selected.unsqueeze(1)).squeeze(1)
        oracle = target_matrix.max(dim=1).values
    return loss, {
        "selected_accuracy": float(selected_correct.mean().item()),
        "oracle_accuracy": float(oracle.mean().item()),
        "target_spread": float(spread.mean().item()),
        "bce": float(bce.item()),
        "listwise": float(listwise.item()),
        "pairwise": float(pairwise.item()),
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
    verifier: CandidateReaderVerifier,
    args: argparse.Namespace,
    device: torch.device,
    *,
    epoch: int,
    writer: SummaryWriter,
) -> Dict[str, Any]:
    verifier.eval()
    summary: Dict[str, Any] = {"depths": {}, "mean_accuracy": 0.0, "mean_oracle_accuracy": 0.0}
    selected_values: List[float] = []
    oracle_values: List[float] = []
    with torch.no_grad():
        for depth in args.eval_depths:
            cases = build_cases(args.eval_count, args.eval_seed + int(depth), [int(depth)], args)
            selected_total = 0.0
            oracle_total = 0.0
            total = 0.0
            for batch_cases in iter_batches(cases, args.batch_size):
                features, targets, _ = build_candidate_features(
                    model,
                    tokenizer,
                    batch_cases,
                    samples=args.samples,
                    n_steps=args.model_n_steps,
                    max_length=args.max_length,
                    verifier_max_length=args.verifier_max_length,
                    condition_on_operation_ids=args.condition_on_operation_ids,
                    device=device,
                )
                scores = verifier(features)
                bsz = len(batch_cases)
                score_matrix = scores.reshape(bsz, args.samples)
                target_matrix = targets.reshape(bsz, args.samples)
                selected = score_matrix.argmax(dim=1)
                selected_correct = target_matrix.gather(1, selected.unsqueeze(1)).squeeze(1)
                oracle = target_matrix.max(dim=1).values
                selected_total += float(selected_correct.sum().item())
                oracle_total += float(oracle.sum().item())
                total += float(bsz)
            selected_acc = selected_total / total if total else 0.0
            oracle_acc = oracle_total / total if total else 0.0
            summary["depths"][str(depth)] = {
                "selected_accuracy": selected_acc,
                "oracle_accuracy": oracle_acc,
                "total": int(total),
            }
            selected_values.append(selected_acc)
            oracle_values.append(oracle_acc)
            writer.add_scalar("Stage50/Eval/SelectedAccuracy", selected_acc, epoch * 100 + int(depth))
            writer.add_scalar("Stage50/Eval/OracleAccuracy", oracle_acc, epoch * 100 + int(depth))
            writer.add_scalar(f"Stage50/Eval/Depth_{depth}/SelectedAccuracy", selected_acc, epoch)
            writer.add_scalar(f"Stage50/Eval/Depth_{depth}/OracleAccuracy", oracle_acc, epoch)
            print(
                f"Eval epoch {epoch:3d} depth={int(depth):2d} "
                f"selected={selected_acc:.4f} oracle={oracle_acc:.4f}",
                flush=True,
            )
    summary["mean_accuracy"] = sum(selected_values) / len(selected_values) if selected_values else 0.0
    summary["mean_oracle_accuracy"] = sum(oracle_values) / len(oracle_values) if oracle_values else 0.0
    writer.add_scalar("Stage50/Eval/MeanSelectedAccuracy", summary["mean_accuracy"], epoch)
    writer.add_scalar("Stage50/Eval/MeanOracleAccuracy", summary["mean_oracle_accuracy"], epoch)
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
    parser.add_argument("--seed", type=int, default=92)
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

    input_dim = int(model.hidden_size) + int(model.d_state) * 3 + 22
    verifier = CandidateReaderVerifier(input_dim=input_dim, hidden_dim=int(args.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(verifier.parameters(), lr=float(args.lr))

    train_cases = build_cases(args.train_count, args.seed, args.train_depths, args)
    with open(os.path.join(args.out_dir, "run_info.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "args": vars(args),
                "load_stats": load_stats,
                "override_stats": override_stats,
                "input_dim": input_dim,
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
        total = 0.0
        for batch_cases in iter_batches(train_cases, args.batch_size):
            features, targets, feature_metrics = build_candidate_features(
                model,
                tokenizer,
                batch_cases,
                samples=args.samples,
                n_steps=args.model_n_steps,
                max_length=args.max_length,
                verifier_max_length=args.verifier_max_length,
                condition_on_operation_ids=args.condition_on_operation_ids,
                device=device,
            )
            scores = verifier(features)
            loss, metrics = verifier_loss(scores, targets, args.samples)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(verifier.parameters(), 1.0)
            optimizer.step()

            bsz = len(batch_cases)
            total_loss += float(loss.item()) * bsz
            total_selected += float(metrics["selected_accuracy"]) * bsz
            total_oracle += float(metrics["oracle_accuracy"]) * bsz
            total += float(bsz)

        train_loss = total_loss / total if total else 0.0
        train_selected = total_selected / total if total else 0.0
        train_oracle = total_oracle / total if total else 0.0
        writer.add_scalar("Stage50/Train/Loss", train_loss, epoch)
        writer.add_scalar("Stage50/Train/SelectedAccuracy", train_selected, epoch)
        writer.add_scalar("Stage50/Train/OracleAccuracy", train_oracle, epoch)
        print(
            f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
            f"selected={train_selected:.4f} | oracle={train_oracle:.4f}",
            flush=True,
        )

        eval_summary = evaluate(model, tokenizer, verifier, args, device, epoch=epoch, writer=writer)
        if float(eval_summary["mean_accuracy"]) > best_eval:
            best_eval = float(eval_summary["mean_accuracy"])
            best_summary = eval_summary
            torch.save(
                {
                    "verifier": verifier.state_dict(),
                    "args": vars(args),
                    "input_dim": input_dim,
                    "eval_summary": eval_summary,
                },
                os.path.join(args.out_dir, "best_verifier.pt"),
            )
        with open(os.path.join(args.out_dir, "latest_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_selected_accuracy": train_selected,
                    "train_oracle_accuracy": train_oracle,
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
