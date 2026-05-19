#!/usr/bin/env python3
"""
Qwen Backbone + State-Transition-First Training.

Trains the state-transition-first architecture with a real Qwen backbone
as prompt compressor. This is the canonical integration path.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/510_train_qwen_state_transition.py \
        --qwen-model-id Qwen/Qwen3.5-2B \
        --out-dir local_eval/qwen_state_transition_s80_20260519 \
        --epochs 20 \
        --train-cases 4096 \
        --eval-cases 192 \
        --eval-seed 9337
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from qtrm_mm.state_transition_core import N_OPERATIONS
from qtrm_mm.losses import (
    state_transition_loss,
    state_transition_causality_loss,
    state_monotonic_improvement_loss,
)


# =============================================================================
# Synthetic Data Generation (same as script 500)
# =============================================================================

@dataclass(frozen=True)
class SyntheticCase:
    prompt: str
    label: str
    family: str
    state_targets: list[int]
    operation_ids: list[int]

OP_ADD = 0
OP_MUL = 1
OP_SUB = 2
OP_FINAL = 3

_CHAIN5_RE = re.compile(
    r"Start (\d+); add (\d+); multiply by (\d+); subtract (\d+); add (\d+)"
)
_CHECKSUM4_RE = re.compile(r"a=(\d+), b=(\d+), c=(\d+), d=(\d+)")


def build_synthetic_cases(
    count: int,
    seed: int,
    case_mode: str = "hard_v1",
) -> list[SyntheticCase]:
    """Build synthetic cases with explicit state targets."""
    rng = random.Random(seed)
    
    if case_mode == "hard_v1":
        families = ("checksum4", "chain5", "select_pair")
    elif case_mode == "hard_v1_balanced":
        families = ("select_pair", "checksum4", "chain5")
    else:
        families = ("checksum4", "chain5", "select_pair")
    
    cases = []
    for idx in range(count):
        family = families[idx % len(families)]
        
        if family == "chain5":
            start = rng.randrange(10)
            add_a = rng.randrange(10)
            mul = rng.choice((1, 3, 7, 9))
            sub = rng.randrange(10)
            add_b = rng.randrange(10)
            
            after_add = (start + add_a) % 10
            after_mul = (after_add * mul) % 10
            after_sub = (after_mul - sub) % 10
            final = (after_sub + add_b) % 10
            
            state_targets = [after_add, after_mul, after_sub, final]
            operation_ids = [OP_ADD, OP_MUL, OP_SUB, OP_ADD]
            
            prompt = (
                "Follow the five-step digit chain mod 10. "
                f"Start {start}; add {add_a}; multiply by {mul}; "
                f"subtract {sub}; add {add_b}. "
                "Answer with one digit. Answer: "
            )
            
        elif family == "checksum4":
            a, b, c, d = (rng.randrange(10) for _ in range(4))
            
            partial_1 = a % 10
            partial_2 = (a + 2 * b) % 10
            partial_3 = (a + 2 * b + 3 * c) % 10
            final = (a + 2 * b + 3 * c + 4 * d) % 10
            
            state_targets = [partial_1, partial_2, partial_3, final]
            operation_ids = [OP_ADD, OP_ADD, OP_ADD, OP_ADD]
            
            prompt = (
                "Compute the extended checksum mod 10. "
                "Rule: (a + 2*b + 3*c + 4*d) mod 10. "
                f"a={a}, b={b}, c={c}, d={d}. "
                "Answer with one digit. Answer: "
            )
            
        elif family == "select_pair":
            digits = [rng.randrange(10) for _ in range(7)]
            first = rng.randrange(len(digits))
            second = rng.randrange(len(digits))
            
            sel_1 = digits[first]
            sel_2 = (sel_1 + digits[second]) % 10
            add_idx = (sel_2 + first) % 10
            final = (add_idx + second) % 10
            
            state_targets = [sel_1, sel_2, add_idx, final]
            operation_ids = [OP_ADD, OP_ADD, OP_ADD, OP_ADD]
            
            prompt = (
                "Read the digit list and answer mod 10. "
                f"Digits: {digits}. Take indices {first} and {second}; "
                "add both selected digits and both indices, mod 10. "
                "Answer with one digit. Answer: "
            )
        else:
            raise ValueError(f"unknown family: {family}")
        
        cases.append(SyntheticCase(
            prompt=prompt,
            label=str(final),
            family=family,
            state_targets=state_targets,
            operation_ids=operation_ids,
        ))
    
    return cases


# =============================================================================
# Dataset
# =============================================================================

class SyntheticDataset(Dataset):
    def __init__(
        self,
        cases: list[SyntheticCase],
        tokenizer,
        max_seq_len: int = 128,
        max_steps: int = 4,
    ):
        self.cases = cases
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_steps = max_steps
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        case = self.cases[idx]
        
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                case.prompt,
                max_length=self.max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = tokens.input_ids.squeeze(0)
            attention_mask = tokens.attention_mask.squeeze(0)
        else:
            # Fallback: simple encoding
            text = case.prompt[:self.max_seq_len]
            input_ids = torch.tensor([ord(c) % 1000 for c in text], dtype=torch.long)
            padding = torch.zeros(self.max_seq_len - len(input_ids), dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
            attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)
        
        # State targets
        state_targets = torch.full((self.max_steps + 1,), -100, dtype=torch.long)
        for t, s in enumerate(case.state_targets[:self.max_steps]):
            state_targets[t + 1] = s
        
        # Operation IDs
        op_ids = torch.full((self.max_steps,), -100, dtype=torch.long)
        for t, o in enumerate(case.operation_ids[:self.max_steps]):
            op_ids[t] = o
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "state_targets": state_targets,
            "operation_ids": op_ids,
            "answer_target": torch.tensor(int(case.label), dtype=torch.long),
            "family": case.family,
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "state_targets": torch.stack([b["state_targets"] for b in batch]),
        "operation_ids": torch.stack([b["operation_ids"] for b in batch]),
        "answer_target": torch.stack([b["answer_target"] for b in batch]),
        "families": [b["family"] for b in batch],
    }


# =============================================================================
# Training
# =============================================================================

def train_epoch(
    model,
    train_loader,
    optimizer,
    args,
    device,
):
    model.train()
    metrics = {
        "total_loss": 0.0,
        "state_loss": 0.0,
        "answer_loss": 0.0,
        "consistency_loss": 0.0,
        "state_accuracy": 0.0,
        "answer_accuracy": 0.0,
    }
    n_batches = 0
    
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        state_targets = batch["state_targets"].to(device)
        operation_ids = batch["operation_ids"].to(device)
        answer_target = batch["answer_target"].to(device)
        
        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            operation_ids=operation_ids,
            n_steps=args.n_steps,
        )
        
        # Primary loss: state transition
        loss, loss_metrics = state_transition_loss(
            state_digit_logits=outputs["state_digit_logits"],
            state_targets=state_targets,
            answer_logits=outputs["answer_logits"],
            answer_targets=answer_target,
            operation_logits=outputs.get("operation_logits"),
            operation_targets=operation_ids,
            state_weight=args.state_weight,
            answer_weight=args.answer_weight,
            consistency_weight=args.consistency_weight,
            operation_weight=args.operation_weight,
        )
        
        # Monotonic improvement
        if args.monotonic_weight > 0.0:
            mono_loss, mono_metrics = state_monotonic_improvement_loss(
                state_digit_logits=outputs["state_digit_logits"],
                answer_targets=answer_target,
                margin=args.monotonic_margin,
                weight=args.monotonic_weight,
            )
            loss = loss + mono_loss
        
        # Causality gate (if baseline logits available)
        if (
            args.causality_weight > 0.0
            and outputs.get("baseline_logits") is not None
            and model.label_token_ids is not None
        ):
            baseline_logits = outputs["baseline_logits"][:, -1, :]  # last token
            label_ids = model.label_token_ids.to(device)
            causal_loss, causal_metrics = state_transition_causality_loss(
                answer_logits=outputs["answer_logits"],
                base_logits=baseline_logits,
                answer_targets=answer_target,
                label_token_ids=label_ids,
                margin=args.causality_margin,
                weight=args.causality_weight,
            )
            loss = loss + causal_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )
        
        optimizer.step()
        
        # Accumulate
        metrics["total_loss"] += loss.item()
        metrics["state_loss"] += loss_metrics["state_loss"].item()
        metrics["answer_loss"] += loss_metrics["answer_loss"].item()
        metrics["consistency_loss"] += loss_metrics["consistency_loss"].item()
        metrics["state_accuracy"] += loss_metrics["state_accuracy"].item()
        metrics["answer_accuracy"] += loss_metrics["answer_accuracy"].item()
        n_batches += 1
    
    # Average
    for key in metrics:
        metrics[key] /= max(n_batches, 1)
    
    return metrics


@torch.no_grad()
def evaluate(
    model,
    loader,
    args,
    device,
) -> dict:
    model.eval()
    
    all_state_acc = []
    all_answer_acc = []
    family_state_acc = {}
    family_answer_acc = []
    
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        state_targets = batch["state_targets"].to(device)
        operation_ids = batch["operation_ids"].to(device)
        answer_target = batch["answer_target"].to(device)
        families = batch["families"]
        
        # Core ON
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            operation_ids=operation_ids,
            n_steps=args.n_steps,
        )
        
        # State accuracy
        state_preds = outputs["state_digit_logits"].argmax(dim=-1)
        valid_mask = state_targets != -100
        if valid_mask.any():
            state_acc = (state_preds == state_targets).float()[valid_mask].mean().item()
            all_state_acc.append(state_acc)
        
        # Answer accuracy
        answer_preds = outputs["answer_logits"].argmax(dim=-1)
        answer_acc = (answer_preds == answer_target).float().mean().item()
        all_answer_acc.append(answer_acc)
        
        # Per-family
        for family in set(families):
            fam_mask = [f == family for f in families]
            fam_valid = valid_mask[fam_mask]
            if fam_valid.any():
                fam_state = (state_preds[fam_mask] == state_targets[fam_mask]).float()[fam_valid].mean().item()
                if family not in family_state_acc:
                    family_state_acc[family] = []
                family_state_acc[family].append(fam_state)
            
            fam_answer = (answer_preds[fam_mask] == answer_target[fam_mask]).float().mean().item()
            if family not in family_answer_acc:
                family_answer_acc[family] = []
            family_answer_acc[family].append(fam_answer)
    
    result = {
        "state_accuracy": sum(all_state_acc) / len(all_state_acc) if all_state_acc else 0.0,
        "answer_accuracy": sum(all_answer_acc) / len(all_answer_acc) if all_answer_acc else 0.0,
        "family_state_accuracy": {
            k: sum(v) / len(v) for k, v in family_state_acc.items()
        },
        "family_answer_accuracy": {
            k: sum(v) / len(v) for k, v in family_answer_acc.items()
        },
    }
    
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen State-Transition Training")
    
    # Model
    parser.add_argument("--qwen-model-id", type=str, required=True,
                       help="Qwen model ID (e.g., Qwen/Qwen3.5-2B)")
    parser.add_argument("--d-state", type=int, default=None,
                       help="State dimension (default: Qwen hidden size)")
    parser.add_argument("--n-steps", type=int, default=4,
                       help="Number of transition steps")
    parser.add_argument("--freeze-qwen", action="store_true", default=True)
    parser.add_argument("--trainable-qwen-layers", type=str, default=None,
                       help="Comma-separated Qwen layer indices to unfreeze")
    
    # Data
    parser.add_argument("--train-cases", type=int, default=4096)
    parser.add_argument("--eval-cases", type=int, default=192)
    parser.add_argument("--train-seed", type=int, default=20260519)
    parser.add_argument("--eval-seed", type=int, default=9337)
    parser.add_argument("--case-mode", type=str, default="hard_v1")
    parser.add_argument("--max-seq-len", type=int, default=128)
    
    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=2)
    
    # Loss weights
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--answer-weight", type=float, default=0.5)
    parser.add_argument("--consistency-weight", type=float, default=0.3)
    parser.add_argument("--operation-weight", type=float, default=0.0)
    parser.add_argument("--monotonic-weight", type=float, default=0.05)
    parser.add_argument("--monotonic-margin", type=float, default=0.0)
    parser.add_argument("--causality-weight", type=float, default=0.2)
    parser.add_argument("--causality-margin", type=float, default=0.1)
    
    # Output
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}
    dtype = dtype_map.get(args.dtype, torch.bfloat16)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Model: {args.qwen_model_id}")
    print(f"Output: {out_dir}")
    
    # Load model
    try:
        from qtrm_mm.qwen_backbone_state_transition import build_qwen_state_transition_model
        model, tokenizer = build_qwen_state_transition_model(
            args.qwen_model_id,
            d_state=args.d_state,
            n_operations=N_OPERATIONS,
            n_steps=args.n_steps,
            freeze_qwen=args.freeze_qwen,
            max_seq_len=args.max_seq_len,
            dtype=dtype,
            device=device,
        )
        print(f"Loaded Qwen model: {args.qwen_model_id}")
    except Exception as e:
        print(f"Failed to load Qwen model: {e}")
        print("Falling back to random initialization...")
        sys.exit(1)
    
    # Optionally unfreeze some Qwen layers
    if args.trainable_qwen_layers:
        layer_indices = [int(x.strip()) for x in args.trainable_qwen_layers.split(",")]
        model.set_qwen_partial_trainable(layer_indices=layer_indices)
        print(f"Unfrozen Qwen layers: {layer_indices}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    qwen_params = sum(p.numel() for p in model.qwen.parameters())
    qwen_trainable = sum(p.numel() for p in model.qwen.parameters() if p.requires_grad)
    
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Qwen params: {qwen_params:,} (trainable: {qwen_trainable:,})")
    
    # Build datasets
    print("Building datasets...")
    train_cases = build_synthetic_cases(
        count=args.train_cases,
        seed=args.train_seed,
        case_mode=args.case_mode,
    )
    eval_cases = build_synthetic_cases(
        count=args.eval_cases,
        seed=args.eval_seed,
        case_mode=args.case_mode,
    )
    
    train_dataset = SyntheticDataset(
        train_cases, tokenizer, args.max_seq_len, args.n_steps
    )
    eval_dataset = SyntheticDataset(
        eval_cases, tokenizer, args.max_seq_len, args.n_steps
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )
    
    # Training loop
    print("Starting training...")
    best_state_acc = 0.0
    best_answer_acc = 0.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, args, device)
        
        for _ in range(len(train_loader)):
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Evaluate
        eval_metrics = {}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            eval_metrics = evaluate(model, eval_loader, args, device)
            
            if eval_metrics.get("state_accuracy", 0) > best_state_acc:
                best_state_acc = eval_metrics["state_accuracy"]
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "metrics": eval_metrics,
                }, out_dir / "best_state.pt")
            
            if eval_metrics.get("answer_accuracy", 0) > best_answer_acc:
                best_answer_acc = eval_metrics["answer_accuracy"]
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "metrics": eval_metrics,
                }, out_dir / "best_answer.pt")
        
        # Log
        log_entry = {
            "epoch": epoch,
            "time": epoch_time,
            "train": train_metrics,
            "eval": eval_metrics,
            "best_state_acc": best_state_acc,
            "best_answer_acc": best_answer_acc,
            "lr": scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else args.lr,
        }
        history.append(log_entry)
        
        # Print
        eval_str = ""
        if eval_metrics:
            eval_str = (
                f" | eval_state_acc={eval_metrics['state_accuracy']:.4f} "
                f"eval_answer_acc={eval_metrics['answer_accuracy']:.4f}"
            )
            fam_strs = []
            for fam in sorted(eval_metrics.get("family_answer_accuracy", {}).keys()):
                fam_strs.append(
                    f"{fam}:{eval_metrics['family_answer_accuracy'][fam]:.4f}"
                )
            if fam_strs:
                eval_str += f" | families={','.join(fam_strs)}"
        
        print(
            f"Epoch {epoch:3d} | loss={train_metrics['total_loss']:.4f} "
            f"state_loss={train_metrics['state_loss']:.4f} "
            f"answer_loss={train_metrics['answer_loss']:.4f} "
            f"state_acc={train_metrics['state_accuracy']:.4f} "
            f"answer_acc={train_metrics['answer_accuracy']:.4f}"
            f"{eval_str}"
        )
        
        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "train_metrics": train_metrics,
        }, out_dir / "last.pt")
        
        # Save history
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)
        
        # Save report
        final_report = {
            "config": vars(args),
            "history": history,
            "best_state_accuracy": best_state_acc,
            "best_answer_accuracy": best_answer_acc,
            "final_eval": eval_metrics if eval_metrics else None,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }
        with open(out_dir / "report.json", "w") as f:
            json.dump(final_report, f, indent=2, default=str)
    
    print(f"\nTraining complete!")
    print(f"Best state accuracy: {best_state_acc:.4f}")
    print(f"Best answer accuracy: {best_answer_acc:.4f}")
    print(f"Final report: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
