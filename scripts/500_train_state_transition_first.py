#!/usr/bin/env python3
"""
State-Transition-First Training Script.

Trains a state machine where intermediate states are the PRIMARY
supervised target, not a residual adapter. This attacks the root
cause of the chain5 non-regression problem.

Architecture:
    prompt tokens -> tokenizer -> Qwen backbone -> workspace compression
    -> z_0 (initial state) -> operation-conditioned transitions
    -> z_T (final state) -> answer head -> answer logits

Loss:
    PRIMARY: State prediction at each depth step (cross-entropy)
    SECONDARY: Final answer from last state (cross-entropy)
    TERTIARY: State-answer consistency
    QUATERNARY: Causality gate (state path beats donor-only)
"""

from __future__ import annotations

import argparse
import json
import math
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

from qtrm_mm.config import QTRMConfig
from qtrm_mm.state_transition_core import (
    StateTransitionCore,
    StateTransitionOutput,
)
from qtrm_mm.losses import (
    state_transition_loss,
    state_transition_causality_loss,
    state_monotonic_improvement_loss,
)


# =============================================================================
# Synthetic Data Generation
# =============================================================================

@dataclass(frozen=True)
class SyntheticCase:
    prompt: str
    label: str
    family: str
    state_targets: list[int]  # intermediate state labels
    operation_ids: list[int]  # operation IDs for each step


# Operation codes
OP_ADD = 0
OP_MUL = 1
OP_SUB = 2
OP_FINAL = 3
N_OPERATIONS = 4

_CHAIN5_RE = re.compile(
    r"Start (\d+); add (\d+); multiply by (\d+); subtract (\d+); add (\d+)"
)
_CHECKSUM4_RE = re.compile(r"a=(\d+), b=(\d+), c=(\d+), d=(\d+)")
_SELECT_PAIR_RE = re.compile(
    r"Digits: \[([\d, ]+)\].*Take indices (\d+) and (\d+)"
)


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


class SyntheticDataset(Dataset):
    """Dataset for state-transition training."""
    
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
        
        # Tokenize prompt
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
            # Simple encoding for random init
            text = case.prompt[:self.max_seq_len]
            input_ids = torch.tensor([ord(c) % 1000 for c in text], dtype=torch.long)
            # Pad to max_seq_len
            padding = torch.zeros(self.max_seq_len - len(input_ids), dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
            attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)
        
        
        # State targets: pad to max_steps
        state_targets = torch.full(
            (self.max_steps + 1,), -100, dtype=torch.long
        )
        for t, s in enumerate(case.state_targets[:self.max_steps]):
            state_targets[t + 1] = s  # offset by 1 for initial state
        
        # Operation IDs: pad to max_steps
        op_ids = torch.full(
            (self.max_steps,), -100, dtype=torch.long
        )
        for t, o in enumerate(case.operation_ids[:self.max_steps]):
            op_ids[t] = o
        
        # Answer target
        answer_target = torch.tensor(int(case.label), dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "state_targets": state_targets,
            "operation_ids": op_ids,
            "answer_target": answer_target,
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
# Model Wrapper
# =============================================================================

class StateTransitionModel(torch.nn.Module):
    """
    Complete state-transition-first model with Qwen backbone.
    
    Uses the Qwen backbone as a prompt compressor (frozen or partially
    trainable), then feeds the compressed representation into the
    state transition core.
    """
    
    def __init__(
        self,
        qwen_model,
        d_state: int = 256,
        n_operations: int = N_OPERATIONS,
        n_steps: int = 4,
        freeze_qwen: bool = True,
        trainable_layers: Optional[list[int]] = None,
    ):
        super().__init__()
        
        # Qwen backbone as prompt compressor
        self.qwen = qwen_model
        if freeze_qwen:
            for param in self.qwen.parameters():
                param.requires_grad = False
            # Optionally unfreeze some layers
            if trainable_layers:
                text_model = self._find_text_model()
                layers = getattr(text_model, "layers", None)
                if layers:
                    for idx in trainable_layers:
                        for param in layers[idx].parameters():
                            param.requires_grad = True
        
        # Get model dimensions
        config = self._find_config()
        self.d_qwen = config.hidden_size if hasattr(config, 'hidden_size') else 768
        self.vocab_size = config.vocab_size if hasattr(config, 'vocab_size') else 152064
        
        # Compressor: Qwen hidden -> state space
        self.compressor = torch.nn.Sequential(
            torch.nn.LayerNorm(self.d_qwen),
            torch.nn.Linear(self.d_qwen, d_state),
            torch.nn.GELU(),
            torch.nn.Linear(d_state, d_state),
        )
        
        # State transition core
        qtrm_cfg = QTRMConfig(
            d_model=d_state,
            num_actions=n_operations,
            outer_steps=n_steps,
        )
        self.core = StateTransitionCore(
            cfg=qtrm_cfg,
            d_state=d_state,
            n_operations=n_operations,
            n_steps=n_steps,
        )
        
        # Store for label token IDs
        self.label_token_ids = None
        self._setup_label_tokens(tokenizer=None)
    
    def _find_text_model(self):
        for path in ("model.language_model", "language_model", "model", ""):
            candidate = self.qwen
            for part in path.split("."):
                if part:
                    candidate = getattr(candidate, part, None)
                if candidate is None:
                    break
            if candidate is not None and hasattr(candidate, "layers"):
                return candidate
        return self.qwen
    
    def _find_config(self):
        for attr in ("config", "model.config", "model.language_model.config"):
            candidate = self.qwen
            for part in attr.split("."):
                if part:
                    candidate = getattr(candidate, part, None)
                if candidate is None:
                    break
            if candidate is not None and hasattr(candidate, "hidden_size"):
                return candidate
        return type('Config', (), {
            'hidden_size': self.d_qwen,
            'vocab_size': self.vocab_size,
        })()
    
    def _setup_label_tokens(self, tokenizer):
        """Setup digit token IDs for the label choices."""
        if tokenizer is not None:
            digit_chars = [str(i) for i in range(10)]
            self.label_token_ids = torch.tensor(
                [tokenizer.encode(d, add_special_tokens=False)[0] for d in digit_chars],
                dtype=torch.long,
            )
        else:
            # Will be set later
            self.label_token_ids = None
    
    def set_label_token_ids(self, token_ids):
        """Set label token IDs after tokenizer is available."""
        self.label_token_ids = torch.tensor(token_ids, dtype=torch.long)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        force_core_off: bool = False,
        operation_ids: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
    ):
        """
        Forward pass.
        
        Returns:
            dict with logits, core outputs, and telemetry
        """
        # Qwen backbone forward
        with torch.set_grad_enabled(not all(
            not p.requires_grad for p in self.qwen.parameters()
        )):
            qwen_outputs = self.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        
        # Get last hidden state
        if hasattr(qwen_outputs, 'hidden_states') and qwen_outputs.hidden_states:
            hidden = qwen_outputs.hidden_states[-1]
        elif hasattr(qwen_outputs, 'last_hidden_state'):
            hidden = qwen_outputs.last_hidden_state
        else:
            hidden = qwen_outputs[0] if isinstance(qwen_outputs, tuple) else qwen_outputs
        
        # Compress to workspace
        # Use mean-pooling over attended tokens for initial workspace
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            workspace = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            workspace = workspace.unsqueeze(1)  # (B, 1, d_state)
        else:
            workspace = hidden.mean(dim=1, keepdim=True)
        
        workspace = self.compressor(workspace)
        
        if force_core_off:
            # For core-off baseline: use Qwen logits directly
            return {
                "logits": qwen_outputs.logits if hasattr(qwen_outputs, 'logits') else None,
                "qtrm_core_step_states": None,
                "state_digit_logits": None,
                "answer_logits": None,
                "operation_logits": None,
            }
        
        # State transition core
        core_output = self.core(
            workspace=workspace,
            operation_ids=operation_ids,
            n_steps=n_steps,
        )
        
        return {
            "logits": qwen_outputs.logits if hasattr(qwen_outputs, 'logits') else None,
            "qtrm_core_step_states": core_output.state_trajectory,
            "state_digit_logits": core_output.state_digit_logits,
            "answer_logits": core_output.answer_logits,
            "operation_logits": core_output.operation_logits,
            "state_norms": core_output.state_norms,
            "transition_norms": core_output.transition_norms,
            "state_cosines": core_output.state_cosines,
        }


# =============================================================================
# Training Loop
# =============================================================================

def train(
    model: StateTransitionModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    state_loss_sum = 0.0
    answer_loss_sum = 0.0
    consistency_loss_sum = 0.0
    state_acc_sum = 0.0
    answer_acc_sum = 0.0
    n_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
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
        
        # Compute loss
        loss_dict = state_transition_loss(
            state_digit_logits=outputs["state_digit_logits"],
            state_targets=state_targets,
            answer_logits=outputs["answer_logits"],
            answer_targets=answer_target,
            operation_logits=outputs["operation_logits"],
            operation_targets=operation_ids,
            state_weight=args.state_weight,
            answer_weight=args.answer_weight,
            consistency_weight=args.consistency_weight,
            operation_weight=args.operation_weight,
        )
        loss, metrics = loss_dict
        
        # Monotonic improvement
        if args.monotonic_weight > 0.0:
            mono_loss, mono_metrics = state_monotonic_improvement_loss(
                state_digit_logits=outputs["state_digit_logits"],
                answer_targets=answer_target,
                margin=args.monotonic_margin,
                weight=args.monotonic_weight,
            )
            loss = loss + mono_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )
        
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        state_loss_sum += metrics["state_loss"].item()
        answer_loss_sum += metrics["answer_loss"].item()
        consistency_loss_sum += metrics["consistency_loss"].item()
        state_acc_sum += metrics["state_accuracy"].item()
        answer_acc_sum += metrics["answer_accuracy"].item()
        n_batches += 1
    
    return {
        "loss": total_loss / n_batches,
        "state_loss": state_loss_sum / n_batches,
        "answer_loss": answer_loss_sum / n_batches,
        "consistency_loss": consistency_loss_sum / n_batches,
        "state_accuracy": state_acc_sum / n_batches,
        "answer_accuracy": answer_acc_sum / n_batches,
    }


@torch.no_grad()
def evaluate(
    model: StateTransitionModel,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    label_token_ids: torch.Tensor,
) -> dict:
    """Evaluate with full metrics."""
    model.eval()
    
    all_metrics = {
        "state_accuracy": [],
        "answer_accuracy": [],
        "family_state_accuracy": {},
        "family_answer_accuracy": {},
        "core_on_answer_accuracy": [],
        "core_off_answer_accuracy": [],
    }
    
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
            all_metrics["state_accuracy"].append(state_acc)
        
        # Answer accuracy
        answer_preds = outputs["answer_logits"].argmax(dim=-1)
        answer_acc = (answer_preds == answer_target).float().mean().item()
        all_metrics["answer_accuracy"].append(answer_acc)
        
        # Per-family accuracy
        for family in set(families):
            fam_mask = [f == family for f in families]
            if any(fam_mask):
                fam_state_acc = (state_preds[fam_mask] == state_targets[fam_mask]).float()[
                    valid_mask[fam_mask]
                ].mean().item() if valid_mask[fam_mask].any() else 0.0
                fam_answer_acc = (answer_preds[fam_mask] == answer_target[fam_mask]).float().mean().item()
                
                if family not in all_metrics["family_state_accuracy"]:
                    all_metrics["family_state_accuracy"][family] = []
                    all_metrics["family_answer_accuracy"][family] = []
                all_metrics["family_state_accuracy"][family].append(fam_state_acc)
                all_metrics["family_answer_accuracy"][family].append(fam_answer_acc)
        
        # Core OFF baseline
        outputs_off = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            force_core_off=True,
        )
        
        if outputs_off["logits"] is not None and label_token_ids is not None:
            # Extract answer from Qwen logits
            qwen_logits = outputs_off["logits"][:, -1, :]  # last token
            core_choice = qwen_logits.float().index_select(
                dim=-1,
                index=label_token_ids.to(device),
            )
            core_off_preds = core_choice.argmax(dim=-1)
            core_off_acc = (core_off_preds == answer_target).float().mean().item()
            all_metrics["core_off_answer_accuracy"].append(core_off_acc)
        
        # Core ON vs OFF comparison
        core_on_acc = (answer_preds == answer_target).float().mean().item()
        all_metrics["core_on_answer_accuracy"].append(core_on_acc)
    
    # Aggregate
    result = {}
    result["state_accuracy"] = (
        sum(all_metrics["state_accuracy"]) / len(all_metrics["state_accuracy"])
        if all_metrics["state_accuracy"] else 0.0
    )
    result["answer_accuracy"] = (
        sum(all_metrics["answer_accuracy"]) / len(all_metrics["answer_accuracy"])
        if all_metrics["answer_accuracy"] else 0.0
    )
    result["core_on_answer_accuracy"] = (
        sum(all_metrics["core_on_answer_accuracy"]) / len(all_metrics["core_on_answer_accuracy"])
        if all_metrics["core_on_answer_accuracy"] else 0.0
    )
    result["core_off_answer_accuracy"] = (
        sum(all_metrics["core_off_answer_accuracy"]) / len(all_metrics["core_off_answer_accuracy"])
        if all_metrics["core_off_answer_accuracy"] else 0.0
    )
    result["core_gain"] = result["core_on_answer_accuracy"] - result["core_off_answer_accuracy"]
    
    # Per-family
    result["family_state_accuracy"] = {}
    result["family_answer_accuracy"] = {}
    for family, accs in all_metrics["family_state_accuracy"].items():
        result["family_state_accuracy"][family] = sum(accs) / len(accs) if accs else 0.0
        result["family_answer_accuracy"][family] = (
            sum(all_metrics["family_answer_accuracy"].get(family, [0.0])) /
            len(all_metrics["family_answer_accuracy"].get(family, [1.0]))
        )
    
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="State-Transition-First Training")
    
    # Model
    parser.add_argument("--qwen-model-id", type=str, default=None,
                       help="Qwen model ID for backbone (default: random init)")
    parser.add_argument("--d-state", type=int, default=256, help="State dimension")
    parser.add_argument("--n-steps", type=int, default=4, help="Number of transition steps")
    parser.add_argument("--freeze-qwen", action="store_true", default=True,
                       help="Freeze Qwen backbone")
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
    
    # Output
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Output: {out_dir}")
    
    # Load Qwen model or use random init
    qwen_model = None
    tokenizer = None
    label_token_ids = None
    
    if args.qwen_model_id:
        print(f"Loading Qwen model: {args.qwen_model_id}")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            qwen_model = AutoModelForCausalLM.from_pretrained(
                args.qwen_model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(args.qwen_model_id)
            # Setup label token IDs
            digit_chars = [str(i) for i in range(10)]
            label_token_ids = [
                tokenizer.encode(d, add_special_tokens=False)[0]
                for d in digit_chars
            ]
        except Exception as e:
            print(f"Failed to load Qwen model: {e}")
            print("Falling back to random initialization...")
            qwen_model = None
            tokenizer = None
    
    if qwen_model is None:
        # Simple random init model for smoke testing
        print("Using random initialization for smoke test")
        class SimpleBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.d_model = args.d_state
                self.embed = torch.nn.Embedding(1000, args.d_state)
                self.layers = torch.nn.ModuleList([
                    torch.nn.Sequential(
                        torch.nn.LayerNorm(args.d_state),
                        torch.nn.Linear(args.d_state, args.d_state * 4),
                        torch.nn.GELU(),
                        torch.nn.Linear(args.d_state * 4, args.d_state),
                    )
                    for _ in range(2)
                ])
                self.config = type('Config', (), {
                    'hidden_size': args.d_state,
                    'vocab_size': 1000,
                })()
            
            def forward(self, input_ids, attention_mask=None, **kwargs):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = x + layer(x)
                class Outputs:
                    def __init__(self, last, logits):
                        self.last_hidden_state = last
                        self.logits = logits
                        self.hidden_states = (last,)
                logits = x  # simplified
                return Outputs(x, logits)
            
            def parameters(self):
                return super(SimpleBackbone, self).parameters()
        
        qwen_model = SimpleBackbone().to(device)
        # For simple backbone, use digit indices directly as labels
        label_token_ids = list(range(10))
        # Simple tokenizer for random init
        class SimpleTokenizer:
            def __call__(self, text, max_length=128, truncation=True, padding="max_length", return_tensors="pt"):
                # Simple encoding: use character ord values mod vocab_size
                tokens = [ord(c) % 1000 for c in text[:max_length]]
                # Pad to max_length
                while len(tokens) < max_length:
                    tokens.append(0)
                import torch
                return type('Tokens', (), {
                    'input_ids': torch.tensor(tokens, dtype=torch.long).unsqueeze(0),
                    'attention_mask': torch.ones(max_length, dtype=torch.long).unsqueeze(0),
                })()
            def encode(self, text, add_special_tokens=False):
                return [ord(c) % 1000 for c in text]
        tokenizer = SimpleTokenizer()
    
    # Build model
    trainable_layers = None
    if args.trainable_qwen_layers:
        trainable_layers = [int(x.strip()) for x in args.trainable_qwen_layers.split(",")]
    
    model = StateTransitionModel(
        qwen_model=qwen_model,
        d_state=args.d_state,
        n_operations=N_OPERATIONS,
        n_steps=args.n_steps,
        freeze_qwen=args.freeze_qwen,
        trainable_layers=trainable_layers,
    ).to(device).to(dtype)
    
    if label_token_ids:
        model.set_label_token_ids(label_token_ids)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    
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
        train_metrics = train(model, train_loader, optimizer, args, device, epoch)
        
        # Step scheduler
        for _ in range(len(train_loader)):
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Evaluate
        eval_metrics = {}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            label_ids = model.label_token_ids
            if label_ids is not None:
                label_ids = label_ids.to(device)
            eval_metrics = evaluate(
                model, eval_loader, args, device, label_ids
            )
            
            if eval_metrics.get("state_accuracy", 0) > best_state_acc:
                best_state_acc = eval_metrics["state_accuracy"]
                torch.save(model.state_dict(), out_dir / "best_state.pt")
            
            if eval_metrics.get("answer_accuracy", 0) > best_answer_acc:
                best_answer_acc = eval_metrics["answer_accuracy"]
                torch.save(model.state_dict(), out_dir / "best_answer.pt")
        
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
                f"eval_answer_acc={eval_metrics['answer_accuracy']:.4f} "
                f"core_gain={eval_metrics.get('core_gain', 0):.4f}"
            )
            # Per-family
            fam_strs = []
            for fam in sorted(eval_metrics.get("family_answer_accuracy", {}).keys()):
                fam_strs.append(
                    f"{fam}:{eval_metrics['family_answer_accuracy'][fam]:.4f}"
                )
            if fam_strs:
                eval_str += f" | families={','.join(fam_strs)}"
        
        print(
            f"Epoch {epoch:3d} | loss={train_metrics['loss']:.4f} "
            f"state_loss={train_metrics['state_loss']:.4f} "
            f"answer_loss={train_metrics['answer_loss']:.4f} "
            f"state_acc={train_metrics['state_accuracy']:.4f} "
            f"answer_acc={train_metrics['answer_accuracy']:.4f}"
            f"{eval_str}"
        )
        
        # Save checkpoint
        torch.save(model.state_dict(), out_dir / "last.pt")
        
        # Save history
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)
        
        # Save final report
        final_report = {
            "config": vars(args),
            "history": history,
            "best_state_accuracy": best_state_acc,
            "best_answer_accuracy": best_answer_acc,
            "final_eval": eval_metrics if eval_metrics else None,
        }
        with open(out_dir / "report.json", "w") as f:
            json.dump(final_report, f, indent=2, default=str)
    
    print(f"\nTraining complete!")
    print(f"Best state accuracy: {best_state_acc:.4f}")
    print(f"Best answer accuracy: {best_answer_acc:.4f}")
    print(f"Final report: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
