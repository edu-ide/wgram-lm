#!/usr/bin/env python3
"""
Minimal probe at d123cdc commit.
Uses the StateTransitionCore + the exact equation_state_binding loss from this commit.
Runs on a small subset of the 72 heldout.
Reports basic loss dynamics and a crude forced_choice signal using the provided choices.
"""

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

from qtrm_mm.state_transition_core import StateTransitionCore, N_OPERATIONS
from qtrm_mm.config import QTRMConfig

# Load the equation_state_binding loss exactly as done in the 627 probe at this commit
ROOT = Path(__file__).resolve().parent
LOSS_FILE = ROOT / "src" / "qtrm_mm" / "losses" / "equation_state_binding.py"
_loss_globals: Dict[str, Any] = {"__name__": "equation_state_binding", "__file__": str(LOSS_FILE)}
with open(LOSS_FILE, "r", encoding="utf-8") as f:
    exec(compile(f.read(), str(LOSS_FILE), "exec"), _loss_globals)
compute_equation_state_binding_loss = _loss_globals["compute_equation_state_binding_loss"]
EquationStateBindingConfig = _loss_globals["EquationStateBindingConfig"]

def load_heldout_subset(path: str, limit: int = 8):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            rec = json.loads(line)
            items.append(rec)
    return items

def main():
    random.seed(42)
    torch.manual_seed(42)

    heldout_path = "data/eval/pure_recursive_reasoning_heldout_72.jsonl"
    items = load_heldout_subset(heldout_path, limit=8)
    print(f"Loaded {len(items)} cases from heldout 72 (d123cdc probe)")

    d_state = 256
    n_steps = 4
    device = torch.device("cpu")

    core = StateTransitionCore(
        cfg=QTRMConfig(d_model=d_state, num_actions=N_OPERATIONS, outer_steps=n_steps),
        d_state=d_state,
        n_operations=N_OPERATIONS,
        n_steps=n_steps,
    ).to(device)
    core.eval()

    binding_cfg = EquationStateBindingConfig(d_state=d_state, n_ops=N_OPERATIONS)

    total_loss = 0.0
    binding_loss_sum = 0.0
    ranks = []

    for i, rec in enumerate(items):
        workspace = torch.randn(1, 4, d_state, device=device) * 0.05
        op_ids = torch.randint(0, N_OPERATIONS, (1, n_steps), device=device)

        with torch.no_grad():
            out = core(workspace, operation_ids=op_ids, n_steps=n_steps)

        state_loss = F.mse_loss(
            out.state_trajectory[:, -1],
            torch.zeros_like(out.state_trajectory[:, -1])
        )

        gold = rec.get("answer_aliases", [""])[0]
        choices = rec.get("choices", [])
        ans_loss = torch.tensor(0.0, device=device)
        gold_idx = -1
        if gold in choices:
            gold_idx = choices.index(gold)
            ans_loss = F.cross_entropy(
                out.answer_logits,
                torch.tensor([gold_idx % 10], device=device)
            )

        b_loss = torch.tensor(0.0, device=device)
        q = rec.get("question", "")
        if any(op in q for op in ["+", "-", "*", "/"]):
            try:
                b_loss, _ = compute_equation_state_binding_loss(
                    out.state_trajectory[:, -1],
                    [q],
                    binding_cfg
                )
            except Exception:
                b_loss = torch.tensor(0.0, device=device)

        loss = state_loss + ans_loss * 0.5 + b_loss * 0.1
        total_loss += loss.item()
        binding_loss_sum += b_loss.item()

        # Crude forced_choice rank using answer_logits as proxy
        if choices and gold in choices and gold_idx >= 0:
            gold_conf = out.answer_logits[0, gold_idx % 10].item()
            better = sum(1 for j, c in enumerate(choices) if c != gold and 
                         out.answer_logits[0, j % 10].item() > gold_conf)
            rank = better + 1
            ranks.append(rank)
            print(f"Case {i}: gold among {len(choices)} choices → crude rank {rank}/{len(choices)}")
        else:
            print(f"Case {i}: gold not cleanly mappable in provided choices")

        print(f"  loss={loss.item():.4f} (state={state_loss.item():.4f}, ans={ans_loss.item():.4f}, bind={b_loss.item():.4f})")

    print("\n=== d123cdc probe summary (first 8 cases) ===")
    print(f"Avg total loss : {total_loss / len(items):.4f}")
    if ranks:
        print(f"Avg crude rank : {sum(ranks)/len(ranks):.2f} / {len(choices)} (lower is better)")
    print("This is a minimal diagnostic using the StateTransitionCore + equation_state_binding loss exactly as they existed at d123cdc.")

if __name__ == "__main__":
    main()
