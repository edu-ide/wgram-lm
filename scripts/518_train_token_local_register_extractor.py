"""Train a token-local monotonic typed-register extractor.

Stage53C fixes the specific Stage53A/B failure mode: free per-slot queries
learned to ask "what belongs in slot t?" but did not copy long ledgers
reliably. This script trains a local token tagger instead:

  frozen Qwen token states -> token roles/digits/ops -> monotonic packing
  -> typed register executor -> K-sample candidate selection

The probe is still local-only and diagnostic. It tests whether exact prompt
reading is the missing piece before changing the full QTRM/GRAM architecture.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from wgram_lm.qwen_backbone_state_transition import build_qwen_state_transition_model


ROLE_OTHER = 0
ROLE_START = 1
ROLE_OP = 2
ROLE_ARG = 3
IGNORE_INDEX = -100
OP_TO_ID = {"add": 0, "mul": 1, "sub": 2, "copy": 3}
ID_TO_OP = {value: key for key, value in OP_TO_ID.items()}
OP_WORD_TO_ID = {
    "add": OP_TO_ID["add"],
    "plus": OP_TO_ID["add"],
    "mul": OP_TO_ID["mul"],
    "multiply": OP_TO_ID["mul"],
    "times": OP_TO_ID["mul"],
    "sub": OP_TO_ID["sub"],
    "subtract": OP_TO_ID["sub"],
    "minus": OP_TO_ID["sub"],
    "copy": OP_TO_ID["copy"],
}


def _load_script(filename: str, module_name: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


train511 = _load_script("511_train_qwen_state_transition_hrmtext.py", "qtrm_stage511")
stage517 = _load_script("517_train_qwen_register_extractor.py", "qtrm_stage517")


class TokenLocalRegisterExtractor(nn.Module):
    """Token role tagger plus sequence heads for start/depth fallback."""

    def __init__(self, *, hidden_size: int, max_steps: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(float(dropout))
        self.role_head = nn.Linear(hidden_size, 4)
        self.digit_head = nn.Linear(hidden_size, 10)
        self.operation_head = nn.Linear(hidden_size, 4)
        self.sequence_norm = nn.LayerNorm(hidden_size)
        self.initial_head = nn.Linear(hidden_size, 10)
        self.depth_head = nn.Linear(hidden_size, self.max_steps + 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.dropout(self.norm(hidden_states.float()))
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = self.sequence_norm(pooled)
        return {
            "role_logits": self.role_head(hidden),
            "digit_logits": self.digit_head(hidden),
            "operation_logits": self.operation_head(hidden),
            "initial_logits": self.initial_head(pooled),
            "depth_logits": self.depth_head(pooled),
        }


def configure_reproducibility(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def init_aim_run(args: argparse.Namespace) -> Optional[Any]:
    if not args.aim_repo:
        return None
    try:
        from aim import Run
    except ImportError as exc:
        print(f"[warn] Aim logging disabled; package is not installed: {exc}", flush=True)
        return None
    run = Run(repo=args.aim_repo, experiment=args.aim_experiment)
    run.name = args.aim_run_name or args.run_name or os.path.basename(os.path.normpath(args.out_dir))
    run["hparams"] = dict(vars(args))
    run["paths"] = {"out_dir": args.out_dir, "tensorboard_logdir": os.path.join(args.out_dir, "logs")}
    return run


def track_aim_scalar(
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


def build_cases(
    *,
    count: int,
    seed: int,
    depths: Sequence[int],
    max_steps: int,
    args: argparse.Namespace,
    surface_mode: str = "canonical",
) -> List[Any]:
    cases = train511.build_generalized_synthetic_cases(
        count=int(count),
        seed=int(seed),
        depths=[int(depth) for depth in depths],
        max_steps=int(max_steps),
        condition_prefix=args.reasoning_condition_prefix,
        family_mix=args.synthetic_family_mix,
        sampling_strategy=args.synthetic_sampling_strategy,
    )
    return apply_surface_mode(cases, mode=surface_mode, seed=seed)


def _clone_case(case: Any, prompt_text: str, family_suffix: str) -> Any:
    return train511.SyntheticCase(
        prompt_text=prompt_text,
        operation_ids=list(case.operation_ids),
        answer_label=int(case.answer_label),
        state_labels=list(case.state_labels),
        family=f"{case.family}:{family_suffix}" if family_suffix else case.family,
        depth=int(case.depth),
        operation_args=list(case.operation_args or [0] * len(case.operation_ids)),
        initial_label=int(case.initial_label),
    )


def _operation_pairs(case: Any) -> List[Tuple[str, int]]:
    operation_args = list(case.operation_args or [0] * len(case.operation_ids))
    pairs: List[Tuple[str, int]] = []
    for op_id, arg in zip(list(case.operation_ids)[: int(case.depth)], operation_args[: int(case.depth)]):
        pairs.append((ID_TO_OP.get(int(op_id), "copy"), int(arg) % 10))
    return pairs


def rewrite_surface(case: Any, *, mode: str) -> Any:
    if mode == "canonical":
        return case
    pairs = _operation_pairs(case)
    ledger = " | ".join(f"{op} {arg}" for op, arg in pairs)
    if mode == "ledger":
        prompt = (
            f"Condition: synth,typed\n"
            f"Task: {case.family} depth {int(case.depth)} modulo 10.\n"
            f"Initial value: {int(case.initial_label)}.\n"
            f"Operation ledger: {ledger}.\n"
            "Return the final digit.\n"
            "Answer:"
        )
        return _clone_case(case, prompt, mode)
    if mode == "prose":
        phrases = []
        for op, arg in pairs:
            if op == "add":
                phrases.append(f"Then add {arg}")
            elif op == "mul":
                phrases.append(f"Then multiply by {arg}")
            elif op == "sub":
                phrases.append(f"Then subtract {arg}")
            else:
                phrases.append(f"Then copy {arg}")
        prompt = (
            f"Condition: synth,prose\n"
            "Modulo-10 reasoning.\n"
            f"Begin at digit {int(case.initial_label)}. "
            + ". ".join(phrases)
            + ".\nReturn the final digit.\nAnswer:"
        )
        return _clone_case(case, prompt, mode)
    if mode == "heldout":
        phrases = []
        for op, arg in pairs:
            if op == "add":
                phrases.append(f"plus {arg}")
            elif op == "mul":
                phrases.append(f"times {arg}")
            elif op == "sub":
                phrases.append(f"minus {arg}")
            else:
                phrases.append(f"copy {arg}")
        prompt = (
            f"Condition: synth,ood\n"
            f"Start digit: {int(case.initial_label)}.\n"
            f"Work list => {'; '.join(phrases)}.\n"
            "All arithmetic wraps modulo 10. Answer:"
        )
        return _clone_case(case, prompt, mode)
    raise ValueError(f"unknown surface mode: {mode}")


def apply_surface_mode(cases: Sequence[Any], *, mode: str, seed: int) -> List[Any]:
    mode = str(mode)
    if mode == "canonical":
        return list(cases)
    rng = random.Random(int(seed) + 5519)
    if mode == "mixed":
        options = ("canonical", "ledger", "prose")
        return [rewrite_surface(case, mode=rng.choice(options)) for case in cases]
    if mode == "mixed_all":
        options = ("canonical", "ledger", "prose", "heldout")
        return [rewrite_surface(case, mode=rng.choice(options)) for case in cases]
    if mode in {"ledger", "prose", "heldout"}:
        return [rewrite_surface(case, mode=mode) for case in cases]
    raise ValueError(f"unknown surface mode: {mode}")


def encode_with_offsets(tokenizer: Any, texts: Sequence[str], max_length: int, device: torch.device) -> Dict[str, Any]:
    encoded = tokenizer(
        list(texts),
        truncation=True,
        max_length=int(max_length),
        padding="max_length",
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "offset_mapping": encoded["offset_mapping"],
    }


@torch.inference_mode()
def qwen_hidden(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model.qwen(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    if hasattr(outputs, "hidden_states") and outputs.hidden_states:
        hidden = outputs.hidden_states[-1]
    elif hasattr(outputs, "last_hidden_state"):
        hidden = outputs.last_hidden_state
    else:
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
    return hidden.detach()


def _overlaps(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> bool:
    a0, a1 = span_a
    b0, b1 = span_b
    return max(a0, b0) < min(a1, b1)


def _append_op_arg_spans(
    spans: List[Tuple[int, int, int, Optional[int], Optional[int]]],
    *,
    segment: str,
    field_start: int,
) -> None:
    pattern = re.compile(
        r"(add|mul|sub|copy|multiply|subtract|plus|times|minus)"
        r"(?:\s+by)?\s*[:= ]\s*(\d)"
    )
    for match in pattern.finditer(segment):
        op_word = match.group(1)
        arg_text = match.group(2)
        spans.append(
            (
                field_start + match.start(1),
                field_start + match.end(1),
                ROLE_OP,
                None,
                OP_WORD_TO_ID[op_word],
            )
        )
        spans.append(
            (
                field_start + match.start(2),
                field_start + match.end(2),
                ROLE_ARG,
                int(arg_text),
                None,
            )
        )


def _case_spans(case: Any) -> List[Tuple[int, int, int, Optional[int], Optional[int]]]:
    """Return (start, end, role, digit_label, op_label) spans."""
    text = str(case.prompt_text)
    spans: List[Tuple[int, int, int, Optional[int], Optional[int]]] = []

    for start_pattern in (
        r"start=(\d)",
        r"Initial value:\s*(\d)",
        r"Begin at digit\s+(\d)",
        r"Start digit:\s*(\d)",
    ):
        start_match = re.search(start_pattern, text)
        if start_match:
            spans.append((start_match.start(1), start_match.end(1), ROLE_START, int(case.initial_label), None))
            break

    if "steps=" in text:
        steps_match = re.search(r"steps=([^\n.]+)", text)
        if steps_match:
            field_start = steps_match.start(1)
            _append_op_arg_spans(spans, segment=steps_match.group(1), field_start=field_start)
    for field_pattern in (
        r"Operation ledger:\s*([^\n.]+)",
        r"Work list\s*=>\s*([^\n.]+)",
    ):
        field_match = re.search(field_pattern, text)
        if field_match:
            _append_op_arg_spans(spans, segment=field_match.group(1), field_start=field_match.start(1))
    if "Then " in text:
        _append_op_arg_spans(spans, segment=text, field_start=0)
    if "digits=" in text:
        digits_match = re.search(r"digits=([^\n.]+)", text)
        if digits_match:
            field_start = digits_match.start(1)
            for match in re.finditer(r"\d", digits_match.group(1)):
                spans.append(
                    (
                        field_start + match.start(0),
                        field_start + match.end(0),
                        ROLE_ARG,
                        int(match.group(0)),
                        None,
                    )
                )
    if "ops=" in text:
        ops_match = re.search(r"ops=([^\n.]+)", text)
        if ops_match:
            field_start = ops_match.start(1)
            for match in re.finditer(r"add|mul|sub|copy", ops_match.group(1)):
                op_text = match.group(0)
                if op_text in OP_TO_ID:
                    spans.append(
                        (
                            field_start + match.start(0),
                            field_start + match.end(0),
                            ROLE_OP,
                            None,
                            OP_TO_ID[op_text],
                        )
                    )
    return spans


def token_targets(
    *,
    cases: Sequence[Any],
    offsets: torch.Tensor,
    attention_mask: torch.Tensor,
    max_steps: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    batch, seq_len = offsets.shape[:2]
    role_labels = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
    digit_labels = torch.full((batch, seq_len), IGNORE_INDEX, dtype=torch.long, device=device)
    op_labels = torch.full((batch, seq_len), IGNORE_INDEX, dtype=torch.long, device=device)
    offsets_list = offsets.detach().cpu().tolist()
    attention_list = attention_mask.detach().cpu().tolist()

    for row, case in enumerate(cases):
        spans = _case_spans(case)
        for token_index, ((tok_start, tok_end), active) in enumerate(zip(offsets_list[row], attention_list[row])):
            if not active or int(tok_end) <= int(tok_start):
                continue
            token_span = (int(tok_start), int(tok_end))
            for span_start, span_end, role, digit, op_id in spans:
                if _overlaps(token_span, (span_start, span_end)):
                    role_labels[row, token_index] = int(role)
                    if digit is not None:
                        digit_labels[row, token_index] = int(digit)
                    if op_id is not None:
                        op_labels[row, token_index] = int(op_id)
                    break

    case_step_targets = stage517.case_targets(cases, int(max_steps), device)
    return {
        **case_step_targets,
        "role_labels": role_labels,
        "digit_token_labels": digit_labels,
        "operation_token_labels": op_labels,
    }


def compute_loss(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    active_tokens = attention_mask.to(torch.bool)
    role_logits = outputs["role_logits"][active_tokens]
    role_labels = targets["role_labels"][active_tokens]
    role_weight = torch.tensor([0.08, 1.0, 1.0, 1.0], device=role_logits.device, dtype=role_logits.dtype)
    role_loss = F.cross_entropy(role_logits, role_labels, weight=role_weight)
    digit_token_mask = targets["digit_token_labels"].ne(IGNORE_INDEX) & active_tokens
    op_token_mask = targets["operation_token_labels"].ne(IGNORE_INDEX) & active_tokens
    digit_loss = F.cross_entropy(outputs["digit_logits"][digit_token_mask], targets["digit_token_labels"][digit_token_mask])
    op_loss = F.cross_entropy(outputs["operation_logits"][op_token_mask], targets["operation_token_labels"][op_token_mask])
    initial_loss = F.cross_entropy(outputs["initial_logits"], targets["initial_labels"])
    depth_loss = F.cross_entropy(outputs["depth_logits"], targets["depths"].clamp(min=0, max=outputs["depth_logits"].size(-1) - 1))
    loss = role_loss + digit_loss + op_loss + 0.5 * initial_loss + 0.5 * depth_loss
    return loss, {
        "loss": float(loss.detach().item()),
        "loss_role": float(role_loss.detach().item()),
        "loss_digit_token": float(digit_loss.detach().item()),
        "loss_operation_token": float(op_loss.detach().item()),
        "loss_initial": float(initial_loss.detach().item()),
        "loss_depth": float(depth_loss.detach().item()),
    }


def _group_predictions(
    *,
    role_pred: Sequence[int],
    value_pred: Sequence[int],
    attention: Sequence[int],
    role: int,
    max_items: int,
) -> List[int]:
    groups: List[List[int]] = []
    current: List[int] = []
    for token_role, token_value, active in zip(role_pred, value_pred, attention):
        if not active:
            if current:
                groups.append(current)
                current = []
            continue
        if int(token_role) == int(role):
            current.append(int(token_value))
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    values: List[int] = []
    for group in groups[: int(max_items)]:
        counts: Dict[int, int] = {}
        for value in group:
            counts[value] = counts.get(value, 0) + 1
        values.append(max(counts.items(), key=lambda item: item[1])[0])
    return values


def pack_registers(outputs: Dict[str, torch.Tensor], attention_mask: torch.Tensor, max_steps: int) -> Dict[str, torch.Tensor]:
    role_pred = outputs["role_logits"].argmax(dim=-1).detach().cpu().tolist()
    digit_pred = outputs["digit_logits"].argmax(dim=-1).detach().cpu().tolist()
    op_pred = outputs["operation_logits"].argmax(dim=-1).detach().cpu().tolist()
    attention = attention_mask.detach().cpu().tolist()
    initial_fallback = outputs["initial_logits"].argmax(dim=-1).detach().cpu().tolist()
    depth_pred = outputs["depth_logits"].argmax(dim=-1).detach().cpu().tolist()

    initial_values: List[int] = []
    operation_rows: List[List[int]] = []
    argument_rows: List[List[int]] = []
    depths: List[int] = []
    for row in range(len(role_pred)):
        start_digits = _group_predictions(
            role_pred=role_pred[row],
            value_pred=digit_pred[row],
            attention=attention[row],
            role=ROLE_START,
            max_items=1,
        )
        op_values = _group_predictions(
            role_pred=role_pred[row],
            value_pred=op_pred[row],
            attention=attention[row],
            role=ROLE_OP,
            max_items=max_steps,
        )
        arg_values = _group_predictions(
            role_pred=role_pred[row],
            value_pred=digit_pred[row],
            attention=attention[row],
            role=ROLE_ARG,
            max_items=max_steps,
        )
        initial_values.append(int(start_digits[0]) if start_digits else int(initial_fallback[row]))
        depth = max(1, min(int(max_steps), len(op_values), len(arg_values)))
        depths.append(depth)
        op_values = (op_values + [OP_TO_ID["copy"]] * max_steps)[:max_steps]
        arg_values = (arg_values + [0] * max_steps)[:max_steps]
        operation_rows.append(op_values)
        argument_rows.append(arg_values)

    device = outputs["role_logits"].device
    return {
        "initial": torch.tensor(initial_values, dtype=torch.long, device=device),
        "operation_ids": torch.tensor(operation_rows, dtype=torch.long, device=device),
        "operation_args": torch.tensor(argument_rows, dtype=torch.long, device=device),
        "depths": torch.tensor(depths, dtype=torch.long, device=device),
    }


def execute_predicted_registers(registers: Dict[str, torch.Tensor], *, forced_depths: Optional[torch.Tensor] = None) -> torch.Tensor:
    return stage517.execute_predicted_registers(
        initial_digits=registers["initial"],
        operation_ids=registers["operation_ids"],
        operation_args=registers["operation_args"],
        depths=forced_depths if forced_depths is not None else registers["depths"],
    )


def field_metrics(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], attention_mask: torch.Tensor) -> Dict[str, float]:
    active_tokens = attention_mask.to(torch.bool)
    role_pred = outputs["role_logits"].argmax(dim=-1)
    digit_pred = outputs["digit_logits"].argmax(dim=-1)
    op_pred = outputs["operation_logits"].argmax(dim=-1)
    digit_mask = targets["digit_token_labels"].ne(IGNORE_INDEX) & active_tokens
    op_mask = targets["operation_token_labels"].ne(IGNORE_INDEX) & active_tokens
    role_acc = role_pred[active_tokens].eq(targets["role_labels"][active_tokens]).float().mean()
    positive_role_mask = targets["role_labels"].ne(ROLE_OTHER) & active_tokens
    positive_role_acc = role_pred[positive_role_mask].eq(targets["role_labels"][positive_role_mask]).float().mean()
    digit_acc = digit_pred[digit_mask].eq(targets["digit_token_labels"][digit_mask]).float().mean()
    op_acc = op_pred[op_mask].eq(targets["operation_token_labels"][op_mask]).float().mean()
    initial_acc = outputs["initial_logits"].argmax(dim=-1).eq(targets["initial_labels"]).float().mean()
    depth_acc = outputs["depth_logits"].argmax(dim=-1).eq(targets["depths"]).float().mean()
    registers = pack_registers(outputs, attention_mask, int(targets["operation_ids"].size(1)))
    register_answer = execute_predicted_registers(registers, forced_depths=targets["depths"])
    pred_depth_answer = execute_predicted_registers(registers)
    return {
        "role_accuracy": float(role_acc.item()),
        "positive_role_accuracy": float(positive_role_acc.item()),
        "digit_token_accuracy": float(digit_acc.item()),
        "operation_token_accuracy": float(op_acc.item()),
        "initial_accuracy": float(initial_acc.item()),
        "depth_accuracy": float(depth_acc.item()),
        "packed_register_answer_accuracy_oracle_depth": float(register_answer.eq(targets["answer_labels"]).float().mean().item()),
        "packed_register_answer_accuracy_predicted_depth": float(pred_depth_answer.eq(targets["answer_labels"]).float().mean().item()),
    }


def shuffled_batches(cases: Sequence[Any], batch_size: int, rng: random.Random) -> Sequence[List[Any]]:
    indices = list(range(len(cases)))
    rng.shuffle(indices)
    return [[cases[index] for index in indices[start : start + int(batch_size)]] for start in range(0, len(indices), int(batch_size))]


def candidate_selection_details(
    candidate_digits: torch.Tensor,
    register_answers: torch.Tensor,
    answer_labels: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Per-row selector telemetry matching stage517.select_by_register."""
    matches_label = candidate_digits.eq(answer_labels.unsqueeze(1))
    oracle_hit = matches_label.any(dim=1)
    register_matches = candidate_digits.eq(register_answers.unsqueeze(1))
    selected_indices = register_matches.float().argmax(dim=1)
    has_register_match = register_matches.any(dim=1)
    selected_digits = candidate_digits.gather(1, selected_indices.unsqueeze(1)).squeeze(1)
    selected_correct = selected_digits.eq(answer_labels) & has_register_match
    first_correct = candidate_digits[:, 0].eq(answer_labels)
    return {
        "oracle_hit": oracle_hit,
        "register_match": has_register_match,
        "selected_index": selected_indices,
        "selected_digit": selected_digits,
        "selected_correct": selected_correct,
        "first_correct": first_correct,
    }


def failure_type_for_record(*, oracle_hit: bool, selected_correct: bool, register_correct: bool, register_match: bool) -> str:
    if selected_correct:
        return "success"
    if not oracle_hit:
        return "oracle_miss_thinker_candidate_absent"
    if not register_correct:
        return "verifier_register_wrong"
    if not register_match:
        return "verifier_answer_not_exposed"
    return "selector_mismatch"


def maybe_dump_vte_failures(
    *,
    args: argparse.Namespace,
    epoch: int,
    depth: int,
    batch_start: int,
    batch_cases: Sequence[Any],
    candidate_digits: torch.Tensor,
    selection: Dict[str, torch.Tensor],
    registers: Dict[str, torch.Tensor],
    oracle_depth_answer: torch.Tensor,
    predicted_depth_answer: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    if not args.dump_failures_jsonl:
        return
    limit = int(args.dump_failures_limit)
    written = int(getattr(args, "_dump_failures_written", 0))
    if limit >= 0 and written >= limit:
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.dump_failures_jsonl)), exist_ok=True)
    topk = max(1, int(args.candidate_topk_per_sample))
    candidates_cpu = candidate_digits.detach().cpu()
    labels_cpu = labels.detach().cpu()
    oracle_answer_cpu = oracle_depth_answer.detach().cpu()
    predicted_answer_cpu = predicted_depth_answer.detach().cpu()
    selected_digit_cpu = selection["selected_digit"].detach().cpu()
    selected_index_cpu = selection["selected_index"].detach().cpu()
    selected_correct_cpu = selection["selected_correct"].detach().cpu()
    oracle_hit_cpu = selection["oracle_hit"].detach().cpu()
    register_match_cpu = selection["register_match"].detach().cpu()
    first_correct_cpu = selection["first_correct"].detach().cpu()
    initial_cpu = registers["initial"].detach().cpu()
    depth_cpu = registers["depths"].detach().cpu()
    ops_cpu = registers["operation_ids"].detach().cpu()
    packed_args_cpu = registers["operation_args"].detach().cpu()

    with open(args.dump_failures_jsonl, "a", encoding="utf-8") as handle:
        for row_index, case in enumerate(batch_cases):
            label = int(labels_cpu[row_index].item())
            register_answer = int(oracle_answer_cpu[row_index].item())
            selected_correct = bool(selected_correct_cpu[row_index].item())
            oracle_hit = bool(oracle_hit_cpu[row_index].item())
            register_correct = register_answer == label
            register_match = bool(register_match_cpu[row_index].item())
            failure_type = failure_type_for_record(
                oracle_hit=oracle_hit,
                selected_correct=selected_correct,
                register_correct=register_correct,
                register_match=register_match,
            )
            if selected_correct and not bool(args.dump_failures_include_successes):
                continue
            if limit >= 0 and written >= limit:
                break
            flat_candidates = [int(value) for value in candidates_cpu[row_index].tolist()]
            per_sample_topk = [
                flat_candidates[start : start + topk]
                for start in range(0, len(flat_candidates), topk)
            ]
            record = {
                "epoch": int(epoch),
                "depth": int(depth),
                "case_index": int(batch_start + row_index),
                "failure_type": failure_type,
                "selected_correct": selected_correct,
                "oracle_hit": oracle_hit,
                "register_correct": register_correct,
                "register_match": register_match,
                "first_candidate_correct": bool(first_correct_cpu[row_index].item()),
                "label": label,
                "selected_digit": int(selected_digit_cpu[row_index].item()),
                "selected_index": int(selected_index_cpu[row_index].item()),
                "oracle_depth_answer": register_answer,
                "predicted_depth_answer": int(predicted_answer_cpu[row_index].item()),
                "candidate_digits": flat_candidates,
                "candidate_topk_per_sample": topk,
                "candidate_digits_by_sample": per_sample_topk,
                "candidate_label_frequency": int(flat_candidates.count(label)),
                "candidate_unique_digits": sorted(set(flat_candidates)),
                "true_initial": int(case.initial_label),
                "true_depth": int(case.depth),
                "true_ops": [ID_TO_OP.get(int(value), "?") for value in list(case.operation_ids)[: int(case.depth)]],
                "true_args": [int(value) for value in list(case.operation_args or [])[: int(case.depth)]],
                "true_state_labels": [int(value) for value in list(case.state_labels)[: int(case.depth)]],
                "packed_initial": int(initial_cpu[row_index].item()),
                "packed_depth": int(depth_cpu[row_index].item()),
                "packed_ops": [ID_TO_OP.get(int(value), "?") for value in ops_cpu[row_index, : int(case.depth)].tolist()],
                "packed_args": [int(value) for value in packed_args_cpu[row_index, : int(case.depth)].tolist()],
                "family": str(case.family),
                "prompt_text": str(case.prompt_text),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    setattr(args, "_dump_failures_written", written)


def train_epoch(
    *,
    wgram_model: Any,
    tokenizer: Any,
    extractor: TokenLocalRegisterExtractor,
    optimizer: torch.optim.Optimizer,
    cases: Sequence[Any],
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    global_step: int,
    writer: SummaryWriter,
    aim_run: Optional[Any],
) -> Tuple[int, Dict[str, float]]:
    extractor.train()
    rng = random.Random(int(args.seed) + int(epoch) * 7919)
    totals: Dict[str, float] = {}
    count = 0
    started = time.time()
    for batch_cases in shuffled_batches(cases, args.batch_size, rng):
        encoded = encode_with_offsets(tokenizer, [case.prompt_text for case in batch_cases], args.max_length, device)
        targets = token_targets(
            cases=batch_cases,
            offsets=encoded["offset_mapping"],
            attention_mask=encoded["attention_mask"],
            max_steps=args.max_steps,
            device=device,
        )
        with torch.no_grad():
            hidden = qwen_hidden(wgram_model, encoded["input_ids"], encoded["attention_mask"])
        outputs = extractor(hidden, encoded["attention_mask"])
        loss, loss_metrics = compute_loss(outputs, targets, encoded["attention_mask"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(extractor.parameters(), float(args.grad_clip))
        optimizer.step()
        metrics = {**loss_metrics, **field_metrics(outputs, targets, encoded["attention_mask"])}
        batch_size = len(batch_cases)
        count += batch_size
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        if global_step % int(args.log_every) == 0:
            for key, value in metrics.items():
                writer.add_scalar(f"Stage53C/Train/{key}", float(value), global_step)
                track_aim_scalar(aim_run, float(value), name=key, step=global_step, epoch=epoch, context={"split": "train"})
        global_step += 1
    averaged = {key: value / max(1, count) for key, value in totals.items()}
    averaged["seconds"] = time.time() - started
    return global_step, averaged


@torch.inference_mode()
def evaluate_depth(
    *,
    wgram_model: Any,
    tokenizer: Any,
    extractor: TokenLocalRegisterExtractor,
    depth: int,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
) -> Dict[str, Any]:
    extractor.eval()
    cases = build_cases(
        count=args.eval_count,
        seed=args.eval_seed + int(depth),
        depths=[int(depth)],
        max_steps=args.max_steps,
        args=args,
        surface_mode=args.eval_surface_mode,
    )
    metric_totals: Dict[str, float] = {}
    metric_count = 0
    oracle_depth_counts: Dict[str, int] = {}
    predicted_depth_counts: Dict[str, int] = {}
    failure_type_counts: Dict[str, int] = {}
    examples: List[Dict[str, Any]] = []
    for start in range(0, len(cases), int(args.batch_size)):
        batch_cases = cases[start : start + int(args.batch_size)]
        encoded = encode_with_offsets(tokenizer, [case.prompt_text for case in batch_cases], args.max_length, device)
        targets = token_targets(
            cases=batch_cases,
            offsets=encoded["offset_mapping"],
            attention_mask=encoded["attention_mask"],
            max_steps=args.max_steps,
            device=device,
        )
        hidden = qwen_hidden(wgram_model, encoded["input_ids"], encoded["attention_mask"])
        outputs = extractor(hidden, encoded["attention_mask"])
        metrics = field_metrics(outputs, targets, encoded["attention_mask"])
        batch_size = len(batch_cases)
        metric_count += batch_size
        for key, value in metrics.items():
            metric_totals[key] = metric_totals.get(key, 0.0) + float(value) * batch_size

        registers = pack_registers(outputs, encoded["attention_mask"], args.max_steps)
        oracle_depth_answer = execute_predicted_registers(registers, forced_depths=targets["depths"])
        predicted_depth_answer = execute_predicted_registers(registers)
        candidate_digits = stage517.sample_candidate_digits(
            wgram_model,
            tokenizer,
            batch_cases,
            samples=args.samples,
            topk_per_sample=args.candidate_topk_per_sample,
            n_steps=args.model_n_steps,
            max_length=args.max_length,
            condition_on_operation_ids=args.condition_on_operation_ids,
            device=device,
        )
        stage517.merge_counts(oracle_depth_counts, stage517.select_by_register(candidate_digits, oracle_depth_answer, targets["answer_labels"]))
        stage517.merge_counts(predicted_depth_counts, stage517.select_by_register(candidate_digits, predicted_depth_answer, targets["answer_labels"]))
        selection = candidate_selection_details(candidate_digits, oracle_depth_answer, targets["answer_labels"])
        for row_index in range(len(batch_cases)):
            label = int(targets["answer_labels"][row_index].item())
            register_answer = int(oracle_depth_answer[row_index].item())
            failure_type = failure_type_for_record(
                oracle_hit=bool(selection["oracle_hit"][row_index].item()),
                selected_correct=bool(selection["selected_correct"][row_index].item()),
                register_correct=register_answer == label,
                register_match=bool(selection["register_match"][row_index].item()),
            )
            failure_type_counts[failure_type] = int(failure_type_counts.get(failure_type, 0)) + 1
        maybe_dump_vte_failures(
            args=args,
            epoch=int(epoch),
            depth=int(depth),
            batch_start=int(start),
            batch_cases=batch_cases,
            candidate_digits=candidate_digits,
            selection=selection,
            registers=registers,
            oracle_depth_answer=oracle_depth_answer,
            predicted_depth_answer=predicted_depth_answer,
            labels=targets["answer_labels"],
        )

        if len(examples) < 3:
            for row_index, case in enumerate(batch_cases):
                examples.append(
                    {
                        "prompt_text": case.prompt_text,
                        "family": case.family,
                        "depth": int(case.depth),
                        "label": int(case.answer_label),
                        "candidate_digits": [int(value) for value in candidate_digits[row_index].detach().cpu().tolist()],
                        "packed_initial": int(registers["initial"][row_index].item()),
                        "packed_depth": int(registers["depths"][row_index].item()),
                        "packed_ops": [ID_TO_OP.get(int(value), "?") for value in registers["operation_ids"][row_index, : int(case.depth)].detach().cpu().tolist()],
                        "packed_args": [int(value) for value in registers["operation_args"][row_index, : int(case.depth)].detach().cpu().tolist()],
                        "oracle_depth_answer": int(oracle_depth_answer[row_index].item()),
                        "predicted_depth_answer": int(predicted_depth_answer[row_index].item()),
                    }
                )
                if len(examples) >= 3:
                    break
    return {
        **{key: value / max(1, metric_count) for key, value in metric_totals.items()},
        "selector_oracle_depth": stage517.finalize_counts(oracle_depth_counts),
        "selector_predicted_depth": stage517.finalize_counts(predicted_depth_counts),
        "failure_type_counts": failure_type_counts,
        "sample_cases": examples,
    }


def evaluate_all(
    *,
    wgram_model: Any,
    tokenizer: Any,
    extractor: TokenLocalRegisterExtractor,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    aim_run: Optional[Any],
) -> Dict[str, Any]:
    depths: Dict[str, Any] = {}
    selected_values: List[float] = []
    pred_selected_values: List[float] = []
    oracle_values: List[float] = []
    register_values: List[float] = []
    failure_type_counts: Dict[str, int] = {}
    for depth in args.eval_depths:
        result = evaluate_depth(
            wgram_model=wgram_model,
            tokenizer=tokenizer,
            extractor=extractor,
            depth=int(depth),
            args=args,
            device=device,
            epoch=int(epoch),
        )
        depths[str(depth)] = result
        selected = float(result["selector_oracle_depth"]["selected_accuracy"])
        pred_selected = float(result["selector_predicted_depth"]["selected_accuracy"])
        oracle = float(result["selector_oracle_depth"]["oracle_accuracy"])
        register_acc = float(result["packed_register_answer_accuracy_oracle_depth"])
        selected_values.append(selected)
        pred_selected_values.append(pred_selected)
        oracle_values.append(oracle)
        register_values.append(register_acc)
        for key, value in result.get("failure_type_counts", {}).items():
            failure_type_counts[str(key)] = int(failure_type_counts.get(str(key), 0)) + int(value)
        context = {"split": "eval", "depth": str(int(depth))}
        for key in (
            "role_accuracy",
            "positive_role_accuracy",
            "digit_token_accuracy",
            "operation_token_accuracy",
            "packed_register_answer_accuracy_oracle_depth",
            "packed_register_answer_accuracy_predicted_depth",
        ):
            writer.add_scalar(f"Stage53C/EvalDepth{int(depth)}/{key}", float(result[key]), epoch)
            track_aim_scalar(aim_run, float(result[key]), name=key, epoch=epoch, context=context)
        writer.add_scalar(f"Stage53C/EvalDepth{int(depth)}/SelectedOracleDepth", selected, epoch)
        writer.add_scalar(f"Stage53C/EvalDepth{int(depth)}/SelectedPredictedDepth", pred_selected, epoch)
        track_aim_scalar(aim_run, selected, name="selected_accuracy_oracle_depth", epoch=epoch, context=context)
        track_aim_scalar(aim_run, pred_selected, name="selected_accuracy_predicted_depth", epoch=epoch, context=context)
        print(
            f"eval epoch={epoch:02d} depth={int(depth):2d} "
            f"sel_true_depth={selected:.4f} sel_pred_depth={pred_selected:.4f} oracle={oracle:.4f} "
            f"reg={register_acc:.4f} role+={float(result['positive_role_accuracy']):.4f} "
            f"digit_tok={float(result['digit_token_accuracy']):.4f} op_tok={float(result['operation_token_accuracy']):.4f}",
            flush=True,
        )
    failure_total = max(1, sum(failure_type_counts.values()))
    summary = {
        "depths": depths,
        "mean_selected_accuracy_oracle_depth": sum(selected_values) / len(selected_values) if selected_values else 0.0,
        "mean_selected_accuracy_predicted_depth": sum(pred_selected_values) / len(pred_selected_values) if pred_selected_values else 0.0,
        "mean_oracle_accuracy": sum(oracle_values) / len(oracle_values) if oracle_values else 0.0,
        "mean_packed_register_answer_accuracy_oracle_depth": sum(register_values) / len(register_values) if register_values else 0.0,
        "failure_type_counts": failure_type_counts,
        "failure_type_rates": {key: float(value) / failure_total for key, value in sorted(failure_type_counts.items())},
    }
    for key, value in summary.items():
        if key != "depths" and isinstance(value, (int, float)):
            writer.add_scalar(f"Stage53C/EvalMean/{key}", float(value), epoch)
            track_aim_scalar(aim_run, float(value), name=key, epoch=epoch, context={"split": "eval"})
    return summary


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--extractor-checkpoint",
        default="",
        help="Optional best_token_local_register_extractor.pt checkpoint to load before training/eval.",
    )
    parser.add_argument(
        "--eval-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load --extractor-checkpoint and run heldout evaluation without training.",
    )
    parser.add_argument("--answer-path", choices=("state_head", "lm_head"), default="lm_head")
    parser.add_argument("--workspace-pooling", choices=("mean", "last", "attention", "sequence", "none"), default="sequence")
    parser.add_argument("--core-impl", choices=("state_transition", "hybrid_state_transition"), default="state_transition")
    parser.add_argument("--core-update", choices=("mlp", "mini_gated_delta"), default="mlp")
    parser.add_argument("--state-update-schedule", choices=("nested", "two_stream"), default="nested")
    parser.add_argument("--recurrent-readout-pooling", choices=("final", "mean", "attention", "sharp_attention", "hybrid_gate"), default="sharp_attention")
    parser.add_argument("--recurrent-readout-temperature", type=float, default=0.25)
    parser.add_argument("--model-n-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument(
        "--candidate-topk-per-sample",
        type=int,
        default=1,
        help="Expose the top-k answer digits from each stochastic trajectory as candidates.",
    )
    parser.add_argument(
        "--dump-failures-jsonl",
        default="",
        help="Optional JSONL path for VTE failure records during eval.",
    )
    parser.add_argument(
        "--dump-failures-limit",
        type=int,
        default=256,
        help="Maximum VTE records to dump. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--dump-failures-include-successes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include successful selector rows in the dump as well as failures.",
    )
    parser.add_argument("--train-count", type=int, default=3072)
    parser.add_argument("--eval-count", type=int, default=128)
    parser.add_argument("--train-depths", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14])
    parser.add_argument("--eval-depths", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14])
    parser.add_argument("--reasoning-condition-prefix", default="synth")
    parser.add_argument("--synthetic-family-mix", default="balanced")
    parser.add_argument("--synthetic-sampling-strategy", default="random")
    parser.add_argument("--train-surface-mode", choices=("canonical", "ledger", "prose", "heldout", "mixed", "mixed_all"), default="canonical")
    parser.add_argument("--eval-surface-mode", choices=("canonical", "ledger", "prose", "heldout", "mixed", "mixed_all"), default="canonical")
    parser.add_argument("--condition-on-operation-ids", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=98)
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
    parser.add_argument("--aim-repo", default=os.environ.get("QTRM_AIM_REPO", ""))
    parser.add_argument("--aim-experiment", default="qwen35_hrmtext_stage53c_token_local_register")
    parser.add_argument("--aim-run-name", default="")
    args = parser.parse_args()

    configure_reproducibility(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wgram_model, tokenizer = build_qwen_state_transition_model(
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
    load_stats = train511.load_flexible_checkpoint(wgram_model, args.checkpoint, device)
    override_stats = stage517.apply_recurrent_overrides(wgram_model, args)
    wgram_model.eval()
    for parameter in wgram_model.parameters():
        parameter.requires_grad_(False)

    extractor = TokenLocalRegisterExtractor(
        hidden_size=int(wgram_model.hidden_size),
        max_steps=int(args.max_steps),
        dropout=float(args.dropout),
    ).to(device)
    extractor_load_stats: Dict[str, Any] = {"path": "", "loaded": False}
    if args.extractor_checkpoint:
        extractor_checkpoint = torch.load(args.extractor_checkpoint, map_location=device)
        state_dict = extractor_checkpoint.get("extractor", extractor_checkpoint)
        extractor.load_state_dict(state_dict, strict=True)
        extractor_load_stats = {
            "path": args.extractor_checkpoint,
            "loaded": True,
            "epoch": extractor_checkpoint.get("epoch") if isinstance(extractor_checkpoint, dict) else None,
            "best_score": extractor_checkpoint.get("best_score") if isinstance(extractor_checkpoint, dict) else None,
        }
    if args.eval_only and not args.extractor_checkpoint:
        raise ValueError("--eval-only requires --extractor-checkpoint")
    if args.candidate_topk_per_sample <= 0:
        raise ValueError("--candidate-topk-per-sample must be positive")
    if args.dump_failures_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_failures_jsonl)), exist_ok=True)
        with open(args.dump_failures_jsonl, "w", encoding="utf-8"):
            pass
        setattr(args, "_dump_failures_written", 0)
    optimizer = None
    if not args.eval_only:
        optimizer = torch.optim.AdamW(extractor.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "logs"))
    aim_run = init_aim_run(args)
    train_cases = build_cases(
        count=args.train_count,
        seed=args.seed,
        depths=args.train_depths,
        max_steps=args.max_steps,
        args=args,
        surface_mode=args.train_surface_mode,
    )
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "load_stats": load_stats,
        "extractor_load_stats": extractor_load_stats,
        "override_stats": override_stats,
        "train_case_preview": [asdict(case) for case in train_cases[:3]],
        "device": str(device),
    }
    save_json(os.path.join(args.out_dir, "metadata.json"), metadata)
    print(
        f"Stage53C local-only token-local register extractor | train_count={len(train_cases)} "
        f"epochs={args.epochs} device={device} out_dir={args.out_dir}",
        flush=True,
    )

    history: List[Dict[str, Any]] = []
    global_step = 0
    best_score = -1.0
    if args.eval_only:
        eval_summary = evaluate_all(
            wgram_model=wgram_model,
            tokenizer=tokenizer,
            extractor=extractor,
            args=args,
            device=device,
            epoch=0,
            writer=writer,
            aim_run=aim_run,
        )
        history.append({"epoch": 0, "eval": eval_summary})
        save_json(os.path.join(args.out_dir, "summary.json"), {"metadata": metadata, "history": history})
        print(
            f"eval-only mean_selected_true_depth={eval_summary['mean_selected_accuracy_oracle_depth']:.4f} "
            f"mean_selected_pred_depth={eval_summary['mean_selected_accuracy_predicted_depth']:.4f} "
            f"mean_oracle={eval_summary['mean_oracle_accuracy']:.4f} "
            f"mean_register={eval_summary['mean_packed_register_answer_accuracy_oracle_depth']:.4f}",
            flush=True,
        )
        writer.flush()
        writer.close()
        if aim_run is not None:
            aim_run.close()
        return
    for epoch in range(1, int(args.epochs) + 1):
        assert optimizer is not None
        global_step, train_metrics = train_epoch(
            wgram_model=wgram_model,
            tokenizer=tokenizer,
            extractor=extractor,
            optimizer=optimizer,
            cases=train_cases,
            args=args,
            device=device,
            epoch=epoch,
            global_step=global_step,
            writer=writer,
            aim_run=aim_run,
        )
        for key, value in train_metrics.items():
            writer.add_scalar(f"Stage53C/TrainEpoch/{key}", float(value), epoch)
            track_aim_scalar(aim_run, float(value), name=key, epoch=epoch, context={"split": "train_epoch"})
        print(
            f"epoch={epoch:02d} train loss={train_metrics['loss']:.4f} "
            f"role+={train_metrics['positive_role_accuracy']:.4f} "
            f"digit_tok={train_metrics['digit_token_accuracy']:.4f} "
            f"op_tok={train_metrics['operation_token_accuracy']:.4f} "
            f"pack_reg={train_metrics['packed_register_answer_accuracy_oracle_depth']:.4f} "
            f"time={train_metrics['seconds']:.1f}s",
            flush=True,
        )
        eval_summary = evaluate_all(
            wgram_model=wgram_model,
            tokenizer=tokenizer,
            extractor=extractor,
            args=args,
            device=device,
            epoch=epoch,
            writer=writer,
            aim_run=aim_run,
        )
        score = float(eval_summary["mean_selected_accuracy_oracle_depth"])
        history.append({"epoch": epoch, "train": train_metrics, "eval": eval_summary})
        save_json(os.path.join(args.out_dir, "summary.json"), {"metadata": metadata, "history": history})
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "extractor": extractor.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "best_score": best_score,
                    "eval": eval_summary,
                },
                os.path.join(args.out_dir, "best_token_local_register_extractor.pt"),
            )
        print(
            f"epoch={epoch:02d} eval mean_selected_true_depth={eval_summary['mean_selected_accuracy_oracle_depth']:.4f} "
            f"mean_selected_pred_depth={eval_summary['mean_selected_accuracy_predicted_depth']:.4f} "
            f"mean_oracle={eval_summary['mean_oracle_accuracy']:.4f} "
            f"mean_register={eval_summary['mean_packed_register_answer_accuracy_oracle_depth']:.4f}",
            flush=True,
        )
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
