"""
HRM-Text-aligned Qwen state-transition training.

This keeps the existing mandatory QTRM recurrent path, but fixes the data
contract around HRM-Text/Data IO style rows:

- reasoning prompts include the operands needed to determine the label;
- healing rows use condition/instruction/response boundaries;
- healing supervision is response-only and prefix-conditioned.

The current state-transition model emits one final latent state per sequence,
so the healing objective predicts the first response token from the prefix.
That avoids target leakage while preserving HRM-Text's prefix/response split.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from qtrm_mm.qwen_backbone_state_transition import build_qwen_state_transition_model


IGNORE_INDEX = -100


def fit_step_sequence(values: torch.Tensor, n_steps: int) -> torch.Tensor:
    if values.size(1) == n_steps:
        return values
    if values.size(1) > n_steps:
        return values[:, :n_steps]
    pad = values[:, -1:].expand(values.size(0), n_steps - values.size(1))
    return torch.cat([values, pad], dim=1)


@dataclass
class SyntheticCase:
    prompt_text: str
    operation_ids: List[int]
    answer_label: int
    state_labels: List[int]
    family: str
    depth: int = 4
    operation_args: Optional[List[int]] = None
    initial_label: int = 0


OP_TO_ID = {"add": 0, "mul": 1, "sub": 2, "copy": 3}
ID_TO_OP = {value: key for key, value in OP_TO_ID.items()}


def _pad_steps(values: List[int], max_steps: int, *, pad_value: int) -> List[int]:
    if len(values) >= max_steps:
        return values[:max_steps]
    return values + [pad_value] * (max_steps - len(values))


def _clone_synthetic_case(case: SyntheticCase, prompt_text: str, family_suffix: str) -> SyntheticCase:
    return SyntheticCase(
        prompt_text=prompt_text,
        operation_ids=list(case.operation_ids),
        answer_label=int(case.answer_label),
        state_labels=list(case.state_labels),
        family=f"{case.family}:{family_suffix}" if family_suffix else case.family,
        depth=int(case.depth),
        operation_args=list(case.operation_args or [0] * len(case.operation_ids)),
        initial_label=int(case.initial_label),
    )


def _operation_pairs(case: SyntheticCase) -> List[Tuple[str, int]]:
    operation_args = list(case.operation_args or [0] * len(case.operation_ids))
    pairs: List[Tuple[str, int]] = []
    for op_id, arg in zip(list(case.operation_ids)[: int(case.depth)], operation_args[: int(case.depth)]):
        pairs.append((ID_TO_OP.get(int(op_id), "copy"), int(arg) % 10))
    return pairs


def rewrite_synthetic_surface(case: SyntheticCase, *, mode: str) -> SyntheticCase:
    """Rewrite only the prompt surface while preserving the typed data contract."""
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
        return _clone_synthetic_case(case, prompt, mode)
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
        return _clone_synthetic_case(case, prompt, mode)
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
        return _clone_synthetic_case(case, prompt, mode)
    raise ValueError(f"unknown synthetic surface mode: {mode}")


def apply_synthetic_surface_mode(cases: List[SyntheticCase], *, mode: str, seed: int) -> List[SyntheticCase]:
    mode = str(mode)
    if mode == "canonical":
        return cases
    rng = random.Random(int(seed) + 5519)
    if mode == "mixed":
        options = ("canonical", "ledger", "prose")
        return [rewrite_synthetic_surface(case, mode=rng.choice(options)) for case in cases]
    if mode == "mixed_all":
        options = ("canonical", "ledger", "prose", "heldout")
        return [rewrite_synthetic_surface(case, mode=rng.choice(options)) for case in cases]
    if mode in {"ledger", "prose", "heldout"}:
        return [rewrite_synthetic_surface(case, mode=mode) for case in cases]
    raise ValueError(f"unknown synthetic surface mode: {mode}")


def _format_chain_prompt(*, depth: int, start: int, op_text: List[str], condition_prefix: str) -> str:
    return (
        f"Condition: {condition_prefix},cot\n"
        f"Reasoning task: chain{depth} modulo 10.\n"
        f"start={start}.\n"
        f"steps={','.join(op_text)}.\n"
        "Return the final digit.\n"
        "Answer:"
    )


def _format_checksum_prompt(*, depth: int, digits: List[int], condition_prefix: str) -> str:
    return (
        f"Condition: {condition_prefix},direct\n"
        f"Reasoning task: checksum{depth} modulo 10.\n"
        f"digits={','.join(str(digit) for digit in digits)}.\n"
        f"ops={','.join(['add'] * depth)}.\n"
        "Return the final digit.\n"
        "Answer:"
    )


def build_synthetic_cases(count: int = 1024, seed: int = 42) -> List[SyntheticCase]:
    rng = random.Random(seed)
    families = ("chain5", "chain5", "checksum4", "select_pair")
    cases: List[SyntheticCase] = []

    for _ in range(count):
        family = rng.choice(families)
        if family == "chain5":
            start, add_a, mul, sub, add_b = [rng.randint(0, 9) for _ in range(5)]
            s1 = (start + add_a) % 10
            s2 = (s1 * mul) % 10
            s3 = (s2 - sub) % 10
            s4 = (s3 + add_b) % 10
            prompt = (
                "Condition: synth,cot\n"
                "Reasoning task: chain5 modulo 10.\n"
                f"start={start} add_a={add_a} mul={mul} sub={sub} add_b={add_b}.\n"
                "ops=add,mul,sub,add.\n"
                "Return the final digit."
            )
            cases.append(SyntheticCase(prompt, [0, 1, 2, 0], s4, [s1, s2, s3, s4], family, operation_args=[add_a, mul, sub, add_b], initial_label=start))
        elif family == "checksum4":
            digits = [rng.randint(0, 9) for _ in range(4)]
            s1 = digits[0]
            s2 = (s1 + digits[1]) % 10
            s3 = (s2 + digits[2]) % 10
            res = (s3 + digits[3]) % 10
            prompt = (
                "Condition: synth,direct\n"
                "Reasoning task: checksum4 modulo 10.\n"
                f"digits={','.join(str(d) for d in digits)}.\n"
                "ops=add,add,add,add.\n"
                "Return the checksum digit."
            )
            cases.append(SyntheticCase(prompt, [0, 0, 0, 0], res, [s1, s2, s3, res], family, operation_args=digits, initial_label=0))
        else:
            a, b = rng.randint(0, 9), rng.randint(0, 9)
            res = (a + b) % 10
            prompt = (
                "Condition: direct\n"
                "Reasoning task: select_pair modulo 10.\n"
                f"a={a} b={b}.\n"
                "ops=read_a,read_b,add,copy.\n"
                "Return the final digit."
            )
            cases.append(
                SyntheticCase(
                    prompt_text=prompt,
                    operation_ids=[0, 0, 0, 0],
                    answer_label=res,
                    state_labels=[a, b, res, res],
                    family=family,
                    operation_args=[a, b, 0, 0],
                    initial_label=0,
                )
            )

    return cases


def build_generalized_synthetic_cases(
    *,
    count: int = 1024,
    seed: int = 42,
    depths: List[int],
    max_steps: int,
    condition_prefix: str = "synth",
    family_mix: str = "chain2_checksum1",
    sampling_strategy: str = "random",
    depth_family_pattern: Optional[List[Tuple[int, str]]] = None,
) -> List[SyntheticCase]:
    """Build train cases with the same schema used by held-out generalization eval.

    The old tiny benchmark mixed prompt templates and depths differently from the
    evaluator. This generator keeps HRM-Text/data_io-style condition tags while
    making the algorithmic surface identical between train and eval.
    """
    if not depths:
        raise ValueError("depths must not be empty")
    if any(depth <= 0 for depth in depths):
        raise ValueError("all depths must be positive")
    if max(depths) > max_steps:
        raise ValueError("all train depths must be <= max_steps")

    rng = random.Random(seed)
    family_mixes = {
        "chain2_checksum1": ("chain", "chain", "checksum"),
        "balanced": ("chain", "checksum"),
        "checksum2_chain1": ("chain", "checksum", "checksum"),
    }
    if family_mix not in family_mixes:
        raise ValueError(f"unknown family_mix: {family_mix}")
    families = family_mixes[family_mix]
    if sampling_strategy not in {"random", "stratified"}:
        raise ValueError(f"unknown sampling_strategy: {sampling_strategy}")
    if depth_family_pattern is not None:
        if not depth_family_pattern:
            raise ValueError("depth_family_pattern must not be empty")
        allowed_families = {"chain", "checksum"}
        for depth, family in depth_family_pattern:
            if depth <= 0:
                raise ValueError("depth_family_pattern depths must be positive")
            if depth > max_steps:
                raise ValueError("depth_family_pattern depths must be <= max_steps")
            if family not in allowed_families:
                raise ValueError(f"unknown depth_family_pattern family: {family}")
    cases: List[SyntheticCase] = []

    for index in range(count):
        if depth_family_pattern is not None:
            if sampling_strategy == "stratified":
                depth, family = depth_family_pattern[index % len(depth_family_pattern)]
            else:
                depth, family = rng.choice(depth_family_pattern)
        elif sampling_strategy == "stratified":
            depth = depths[index % len(depths)]
            family = families[(index // len(depths)) % len(families)]
        else:
            depth = rng.choice(depths)
            family = rng.choice(families)
        if family == "checksum":
            digits = [rng.randint(0, 9) for _ in range(depth)]
            states: List[int] = []
            value = 0
            for digit in digits:
                value = (value + digit) % 10
                states.append(value)
            cases.append(
                SyntheticCase(
                    prompt_text=_format_checksum_prompt(
                        depth=depth,
                        digits=digits,
                        condition_prefix=condition_prefix,
                    ),
                    operation_ids=_pad_steps([OP_TO_ID["add"]] * depth, max_steps, pad_value=OP_TO_ID["copy"]),
                    answer_label=value,
                    state_labels=_pad_steps(states, max_steps, pad_value=value),
                    family=family,
                    depth=depth,
                    operation_args=_pad_steps(digits, max_steps, pad_value=0),
                    initial_label=0,
                )
            )
            continue

        value = rng.randint(0, 9)
        start = value
        ops: List[int] = []
        args: List[int] = []
        op_text: List[str] = []
        states = []
        for _step in range(depth):
            op = rng.choice(("add", "mul", "sub"))
            arg = rng.randint(0, 9)
            if op == "add":
                value = (value + arg) % 10
            elif op == "mul":
                value = (value * arg) % 10
            else:
                value = (value - arg) % 10
            ops.append(OP_TO_ID[op])
            args.append(arg)
            op_text.append(f"{op}:{arg}")
            states.append(value)
        cases.append(
            SyntheticCase(
                prompt_text=_format_chain_prompt(
                    depth=depth,
                    start=start,
                    op_text=op_text,
                    condition_prefix=condition_prefix,
                ),
                operation_ids=_pad_steps(ops, max_steps, pad_value=OP_TO_ID["copy"]),
                answer_label=value,
                state_labels=_pad_steps(states, max_steps, pad_value=value),
                family=family,
                depth=depth,
                operation_args=_pad_steps(args, max_steps, pad_value=0),
                initial_label=start,
            )
        )

    return cases


def _encode(tokenizer: Any, text: str, max_length: int) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].squeeze(0),
        "attention_mask": encoded["attention_mask"].squeeze(0),
    }


def _unpadded_ids(tokenizer: Any, text: str, max_length: int) -> torch.Tensor:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return encoded["input_ids"].squeeze(0).to(torch.long)


class SyntheticDataset(Dataset):
    def __init__(
        self,
        tokenizer: Any,
        count: int = 4096,
        seed: int = 42,
        max_length: int = 128,
        *,
        schema: str = "legacy",
        depths: Optional[List[int]] = None,
        max_operation_steps: int = 4,
        condition_prefix: str = "synth",
        family_mix: str = "chain2_checksum1",
        sampling_strategy: str = "random",
        depth_family_pattern: Optional[List[Tuple[int, str]]] = None,
        surface_mode: str = "canonical",
    ):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        if schema == "legacy":
            self.cases = build_synthetic_cases(count=count, seed=seed)
        elif schema == "generalized":
            self.cases = build_generalized_synthetic_cases(
                count=count,
                seed=seed,
                depths=depths or [4],
                max_steps=max_operation_steps,
                condition_prefix=condition_prefix,
                family_mix=family_mix,
                sampling_strategy=sampling_strategy,
                depth_family_pattern=depth_family_pattern,
            )
            self.cases = apply_synthetic_surface_mode(self.cases, mode=surface_mode, seed=seed)
        else:
            raise ValueError(f"unknown synthetic schema: {schema}")
        self.data: List[Dict[str, Any]] = []

        for case in self.cases:
            encoded = _encode(tokenizer, case.prompt_text, self.max_length)
            self.data.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "operation_ids": torch.tensor(case.operation_ids, dtype=torch.long),
                    "operation_arg_ids": torch.tensor(
                        case.operation_args or [0] * len(case.operation_ids),
                        dtype=torch.long,
                    ),
                    "initial_labels": torch.tensor(case.initial_label, dtype=torch.long),
                    "reasoning_labels": torch.tensor(case.answer_label, dtype=torch.long),
                    "state_labels": torch.tensor(case.state_labels, dtype=torch.long),
                    "family": case.family,
                    "depth": case.depth,
                    "prompt_text": case.prompt_text,
                    "is_healing": False,
                }
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class HRMTextHealingDataset(Dataset):
    def __init__(
        self,
        tokenizer: Any,
        *,
        rows: Optional[List[Dict[str, str]]] = None,
        count: int = 1024,
        seed: int = 42,
        max_length: int = 128,
        target_tokens: int = 1,
    ):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.target_tokens = max(1, int(target_tokens))
        self.rows = rows if rows is not None else self._load_default_rows(count=count, seed=seed)
        if not self.rows:
            self.rows = [
                {
                    "condition": "direct",
                    "instruction": "Say hello.",
                    "response": "Hello.",
                }
            ] * count

    def _load_default_rows(self, *, count: int, seed: int) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        if count <= 0:
            return rows
        if load_dataset is not None:
            try:
                ds = load_dataset("databricks/databricks-dolly-15k", split="train")
                for item in ds.shuffle(seed=seed).select(range(min(count, len(ds)))):
                    rows.append(
                        {
                            "condition": "synth,cot",
                            "instruction": str(item["instruction"]),
                            "response": str(item["response"]),
                        }
                    )
            except Exception as exc:
                print(f"[warn] failed to load Dolly healing rows: {exc}")
        return rows

    @staticmethod
    def format_prefix(row: Dict[str, str]) -> str:
        condition = row.get("condition", "direct")
        instruction = row.get("instruction", "")
        return f"Condition: {condition}\nInstruction: {instruction}\nResponse:"

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        condition = row.get("condition", "direct")
        prefix_text = self.format_prefix(row)
        response_text = str(row.get("response", ""))

        prefix_ids = _unpadded_ids(self.tokenizer, prefix_text, self.max_length)
        response_room = max(1, self.max_length - min(prefix_ids.numel(), self.max_length - 1))
        response_ids = _unpadded_ids(self.tokenizer, response_text, response_room)
        if response_ids.numel() == 0:
            eos = getattr(self.tokenizer, "eos_token_id", None)
            response_ids = torch.tensor([int(eos) if eos is not None else 0], dtype=torch.long)

        prefix_len = min(prefix_ids.numel(), self.max_length - 1)
        response_ids = response_ids[: self.max_length - prefix_len]
        full_ids = torch.cat([prefix_ids[:prefix_len], response_ids], dim=0)

        input_ids = torch.full((self.max_length,), int(getattr(self.tokenizer, "pad_token_id", 0) or 0), dtype=torch.long)
        attention_mask = torch.zeros((self.max_length,), dtype=torch.long)
        labels = torch.full((self.max_length,), IGNORE_INDEX, dtype=torch.long)
        token_type_ids = torch.zeros((self.max_length,), dtype=torch.long)

        length = min(full_ids.numel(), self.max_length)
        input_ids[:length] = full_ids[:length]
        attention_mask[:length] = 1
        labels[prefix_len:length] = full_ids[prefix_len:length]
        token_type_ids[:prefix_len] = 1

        prefix_only = _encode(self.tokenizer, prefix_text, self.max_length)
        target_ids = labels[prefix_len:].masked_select(labels[prefix_len:] != IGNORE_INDEX)[: self.target_tokens]
        padded_target_ids = torch.full((self.target_tokens,), IGNORE_INDEX, dtype=torch.long)
        if target_ids.numel() > 0:
            padded_target_ids[: target_ids.numel()] = target_ids

        return {
            "input_ids": prefix_only["input_ids"],
            "attention_mask": prefix_only["attention_mask"],
            "labels": labels,
            "token_type_ids": token_type_ids,
            "prefix_len": torch.tensor(prefix_len, dtype=torch.long),
            "response_start": torch.tensor(prefix_len, dtype=torch.long),
            "healing_target_ids": padded_target_ids,
            "condition": condition,
            "is_healing": True,
        }


def _normalize_hrm_text_row(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    instruction = row.get("instruction")
    response = row.get("response")
    if instruction is None or response is None:
        return None
    instruction_text = str(instruction).strip()
    response_text = str(response).strip()
    if not instruction_text or not response_text:
        return None
    return {
        "condition": str(row.get("condition", "direct")).strip() or "direct",
        "instruction": instruction_text,
        "response": response_text,
    }


def load_hrm_text_rows_from_path(
    dataset_path: str,
    *,
    count: int,
    seed: int,
    include_globs: Optional[List[str]] = None,
    rows_per_file_cap: int = 2048,
) -> List[Dict[str, str]]:
    """Load a bounded sample from HRM-Text/data_io cleaned files.

    The cleaned repo is large, so this intentionally samples a bounded number of
    rows per source file instead of materializing the whole dataset.
    """
    if count <= 0:
        return []
    root = os.path.abspath(dataset_path)
    patterns = include_globs or [
        "data/*.jsonl",
        "data/Platypus/*.jsonl",
        "data_clustered/**/*.parquet",
    ]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(root, pattern), recursive=True))
    files = sorted({path for path in files if os.path.isfile(path)})
    if not files:
        raise FileNotFoundError(f"no HRM-Text cleaned files found under {root}")

    rng = random.Random(seed)
    rng.shuffle(files)
    per_file = max(1, min(rows_per_file_cap, (count + len(files) - 1) // len(files)))
    rows: List[Dict[str, str]] = []

    for path in files:
        if len(rows) >= count:
            break
        suffix = os.path.splitext(path)[1].lower()
        loaded = 0
        if suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if loaded >= per_file or len(rows) >= count:
                        break
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    normalized = _normalize_hrm_text_row(row)
                    if normalized is None:
                        continue
                    rows.append(normalized)
                    loaded += 1
            continue

        if suffix == ".parquet" and load_dataset is not None:
            try:
                stream = load_dataset("parquet", data_files=path, split="train", streaming=True)
                for row in stream:
                    if loaded >= per_file or len(rows) >= count:
                        break
                    normalized = _normalize_hrm_text_row(row)
                    if normalized is None:
                        continue
                    rows.append(normalized)
                    loaded += 1
            except Exception as exc:
                print(f"[warn] failed to stream HRM-Text parquet {path}: {exc}", flush=True)

    if not rows:
        raise ValueError(f"no usable HRM-Text rows loaded from {root}")
    if len(rows) > count:
        rows = rows[:count]
    return rows


def parse_depth_family_pattern(values: Optional[List[str]]) -> Optional[List[Tuple[int, str]]]:
    if not values:
        return None
    pattern: List[Tuple[int, str]] = []
    for value in values:
        if ":" not in value:
            raise ValueError("--synthetic-depth-family-pattern entries must look like family:depth")
        family, depth_text = value.split(":", 1)
        family = family.strip()
        try:
            depth = int(depth_text)
        except ValueError as exc:
            raise ValueError(f"invalid depth in --synthetic-depth-family-pattern entry: {value}") from exc
        pattern.append((depth, family))
    return pattern


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key in ("input_ids", "attention_mask", "is_healing"):
        result[key] = torch.stack(
            [torch.tensor(item[key]) if isinstance(item[key], bool) else item[key] for item in batch]
        )

    reasoning_indices = [i for i, item in enumerate(batch) if not item["is_healing"]]
    if reasoning_indices:
        result["r_indices"] = torch.tensor(reasoning_indices, dtype=torch.long)
        result["operation_ids"] = torch.stack([batch[i]["operation_ids"] for i in reasoning_indices])
        result["operation_arg_ids"] = torch.stack([batch[i]["operation_arg_ids"] for i in reasoning_indices])
        result["initial_labels"] = torch.stack([batch[i]["initial_labels"] for i in reasoning_indices])
        result["reasoning_labels"] = torch.stack([batch[i]["reasoning_labels"] for i in reasoning_indices])
        result["state_labels"] = torch.stack([batch[i]["state_labels"] for i in reasoning_indices])
        result["depths"] = torch.tensor([int(batch[i].get("depth", 0)) for i in reasoning_indices], dtype=torch.long)

    healing_indices = [i for i, item in enumerate(batch) if item["is_healing"]]
    if healing_indices:
        result["h_indices"] = torch.tensor(healing_indices, dtype=torch.long)
        result["labels"] = torch.stack([batch[i]["labels"] for i in healing_indices])
        result["token_type_ids"] = torch.stack([batch[i]["token_type_ids"] for i in healing_indices])
        result["prefix_len"] = torch.stack([batch[i]["prefix_len"] for i in healing_indices])
        result["healing_target_ids"] = torch.stack([batch[i]["healing_target_ids"] for i in healing_indices])
        result["conditions"] = [str(batch[i].get("condition", "unknown")) for i in healing_indices]

    return result


def compute_step_answer_loss(
    model: Any,
    state_trajectory: torch.Tensor,
    state_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the final answer readout to every recurrent transition state."""
    step_states = state_trajectory[:, 1:, :]
    if getattr(model, "answer_path", "state_head") == "lm_head":
        logits, _ = model._lm_head_logits_from_state(step_states)
    else:
        normalized = model.core_out_norm(step_states)
        logits = model.answer_head(normalized)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), state_labels.reshape(-1))
    return loss, logits


def _lm_target_embedding_alignment_loss(
    model: Any,
    states: torch.Tensor,
    target_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Align thought hidden states to Qwen LM-head token directions.

    This is the "thinker speaks the speaker's language" loss. It does not add
    a task-specific executor; it keeps the universal path and only teaches the
    recurrent thought state to land in the same hidden geometry used by Qwen's
    LM head.
    """
    lm_head = getattr(model.qwen, "lm_head", None)
    if lm_head is None:
        return states.new_tensor(0.0)
    valid = target_token_ids.ne(IGNORE_INDEX)
    if not bool(valid.any().item()):
        return states.new_tensor(0.0)
    hidden = model._lm_hidden_from_state(states).float()
    safe_targets = target_token_ids.clamp_min(0).to(device=states.device, dtype=torch.long)
    target_embed = lm_head.weight.index_select(0, safe_targets.reshape(-1))
    target_embed = target_embed.reshape(*safe_targets.shape, -1).to(hidden.device).float()
    hidden = F.normalize(hidden[valid], dim=-1)
    target_embed = F.normalize(target_embed[valid], dim=-1)
    return 1.0 - (hidden * target_embed).sum(dim=-1).mean()


def compute_lm_token_loss_from_states(
    model: Any,
    states: torch.Tensor,
    target_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Token CE and hidden-geometry alignment through the normal LM-head path."""
    _, vocab_logits = model._lm_head_logits_from_state(states)
    loss = F.cross_entropy(
        vocab_logits.reshape(-1, vocab_logits.size(-1)),
        target_token_ids.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    alignment_loss = _lm_target_embedding_alignment_loss(model, states, target_token_ids)
    return loss, alignment_loss, vocab_logits


def compute_operation_supervision_loss(
    operation_logits: torch.Tensor,
    operation_ids: torch.Tensor,
) -> torch.Tensor:
    """Supervise the latent trajectory's operation process label at each step."""
    if operation_logits.ndim != 3:
        raise ValueError("operation_logits must have shape (batch, steps, n_ops)")
    if operation_ids.ndim != 2:
        raise ValueError("operation_ids must have shape (batch, steps)")
    if operation_logits.size(0) != operation_ids.size(0) or operation_logits.size(1) != operation_ids.size(1):
        raise ValueError("operation_logits and operation_ids batch/step dimensions must match")
    return F.cross_entropy(
        operation_logits.reshape(-1, operation_logits.size(-1)),
        operation_ids.reshape(-1),
    )


def compute_depth_consistency_loss(
    *,
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL loss that makes a shallower recurrent path match a deeper path."""
    temp = max(float(temperature), 1e-6)
    teacher_probs = F.softmax((teacher_logits / temp).detach(), dim=-1)
    student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp * temp)


def compute_latent_shortcut_consistency_loss(
    *,
    short_state_trajectory: torch.Tensor,
    long_state_trajectory: torch.Tensor,
    min_step: int = 1,
) -> torch.Tensor:
    """Align short-depth recurrent states to long-depth states using Elastic-LSCR.
    
    This maps states at the same normalized time tau = k / T_short to the nearest matching
    normalized time in the long trajectory: t_long = round(tau * T_long).
    """
    if short_state_trajectory.ndim != 3 or long_state_trajectory.ndim != 3:
        raise ValueError("state trajectories must have shape (batch, steps, dim)")
    
    T_short = short_state_trajectory.size(1) - 1
    T_long = long_state_trajectory.size(1) - 1
    
    if T_short < 1 or T_long < 1:
        return torch.tensor(0.0, device=short_state_trajectory.device)
        
    loss_sum = 0.0
    count = 0
    
    for k in range(1, T_short + 1):
        tau = k / T_short
        long_idx = int(round(tau * T_long))
        # Ensure index boundary safety
        long_idx = max(0, min(long_idx, T_long))
        
        # Extract states
        short_state = F.normalize(short_state_trajectory[:, k, :].float(), dim=-1)
        long_state = F.normalize(long_state_trajectory[:, long_idx, :].detach().float(), dim=-1)
        
        # Cosine similarity loss: 1.0 - CosSim
        step_loss = 1.0 - (short_state * long_state).sum(dim=-1).mean()
        loss_sum = loss_sum + step_loss
        count += 1
        
    if count == 0:
        return torch.tensor(0.0, device=short_state_trajectory.device)
        
    return loss_sum / count


def compute_final_readout_logits(model: Any, state_trajectory: torch.Tensor) -> torch.Tensor:
    final_state = state_trajectory[:, -1, :]
    if getattr(model, "answer_path", "state_head") == "lm_head":
        logits, _ = model._lm_head_logits_from_state(final_state)
        return logits
    normalized = model.core_out_norm(final_state)
    return model.answer_head(normalized)


def compute_lattice_candidate_loss(
    candidate_logits: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    min_step: int,
    positive_weight: float,
    negative_weight: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """LDT-style alive-candidate supervision.

    If target_labels has shape (B,), every recurrent step is supervised against
    the final answer. If it has shape (B, T), candidate logits at recurrent
    state step s are supervised against target_labels[:, s - 1]. The latter is
    the sound-transition target for stepwise synthetic state labels.
    """
    if candidate_logits.ndim != 3:
        raise ValueError("candidate_logits must have shape (batch, steps, classes)")
    if target_labels.ndim not in {1, 2}:
        raise ValueError("target_labels must have shape (batch,) or (batch, target_steps)")
    start = max(1, int(min_step))
    if start >= int(candidate_logits.size(1)):
        start = int(candidate_logits.size(1)) - 1
    logits = candidate_logits[:, start:, :]
    if target_labels.ndim == 1:
        step_targets = target_labels.to(torch.long).unsqueeze(1).expand(-1, logits.size(1))
    else:
        target_start = start - 1
        if target_start >= int(target_labels.size(1)):
            target_start = int(target_labels.size(1)) - 1
        step_targets = target_labels[:, target_start : target_start + logits.size(1)].to(torch.long)
        if step_targets.size(1) < logits.size(1):
            pad = step_targets[:, -1:].expand(-1, logits.size(1) - step_targets.size(1))
            step_targets = torch.cat([step_targets, pad], dim=1)
    targets = F.one_hot(step_targets, num_classes=logits.size(-1)).to(logits.dtype)
    raw_loss = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    weights = targets.float() * float(positive_weight) + (1.0 - targets.float()) * float(negative_weight)
    loss = (raw_loss * weights).mean()

    with torch.no_grad():
        probs = torch.sigmoid(logits.float())
        true_probs = probs.gather(-1, step_targets.unsqueeze(-1)).squeeze(-1)
        false_mask = 1.0 - targets.float()
        false_prob = (probs * false_mask).sum() / false_mask.sum().clamp_min(1.0)
        metrics = {
            "true_alive_prob": float(true_probs.mean().item()),
            "false_alive_prob": float(false_prob.item()),
        }
    return loss, metrics


def compute_trajectory_anchor_loss(
    *,
    state_trajectory: torch.Tensor,
    teacher_state: torch.Tensor,
    min_step: int = 1,
) -> torch.Tensor:
    """Align recurrent states to a detached teacher workspace direction.

    This is a lightweight hidden-state distillation gate: the answer still flows
    through the recurrent state path, but each selected recurrent state is kept
    close to the frozen-Qwen workspace direction instead of drifting into a
    train-set-only latent code.
    """
    if state_trajectory.ndim != 3:
        raise ValueError("state_trajectory must have shape (batch, steps, dim)")
    if teacher_state.ndim != 2:
        raise ValueError("teacher_state must have shape (batch, dim)")
    start = max(1, int(min_step))
    if start >= int(state_trajectory.size(1)):
        start = int(state_trajectory.size(1)) - 1
    states = state_trajectory[:, start:, :].float()
    teacher = teacher_state.detach().float().unsqueeze(1).expand_as(states)
    state_direction = F.normalize(states, dim=-1)
    teacher_direction = F.normalize(teacher, dim=-1)
    return 1.0 - (state_direction * teacher_direction).sum(dim=-1).mean()


def encode_text_batch(tokenizer: Any, texts: List[str], max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }


def lattice_threshold_key(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def lattice_threshold_tag(value: float) -> str:
    return lattice_threshold_key(value).replace(".", "p")


def trajectory_cosine_similarity(trajectory: torch.Tensor, samples: int) -> float:
    if samples <= 1 or trajectory.size(0) % samples != 0:
        return 1.0
    batch = trajectory.size(0) // samples
    final_states = trajectory[:, -1, :].reshape(batch, samples, -1)
    final_states = F.normalize(final_states.float(), dim=-1)
    sims = torch.matmul(final_states, final_states.transpose(1, 2))
    mask = ~torch.eye(samples, dtype=torch.bool, device=trajectory.device).unsqueeze(0)
    return float(sims[mask.expand_as(sims)].mean().item())


def select_stochastic_logits(
    out: Dict[str, torch.Tensor],
    *,
    batch_size: int,
    samples: int,
    mode: str,
) -> torch.Tensor:
    logits = out["answer_logits"].reshape(batch_size, samples, -1)
    if mode == "mean":
        return logits.mean(dim=1)
    if mode == "vote":
        preds = logits.argmax(dim=-1)
        selected = []
        for row in preds:
            selected.append(torch.bincount(row, minlength=logits.size(-1)).argmax())
        return F.one_hot(torch.stack(selected), num_classes=logits.size(-1)).to(logits.dtype)
    if mode == "confidence":
        probs = torch.softmax(logits.float(), dim=-1)
        selected = probs.max(dim=-1).values.argmax(dim=-1)
        return logits[torch.arange(batch_size, device=logits.device), selected]
    if mode == "lprm":
        rewards = out["qtrm_trajectory_reward_logits"].reshape(batch_size, samples)
        selected = rewards.argmax(dim=-1)
        return logits[torch.arange(batch_size, device=logits.device), selected]
    raise ValueError(f"unknown stochastic selection mode: {mode}")


def trajectory_reward_logits_from_output(
    model: Any,
    out: Dict[str, torch.Tensor],
    *,
    detach_state: bool = False,
) -> torch.Tensor:
    """Return LPRM reward logits, optionally training only the reward head.

    Stage45 uses this to teach the judge without immediately rewriting the
    recurrent thinker. When detach_state is enabled, gradients flow into
    trajectory_reward_head but not into the sampled trajectory.
    """
    if not detach_state:
        return out["qtrm_trajectory_reward_logits"]
    if hasattr(model, "compute_trajectory_reward_logits"):
        readout_state = out.get("qtrm_readout_state")
        state_trajectory = out.get("qtrm_core_step_states")
        answer_logits = out.get("answer_logits")
        if readout_state is not None and state_trajectory is not None and answer_logits is not None:
            return model.compute_trajectory_reward_logits(
                state_trajectory=state_trajectory,
                readout_state=readout_state,
                answer_logits=answer_logits,
                detach_state=True,
            )
    readout_state = out.get("qtrm_readout_state")
    if readout_state is None:
        return out["qtrm_trajectory_reward_logits"]
    state = readout_state.detach()
    normalized = model.core_out_norm(state)
    return model.trajectory_reward_head(normalized).squeeze(-1)


def compute_multi_trajectory_lprm_loss(
    model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    operation_ids: Optional[torch.Tensor],
    operation_arg_ids: Optional[torch.Tensor],
    initial_labels: torch.Tensor,
    answer_labels: torch.Tensor,
    n_steps: int,
    args: Any,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Train the GRAM/LPRM judge on K sampled thoughts for the same question.

    The inference problem is a within-question selection problem: given K
    stochastic trajectories, choose the one whose answer is most reliable.
    A single-trajectory BCE can reduce loss while still failing to rank the K
    candidates, so this objective combines BCE, listwise selection, and optional
    pairwise ranking over the same K samples used at evaluation time.
    """
    samples = int(args.gram_lprm_train_samples)
    if samples <= 1:
        raise ValueError("compute_multi_trajectory_lprm_loss requires samples > 1")
    batch = int(input_ids.size(0))

    repeated_out = model(
        input_ids=input_ids.repeat_interleave(samples, dim=0),
        attention_mask=attention_mask.repeat_interleave(samples, dim=0),
        operation_ids=(
            operation_ids.repeat_interleave(samples, dim=0)
            if operation_ids is not None
            else None
        ),
        operation_arg_ids=(
            operation_arg_ids.repeat_interleave(samples, dim=0)
            if operation_arg_ids is not None
            else None
        ),
        initial_labels=initial_labels.repeat_interleave(samples, dim=0),
        n_steps=n_steps,
        posterior_labels=None,
    )
    answer_logits = repeated_out["answer_logits"].reshape(batch, samples, -1)
    reward_logits = trajectory_reward_logits_from_output(
        model,
        repeated_out,
        detach_state=bool(args.gram_lprm_detach_state),
    ).reshape(batch, samples)

    with torch.no_grad():
        labels = answer_labels.to(torch.long)
        preds = answer_logits.detach().argmax(dim=-1)
        correct = preds.eq(labels.unsqueeze(1))
        true_probs = torch.softmax(answer_logits.detach().float(), dim=-1)
        true_probs = true_probs.gather(
            2,
            labels.view(batch, 1, 1).expand(-1, samples, 1),
        ).squeeze(-1)
        if args.gram_lprm_target == "correct":
            target_scores = correct.to(reward_logits.dtype)
        else:
            target_scores = true_probs.to(reward_logits.dtype)
        best_indices = target_scores.argmax(dim=1)
        selected_indices = reward_logits.detach().argmax(dim=1)
        selected_correct = correct.gather(1, selected_indices.unsqueeze(1)).squeeze(1)
        oracle_correct = correct.any(dim=1)
        target_spread = target_scores.max(dim=1).values - target_scores.min(dim=1).values

    bce_loss = F.binary_cross_entropy_with_logits(reward_logits, target_scores)
    temp = max(float(args.gram_lprm_listwise_temperature), 1e-6)
    valid_listwise = target_spread > float(args.gram_lprm_pairwise_margin)
    if bool(valid_listwise.any().item()):
        listwise_loss = F.cross_entropy(
            reward_logits[valid_listwise].float() / temp,
            best_indices[valid_listwise],
        )
    else:
        listwise_loss = reward_logits.new_zeros(())

    pairwise_loss = reward_logits.new_zeros(())
    pairwise_count = 0.0
    if float(args.gram_lprm_pairwise_weight) > 0:
        target_delta = target_scores.unsqueeze(2) - target_scores.unsqueeze(1)
        reward_delta = reward_logits.unsqueeze(2) - reward_logits.unsqueeze(1)
        margin = float(args.gram_lprm_pairwise_margin)
        pair_mask = target_delta > margin
        if bool(pair_mask.any().item()):
            pair_losses = F.softplus(-reward_delta.float() / temp)
            pair_weights = target_delta.float().clamp_min(0.0)
            pairwise_loss = (pair_losses[pair_mask] * pair_weights[pair_mask]).mean()
            pairwise_count = float(pair_mask.float().sum().item())

    loss = (
        bce_loss * float(args.gram_lprm_bce_weight)
        + listwise_loss * float(args.gram_lprm_listwise_weight)
        + pairwise_loss * float(args.gram_lprm_pairwise_weight)
    )
    metrics = {
        "selected_accuracy": float(selected_correct.float().mean().item()),
        "oracle_accuracy": float(oracle_correct.float().mean().item()),
        "mean_correct_samples": float(correct.float().mean().item()),
        "target_spread": float(target_spread.float().mean().item()),
        "pairwise_count": pairwise_count,
    }
    return loss, metrics


def compute_multi_trajectory_oracle_ce_loss(
    model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    operation_ids: Optional[torch.Tensor],
    operation_arg_ids: Optional[torch.Tensor],
    initial_labels: torch.Tensor,
    answer_labels: torch.Tensor,
    n_steps: int,
    args: Any,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Backprop through the best sampled trajectory for each question.

    LPRM improves the checker. This loss targets the generator: sample K prior
    trajectories, choose the strongest one under the supervised answer label,
    then train that selected answer path directly.
    """
    samples = int(args.gram_oracle_ce_train_samples)
    if samples <= 0:
        raise ValueError("compute_multi_trajectory_oracle_ce_loss requires samples > 0")
    batch = int(input_ids.size(0))

    repeated_out = model(
        input_ids=input_ids.repeat_interleave(samples, dim=0),
        attention_mask=attention_mask.repeat_interleave(samples, dim=0),
        operation_ids=(
            operation_ids.repeat_interleave(samples, dim=0)
            if operation_ids is not None
            else None
        ),
        operation_arg_ids=(
            operation_arg_ids.repeat_interleave(samples, dim=0)
            if operation_arg_ids is not None
            else None
        ),
        initial_labels=initial_labels.repeat_interleave(samples, dim=0),
        n_steps=n_steps,
        posterior_labels=None,
    )
    answer_logits = repeated_out["answer_logits"].reshape(batch, samples, -1)
    labels = answer_labels.to(torch.long)

    with torch.no_grad():
        preds = answer_logits.detach().argmax(dim=-1)
        correct = preds.eq(labels.unsqueeze(1))
        true_probs = torch.softmax(answer_logits.detach().float(), dim=-1)
        true_probs = true_probs.gather(
            2,
            labels.view(batch, 1, 1).expand(-1, samples, 1),
        ).squeeze(-1)
        if args.gram_oracle_ce_selection == "correct_then_prob":
            best_prob_indices = true_probs.argmax(dim=1)
            correct_scores = true_probs.masked_fill(~correct, -1.0)
            best_correct_indices = correct_scores.argmax(dim=1)
            has_correct = correct.any(dim=1)
            best_indices = torch.where(has_correct, best_correct_indices, best_prob_indices)
        else:
            best_indices = true_probs.argmax(dim=1)

        selected_correct = correct.gather(1, best_indices.unsqueeze(1)).squeeze(1)
        oracle_correct = correct.any(dim=1)
        selected_true_prob = true_probs.gather(1, best_indices.unsqueeze(1)).squeeze(1)

    selected_logits = answer_logits[torch.arange(batch, device=answer_logits.device), best_indices]
    loss = F.cross_entropy(selected_logits.float(), labels)
    metrics = {
        "selected_accuracy": float(selected_correct.float().mean().item()),
        "oracle_accuracy": float(oracle_correct.float().mean().item()),
        "mean_correct_samples": float(correct.float().mean().item()),
        "selected_true_prob": float(selected_true_prob.float().mean().item()),
    }
    return loss, metrics


@torch.inference_mode()
def evaluate_heldout_reasoning(
    model: Any,
    tokenizer: Any,
    *,
    args: Any,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    aim_run: Optional[Any],
) -> Dict[str, Any]:
    """Evaluate the normal recurrent answer path on unseen synthetic cases."""
    was_training = model.training
    model.eval()
    summary: Dict[str, Any] = {"depths": {}, "mean_accuracy": 0.0}
    accuracies: List[float] = []
    threshold_values = sorted(
        {
            float(value)
            for value in [args.lattice_candidate_threshold, *getattr(args, "lattice_eval_thresholds", [])]
            if 0.0 <= float(value) <= 1.0
        }
    )
    primary_threshold_key = lattice_threshold_key(float(args.lattice_candidate_threshold))
    threshold_accumulators: Dict[str, List[Dict[str, float]]] = {
        lattice_threshold_key(value): [] for value in threshold_values
    }
    stochastic_sample_values = sorted({int(value) for value in getattr(args, "stochastic_eval_samples", [1]) if int(value) > 0})
    stochastic_eval_summary: Dict[str, Dict[str, float]] = {
        str(value): {"correct": 0.0, "oracle_correct": 0.0, "total": 0.0, "cosine_sum": 0.0, "batches": 0.0}
        for value in stochastic_sample_values
        if value > 1
    }

    for depth in args.eval_depths:
        cases = build_generalized_synthetic_cases(
            count=args.eval_count,
            seed=args.eval_seed + int(depth),
            depths=[int(depth)],
            max_steps=int(depth),
            condition_prefix=args.reasoning_condition_prefix,
            family_mix=args.synthetic_family_mix,
            sampling_strategy="random",
        )
        cases = apply_synthetic_surface_mode(cases, mode=args.eval_surface_mode, seed=args.eval_seed + int(depth))
        correct = 0
        total = 0
        lattice_buckets: Dict[str, Dict[str, float]] = {
            lattice_threshold_key(threshold): {
                "threshold": float(threshold),
                "singleton": 0.0,
                "singleton_correct": 0.0,
                "true_alive": 0.0,
                "false_alive": 0.0,
                "false_total": 0.0,
                "alive_total": 0.0,
                "total": 0.0,
            }
            for threshold in threshold_values
        }
        by_family: Dict[str, Dict[str, int]] = {}
        for start in range(0, len(cases), args.eval_batch_size):
            batch_cases = cases[start : start + args.eval_batch_size]
            encoded = encode_text_batch(
                tokenizer,
                [case.prompt_text for case in batch_cases],
                args.max_length,
                device,
            )
            operation_ids = torch.tensor(
                [case.operation_ids for case in batch_cases],
                dtype=torch.long,
                device=device,
            )
            operation_arg_ids = torch.tensor(
                [case.operation_args or [0] * len(case.operation_ids) for case in batch_cases],
                dtype=torch.long,
                device=device,
            )
            initial_labels = torch.tensor(
                [case.initial_label for case in batch_cases],
                dtype=torch.long,
                device=device,
            )
            labels = torch.tensor(
                [case.answer_label for case in batch_cases],
                dtype=torch.long,
                device=device,
            )
            out = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                operation_ids=operation_ids if args.condition_on_operation_ids else None,
                operation_arg_ids=operation_arg_ids if args.condition_on_operation_ids else None,
                initial_labels=initial_labels,
                n_steps=int(depth),
            )
            pred = out["answer_logits"].argmax(dim=-1)
            matches = pred.eq(labels)
            correct += int(matches.sum().item())
            total += len(batch_cases)
            for samples, sample_bucket in stochastic_eval_summary.items():
                sample_count = int(samples)
                repeated_out = model(
                    input_ids=encoded["input_ids"].repeat_interleave(sample_count, dim=0),
                    attention_mask=encoded["attention_mask"].repeat_interleave(sample_count, dim=0),
                    operation_ids=(
                        operation_ids.repeat_interleave(sample_count, dim=0)
                        if args.condition_on_operation_ids
                        else None
                    ),
                    operation_arg_ids=(
                        operation_arg_ids.repeat_interleave(sample_count, dim=0)
                        if args.condition_on_operation_ids
                        else None
                    ),
                    initial_labels=initial_labels.repeat_interleave(sample_count, dim=0),
                    n_steps=int(depth),
                )
                selected_logits = select_stochastic_logits(
                    repeated_out,
                    batch_size=len(batch_cases),
                    samples=sample_count,
                    mode=args.stochastic_selection_mode,
                )
                sample_matches = selected_logits.argmax(dim=-1).eq(labels)
                sample_preds = repeated_out["answer_logits"].reshape(len(batch_cases), sample_count, -1).argmax(dim=-1)
                sample_oracle_matches = sample_preds.eq(labels.unsqueeze(1)).any(dim=1)
                sample_bucket["correct"] += float(sample_matches.sum().item())
                sample_bucket["oracle_correct"] += float(sample_oracle_matches.sum().item())
                sample_bucket["total"] += float(len(batch_cases))
                sample_bucket["cosine_sum"] += trajectory_cosine_similarity(
                    repeated_out["qtrm_core_step_states"],
                    samples=sample_count,
                )
                sample_bucket["batches"] += 1.0
            candidate_logits = out["state_digit_logits"][:, -1, :]
            candidate_probs = torch.sigmoid(candidate_logits.float())
            false_targets = F.one_hot(labels, num_classes=candidate_probs.size(-1)).to(torch.bool)
            for key, bucket in lattice_buckets.items():
                alive = candidate_probs.ge(float(bucket["threshold"]))
                singleton = alive.sum(dim=-1).eq(1)
                label_alive = alive.gather(1, labels.unsqueeze(1)).squeeze(1)
                false_alive = alive & ~false_targets
                bucket["singleton"] += float(singleton.sum().item())
                bucket["singleton_correct"] += float((singleton & label_alive).sum().item())
                bucket["true_alive"] += float(label_alive.sum().item())
                bucket["false_alive"] += float(false_alive.sum().item())
                bucket["false_total"] += float((~false_targets).sum().item())
                bucket["alive_total"] += float(alive.sum().item())
                bucket["total"] += float(len(batch_cases))
            for case, ok in zip(batch_cases, matches.tolist()):
                bucket = by_family.setdefault(case.family, {"correct": 0, "total": 0})
                bucket["correct"] += int(bool(ok))
                bucket["total"] += 1

        accuracy = correct / total if total else 0.0
        lattice_threshold_summary: Dict[str, Dict[str, float]] = {}
        for key, bucket in lattice_buckets.items():
            bucket_total = bucket["total"]
            singleton = bucket["singleton"]
            singleton_correct = bucket["singleton_correct"]
            false_total = bucket["false_total"]
            metrics = {
                "threshold": bucket["threshold"],
                "submit_rate": singleton / bucket_total if bucket_total else 0.0,
                "singleton_accuracy": singleton_correct / bucket_total if bucket_total else 0.0,
                "soundness": singleton_correct / singleton if singleton else 0.0,
                "true_alive_rate": bucket["true_alive"] / bucket_total if bucket_total else 0.0,
                "false_alive_rate": bucket["false_alive"] / false_total if false_total else 0.0,
                "mean_alive_count": bucket["alive_total"] / bucket_total if bucket_total else 0.0,
            }
            lattice_threshold_summary[key] = metrics
            threshold_accumulators[key].append(metrics)

        primary_metrics = lattice_threshold_summary[primary_threshold_key]
        lattice_submit_rate = primary_metrics["submit_rate"]
        lattice_singleton_accuracy = primary_metrics["singleton_accuracy"]
        lattice_soundness = primary_metrics["soundness"]
        lattice_true_alive_rate = primary_metrics["true_alive_rate"]
        lattice_false_alive_rate = primary_metrics["false_alive_rate"]
        accuracies.append(float(accuracy))
        family_summary = {
            family: {
                **stats,
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
            }
            for family, stats in by_family.items()
        }
        summary["depths"][str(depth)] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "lattice_submit_rate": lattice_submit_rate,
            "lattice_singleton_accuracy": lattice_singleton_accuracy,
            "lattice_soundness": lattice_soundness,
            "lattice_true_alive_rate": lattice_true_alive_rate,
            "lattice_false_alive_rate": lattice_false_alive_rate,
            "lattice_mean_alive_count": primary_metrics["mean_alive_count"],
            "lattice_thresholds": lattice_threshold_summary,
            "by_family": family_summary,
        }
        writer.add_scalar("Generalization/HeldOut/Accuracy_TRM", accuracy, int(depth))
        writer.add_scalar(f"Generalization/HeldOut/Depth_{depth}/Accuracy_TRM", accuracy, epoch)
        writer.add_scalar("Generalization/HeldOut/EpochDepthAccuracy_TRM", accuracy, epoch * 100 + int(depth))
        writer.add_scalar(f"Generalization/HeldOut/Depth_{depth}/Lattice_SubmitRate", lattice_submit_rate, epoch)
        writer.add_scalar(
            f"Generalization/HeldOut/Depth_{depth}/Lattice_SingletonAccuracy",
            lattice_singleton_accuracy,
            epoch,
        )
        writer.add_scalar(f"Generalization/HeldOut/Depth_{depth}/Lattice_Soundness", lattice_soundness, epoch)
        writer.add_scalar(
            f"Generalization/HeldOut/Depth_{depth}/Lattice_TrueAliveRate",
            lattice_true_alive_rate,
            epoch,
        )
        writer.add_scalar(
            f"Generalization/HeldOut/Depth_{depth}/Lattice_FalseAliveRate",
            lattice_false_alive_rate,
            epoch,
        )
        track_aim_scalar(
            aim_run,
            accuracy,
            name="generalization_trm_accuracy",
            epoch=epoch,
            context={"phase": "generalization", "split": "held_out", "depth": str(depth)},
        )
        lattice_context = {"phase": "generalization", "split": "held_out", "depth": str(depth)}
        track_aim_scalar(aim_run, lattice_submit_rate, name="lattice_submit_rate", epoch=epoch, context=lattice_context)
        track_aim_scalar(
            aim_run,
            lattice_singleton_accuracy,
            name="lattice_singleton_accuracy",
            epoch=epoch,
            context=lattice_context,
        )
        track_aim_scalar(aim_run, lattice_soundness, name="lattice_soundness", epoch=epoch, context=lattice_context)
        track_aim_scalar(
            aim_run,
            lattice_true_alive_rate,
            name="lattice_true_alive_rate",
            epoch=epoch,
            context=lattice_context,
        )
        track_aim_scalar(
            aim_run,
            lattice_false_alive_rate,
            name="lattice_false_alive_rate",
            epoch=epoch,
            context=lattice_context,
        )
        for threshold_key, metrics in lattice_threshold_summary.items():
            threshold_tag = lattice_threshold_tag(float(metrics["threshold"]))
            writer.add_scalar(
                f"Generalization/HeldOut/Depth_{depth}/Lattice_T{threshold_tag}/SubmitRate",
                float(metrics["submit_rate"]),
                epoch,
            )
            writer.add_scalar(
                f"Generalization/HeldOut/Depth_{depth}/Lattice_T{threshold_tag}/SingletonAccuracy",
                float(metrics["singleton_accuracy"]),
                epoch,
            )
            writer.add_scalar(
                f"Generalization/HeldOut/Depth_{depth}/Lattice_T{threshold_tag}/Soundness",
                float(metrics["soundness"]),
                epoch,
            )
            writer.add_scalar(
                f"Generalization/HeldOut/Depth_{depth}/Lattice_T{threshold_tag}/MeanAliveCount",
                float(metrics["mean_alive_count"]),
                epoch,
            )
            threshold_context = {
                "phase": "generalization",
                "split": "held_out",
                "depth": str(depth),
                "threshold": threshold_key,
            }
            track_aim_scalar(
                aim_run,
                float(metrics["submit_rate"]),
                name="lattice_submit_rate_by_threshold",
                epoch=epoch,
                context=threshold_context,
            )
            track_aim_scalar(
                aim_run,
                float(metrics["soundness"]),
                name="lattice_soundness_by_threshold",
                epoch=epoch,
                context=threshold_context,
            )
            track_aim_scalar(
                aim_run,
                float(metrics["mean_alive_count"]),
                name="lattice_mean_alive_count_by_threshold",
                epoch=epoch,
                context=threshold_context,
            )
        for family, stats in family_summary.items():
            writer.add_scalar(
                f"Generalization/HeldOut/Depth_{depth}/Family_{family}/Accuracy_TRM",
                float(stats["accuracy"]),
                epoch,
            )
            track_aim_scalar(
                aim_run,
                float(stats["accuracy"]),
                name="generalization_trm_family_accuracy",
                epoch=epoch,
                context={
                    "phase": "generalization",
                    "split": "held_out",
                    "depth": str(depth),
                    "family": str(family),
                },
            )

    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    summary["mean_accuracy"] = mean_accuracy
    if stochastic_eval_summary:
        summary["stochastic_eval_samples"] = {}
        for samples, sample_bucket in stochastic_eval_summary.items():
            sample_accuracy = sample_bucket["correct"] / sample_bucket["total"] if sample_bucket["total"] else 0.0
            sample_oracle_accuracy = (
                sample_bucket["oracle_correct"] / sample_bucket["total"] if sample_bucket["total"] else 0.0
            )
            sample_cosine = sample_bucket["cosine_sum"] / sample_bucket["batches"] if sample_bucket["batches"] else 1.0
            summary["stochastic_eval_samples"][samples] = {
                "accuracy": sample_accuracy,
                "oracle_accuracy": sample_oracle_accuracy,
                "trajectory_cosine": sample_cosine,
                "selection_mode": args.stochastic_selection_mode,
            }
            writer.add_scalar(f"Generalization/HeldOut/StochasticK{samples}/MeanAccuracy_TRM", sample_accuracy, epoch)
            writer.add_scalar(
                f"Generalization/HeldOut/StochasticK{samples}/MeanOracleAccuracy_TRM",
                sample_oracle_accuracy,
                epoch,
            )
            writer.add_scalar(f"Generalization/HeldOut/StochasticK{samples}/TrajectoryCosine", sample_cosine, epoch)
            track_aim_scalar(
                aim_run,
                sample_accuracy,
                name="generalization_trm_stochastic_accuracy",
                epoch=epoch,
                context={
                    "phase": "generalization",
                    "split": "held_out",
                    "samples": str(samples),
                    "selection": str(args.stochastic_selection_mode),
                },
            )
            track_aim_scalar(
                aim_run,
                sample_oracle_accuracy,
                name="generalization_trm_stochastic_oracle_accuracy",
                epoch=epoch,
                context={
                    "phase": "generalization",
                    "split": "held_out",
                    "samples": str(samples),
                    "selection": str(args.stochastic_selection_mode),
                },
            )
            track_aim_scalar(
                aim_run,
                sample_cosine,
                name="generalization_trm_trajectory_cosine",
                epoch=epoch,
                context={
                    "phase": "generalization",
                    "split": "held_out",
                    "samples": str(samples),
                    "selection": str(args.stochastic_selection_mode),
                },
            )
    summary["lattice_thresholds_mean"] = {}
    for threshold_key, values in threshold_accumulators.items():
        if not values:
            continue
        summary["lattice_thresholds_mean"][threshold_key] = {
            metric: sum(float(item[metric]) for item in values) / len(values)
            for metric in (
                "submit_rate",
                "singleton_accuracy",
                "soundness",
                "true_alive_rate",
                "false_alive_rate",
                "mean_alive_count",
            )
        }
    writer.add_scalar("Generalization/HeldOut/MeanAccuracy_TRM", mean_accuracy, epoch)
    track_aim_scalar(
        aim_run,
        mean_accuracy,
        name="generalization_trm_mean_accuracy",
        epoch=epoch,
        context={"phase": "generalization", "split": "held_out"},
    )
    writer.flush()
    if was_training:
        model.train()
    return summary


def _source_bucket(condition: str) -> Tuple[str, str]:
    parts = [part.strip() for part in str(condition).split(",") if part.strip()]
    family = parts[1] if len(parts) >= 2 else "unknown"
    source = parts[-1] if parts else "unknown"
    return family, source


@torch.inference_mode()
def evaluate_heldout_source_mix(
    model: Any,
    dataset: Dataset,
    *,
    args: Any,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    aim_run: Optional[Any],
) -> Dict[str, Any]:
    """Evaluate recurrent source training on held-out HRM-Text-style rows."""
    was_training = model.training
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.source_eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    by_source: Dict[str, Dict[str, float]] = {}
    by_family: Dict[str, Dict[str, float]] = {}

    for batch in loader:
        if "h_indices" not in batch:
            continue
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["healing_target_ids"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, n_steps=args.n_steps)
        target_steps = min(targets.size(1), out["qtrm_core_step_states"].size(1) - 1)
        target_slice = targets[:, :target_steps]
        step_states = out["qtrm_core_step_states"][:, 1 : 1 + target_steps, :]
        _, _, logits = compute_lm_token_loss_from_states(model, step_states, target_slice)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = target_slice.reshape(-1)
        valid = flat_targets.ne(IGNORE_INDEX)
        if not bool(valid.any().item()):
            continue

        loss_sum = F.cross_entropy(flat_logits, flat_targets, ignore_index=IGNORE_INDEX, reduction="sum")
        pred = logits.argmax(dim=-1)
        matches = pred.eq(target_slice) & target_slice.ne(IGNORE_INDEX)
        total_loss += float(loss_sum.item())
        total_correct += int(matches.sum().item())
        total_tokens += int(target_slice.ne(IGNORE_INDEX).sum().item())

        conditions = batch.get("conditions", ["unknown"] * target_slice.size(0))
        per_sample_correct = matches.sum(dim=1).tolist()
        per_sample_total = target_slice.ne(IGNORE_INDEX).sum(dim=1).tolist()
        for condition, correct, count in zip(conditions, per_sample_correct, per_sample_total):
            if int(count) <= 0:
                continue
            family, source = _source_bucket(condition)
            for bucket_name, buckets in ((family, by_family), (source, by_source)):
                bucket = buckets.setdefault(bucket_name, {"correct": 0.0, "total": 0.0})
                bucket["correct"] += float(correct)
                bucket["total"] += float(count)

    loss = total_loss / total_tokens if total_tokens else 0.0
    accuracy = total_correct / total_tokens if total_tokens else 0.0
    summary: Dict[str, Any] = {
        "loss": loss,
        "accuracy": accuracy,
        "correct": total_correct,
        "total": total_tokens,
        "by_source": {},
        "by_family": {},
    }

    writer.add_scalar("Generalization/VerifiedSource/Loss_TargetTokens", loss, epoch)
    writer.add_scalar("Generalization/VerifiedSource/Accuracy_TargetTokens", accuracy, epoch)
    track_aim_scalar(
        aim_run,
        loss,
        name="verified_source_target_token_loss",
        epoch=epoch,
        context={"phase": "generalization", "split": "verified_source"},
    )
    track_aim_scalar(
        aim_run,
        accuracy,
        name="verified_source_target_token_accuracy",
        epoch=epoch,
        context={"phase": "generalization", "split": "verified_source"},
    )

    for source, stats in sorted(by_source.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        summary["by_source"][source] = {**stats, "accuracy": acc}
        safe_source = source.replace("/", "_")
        writer.add_scalar(f"Generalization/VerifiedSource/Source_{safe_source}/Accuracy_TargetTokens", acc, epoch)
        track_aim_scalar(
            aim_run,
            acc,
            name="verified_source_target_token_accuracy_by_source",
            epoch=epoch,
            context={"phase": "generalization", "split": "verified_source", "source": str(source)},
        )

    for family, stats in sorted(by_family.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        summary["by_family"][family] = {**stats, "accuracy": acc}
        safe_family = family.replace("/", "_")
        writer.add_scalar(f"Generalization/VerifiedSource/Family_{safe_family}/Accuracy_TargetTokens", acc, epoch)
        track_aim_scalar(
            aim_run,
            acc,
            name="verified_source_target_token_accuracy_by_family",
            epoch=epoch,
            context={"phase": "generalization", "split": "verified_source", "family": str(family)},
        )

    writer.flush()
    if was_training:
        model.train()
    return summary


def checkpoint_state_dict(model: Any, trainable_only: bool) -> Dict[str, torch.Tensor]:
    if not trainable_only:
        return model.state_dict()
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = model.state_dict()
    return {name: value for name, value in state.items() if name in trainable_names}


def load_flexible_checkpoint(model: Any, checkpoint_path: str, device: torch.device) -> Dict[str, int]:
    """Load trainable checkpoints even when recurrent depth changed."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    current = model.state_dict()
    loaded: Dict[str, torch.Tensor] = {}
    exact = 0
    partial = 0
    skipped = 0

    for name, value in checkpoint.items():
        if name not in current:
            skipped += 1
            continue
        target = current[name]
        if tuple(value.shape) == tuple(target.shape):
            loaded[name] = value
            exact += 1
            continue
        if value.ndim == target.ndim and value.ndim >= 1 and tuple(value.shape[1:]) == tuple(target.shape[1:]):
            merged = target.clone()
            rows = min(value.shape[0], target.shape[0])
            merged[:rows] = value[:rows].to(dtype=target.dtype, device=target.device)
            loaded[name] = merged
            partial += 1
            continue
        skipped += 1

    # Stage25C shares the supervised state readout with core.state_readout so
    # candidate-lattice feedback uses the same head that receives state/lattice
    # supervision. Older checkpoints kept these heads separate and trained the
    # outer state_readout, while core.state_readout stayed at zero init. Prefer
    # the trained outer weights when loading such checkpoints into the shared
    # architecture.
    for name, value in checkpoint.items():
        if not name.startswith("state_readout."):
            continue
        alias = f"core.{name}"
        if alias not in current:
            continue
        target = current[alias]
        if tuple(value.shape) == tuple(target.shape):
            loaded[alias] = value
            continue
        if value.ndim == target.ndim and value.ndim >= 1 and tuple(value.shape[1:]) == tuple(target.shape[1:]):
            merged = target.clone()
            rows = min(value.shape[0], target.shape[0])
            merged[:rows] = value[:rows].to(dtype=target.dtype, device=target.device)
            loaded[alias] = merged

    current.update(loaded)
    model.load_state_dict(current, strict=True)
    return {"exact": exact, "partial": partial, "skipped": skipped, "checkpoint_tensors": len(checkpoint)}


def apply_recurrent_identity_overrides(model: Any, args: argparse.Namespace) -> Dict[str, int]:
    """Apply post-resume recurrence overrides for identity-biased ablations."""
    stats = {
        "transition_scale": 0,
        "injection_gate": 0,
        "step_embeddings_zeroed": 0,
        "step_embeddings_frozen": 0,
    }
    if args.override_transition_scale is not None:
        value = float(args.override_transition_scale)
        for module in model.modules():
            scale = getattr(module, "transition_scale", None)
            if isinstance(scale, torch.nn.Parameter):
                with torch.no_grad():
                    scale.fill_(value)
                stats["transition_scale"] += 1

    if args.override_injection_gate_logit is not None:
        value = float(args.override_injection_gate_logit)
        for module in model.modules():
            gate = getattr(module, "injection_gate", None)
            if isinstance(gate, torch.nn.Parameter):
                with torch.no_grad():
                    gate.fill_(value)
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


def configure_qwen_partial_training(model: Any, args: argparse.Namespace) -> Dict[str, object]:
    """Open only the Qwen speaker surfaces needed for LM-head alignment tests."""
    if not (
        args.train_qwen_embeddings
        or args.train_qwen_lm_head
        or args.train_qwen_final_norm
    ):
        return {"enabled": False}
    if not hasattr(model, "set_qwen_partial_trainable"):
        return {"enabled": False, "reason": "model_has_no_partial_training_hook"}
    stats = model.set_qwen_partial_trainable(
        train_embeddings=bool(args.train_qwen_embeddings),
        train_lm_head=bool(args.train_qwen_lm_head),
        train_final_norm=bool(args.train_qwen_final_norm),
    )
    stats["enabled"] = True
    return stats


def build_optimizer(model: Any, args: argparse.Namespace) -> torch.optim.Optimizer:
    """Build AdamW with optional LR groups for embedding/unembedding transfer tests."""
    default_params = []
    embedding_params = []
    lm_head_params = []
    final_norm_params = []
    trajectory_reward_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "trajectory_reward" in name:
            trajectory_reward_params.append(parameter)
        elif ".embed_tokens." in name:
            embedding_params.append(parameter)
        elif ".lm_head." in name:
            lm_head_params.append(parameter)
        elif name.startswith("qwen.") and ".norm." in name:
            final_norm_params.append(parameter)
        else:
            default_params.append(parameter)

    param_groups = []
    if default_params:
        param_groups.append({"params": default_params, "lr": args.lr, "name": "default"})
    if embedding_params:
        param_groups.append(
            {
                "params": embedding_params,
                "lr": args.lr * float(args.qwen_embedding_lr_multiplier),
                "name": "qwen_embeddings",
            }
        )
    if lm_head_params:
        param_groups.append(
            {
                "params": lm_head_params,
                "lr": args.lr * float(args.qwen_lm_head_lr_multiplier),
                "name": "qwen_lm_head",
            }
        )
    if final_norm_params:
        param_groups.append(
            {
                "params": final_norm_params,
                "lr": args.lr * float(args.qwen_final_norm_lr_multiplier),
                "name": "qwen_final_norm",
            }
        )
    if trajectory_reward_params:
        param_groups.append(
            {
                "params": trajectory_reward_params,
                "lr": args.lr * float(args.trajectory_reward_lr_multiplier),
                "name": "trajectory_reward",
            }
        )
    if not param_groups:
        raise ValueError("no trainable parameters available for optimizer")
    return torch.optim.AdamW(param_groups, lr=args.lr)


def configure_trajectory_reward_only_training(model: Any, enabled: bool) -> Dict[str, object]:
    """Freeze the thinker and train only the trajectory judge.

    Stage49 uses this to separate two questions that were previously tangled:
    whether GRAM can generate a correct path, and whether the learned verifier
    can select it. If only trajectory_reward* parameters are trainable, a
    failure is evidence against the verifier signal/features rather than another
    recurrent-core training failure.
    """
    if not enabled:
        return {"enabled": False}

    trainable = []
    frozen = 0
    for name, parameter in model.named_parameters():
        if "trajectory_reward" in name:
            parameter.requires_grad_(True)
            trainable.append(name)
        else:
            parameter.requires_grad_(False)
            frozen += 1
    return {
        "enabled": True,
        "trainable_count": len(trainable),
        "frozen_count": frozen,
        "trainable_names": trainable,
    }


def initialize_lattice_feedback_from_readout(model: Any, enabled: bool) -> Dict[str, float]:
    """Tie the candidate-lattice feedback basis to the supervised digit readout.

    LDT-style feedback has to project a 10-way candidate set back into the
    recurrent state. When resuming older checkpoints this projection is new, so
    a zero init would make early epochs behave almost like the no-feedback
    baseline. Using the transposed readout basis gives candidate bits an
    immediate, causal state-space direction while keeping the mapping local.
    """
    if not enabled:
        return {"initialized": 0.0}
    core = getattr(model, "core", None)
    proj = getattr(core, "lattice_feedback_proj", None)
    readout = getattr(model, "state_readout", None)
    if proj is None or readout is None:
        return {"initialized": 0.0}
    head = getattr(readout, "head", None)
    if head is None or tuple(proj.weight.shape) != tuple(head.weight.t().shape):
        return {"initialized": 0.0}
    with torch.no_grad():
        proj.weight.copy_(head.weight.detach().t().to(dtype=proj.weight.dtype, device=proj.weight.device))
    return {
        "initialized": 1.0,
        "projection_norm": float(proj.weight.detach().float().norm().item()),
        "readout_norm": float(head.weight.detach().float().norm().item()),
    }


def init_aim_run(args: Any) -> Optional[Any]:
    if not args.aim_repo:
        return None
    try:
        from aim import Run
    except ImportError as exc:
        print(f"[warn] Aim logging disabled; package is not installed: {exc}", flush=True)
        return None

    run = Run(repo=args.aim_repo, experiment=args.aim_experiment)
    run.name = args.aim_run_name or args.run_name or os.path.basename(os.path.normpath(args.out_dir))
    if args.aim_description:
        run.description = args.aim_description
    run["hparams"] = dict(vars(args))
    run["paths"] = {
        "out_dir": args.out_dir,
        "tensorboard_logdir": os.path.join(args.out_dir, "logs"),
    }
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


def configure_reproducibility(args: Any) -> torch.Generator:
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    generator = torch.Generator()
    generator.manual_seed(args.dataloader_seed if args.dataloader_seed is not None else args.seed)
    return generator


def scheduled_state_supervision_weight(args: Any, epoch: int) -> float:
    base = float(args.state_supervision_weight)
    decay = float(getattr(args, "state_supervision_decay_rate", 1.0))
    floor = float(getattr(args, "state_supervision_min_weight", 0.0))
    return max(floor, base * (decay ** max(0, int(epoch) - 1)))


def train_one_epoch(
    model: Any,
    loader: DataLoader,
    optimizer: Any,
    device: torch.device,
    args: Any,
    writer: SummaryWriter,
    epoch: int,
    aim_run: Optional[Any] = None,
):
    model.train()
    total_loss = 0.0
    total_r_loss = 0.0
    total_h_loss = 0.0
    total_posterior_answer_loss = 0.0
    total_aux_step_answer_loss = 0.0
    total_depth_consistency_loss = 0.0
    total_latent_shortcut_consistency_loss = 0.0
    total_final_readout_loss = 0.0
    total_trajectory_anchor_loss = 0.0
    total_correction_feedback_loss = 0.0
    total_operation_supervision_loss = 0.0
    total_prior_answer_loss = 0.0
    total_prior_posterior_logit_distill_loss = 0.0
    total_stochastic_posterior_kl_loss = 0.0
    total_gram_lprm_loss = 0.0
    total_gram_lprm_selected_acc = 0.0
    total_gram_lprm_oracle_acc = 0.0
    total_gram_lprm_target_spread = 0.0
    n_gram_lprm_multi = 0
    total_gram_oracle_ce_loss = 0.0
    total_gram_oracle_ce_selected_acc = 0.0
    total_gram_oracle_ce_oracle_acc = 0.0
    total_gram_oracle_ce_true_prob = 0.0
    n_gram_oracle_ce_multi = 0
    total_lattice_candidate_loss = 0.0
    total_semantic_lm_alignment_loss = 0.0
    total_semantic_step_alignment_loss = 0.0
    total_readout_gate = 0.0
    total_correction_gate = 0.0
    total_readout_entropy = 0.0
    total_lattice_true_alive = 0.0
    total_lattice_false_alive = 0.0
    total_stochastic_mu_norm = 0.0
    total_stochastic_std_mean = 0.0
    total_stochastic_noise_norm = 0.0
    total_working_register_norm = 0.0
    total_working_register_gate = 0.0
    total_working_register_role_cosine = 0.0
    total_semantic_token_feedback_gate = 0.0
    total_semantic_token_feedback_entropy = 0.0
    n_readout_gate = 0
    n_correction_gate = 0
    n_readout_entropy = 0
    n_lattice_metrics = 0
    n_stochastic_guidance = 0
    n_working_register = 0
    n_semantic_token_feedback = 0
    total_acc = 0.0
    total_prior_acc = 0.0
    n_reasoning = 0
    n_prior_reasoning = 0
    total_healing_correct = 0
    total_healing_tokens = 0
    total_state_norm = 0.0
    total_transition_norm = 0.0
    total_state_cosine = 0.0
    n_norms = 0
    state_supervision_weight = scheduled_state_supervision_weight(args, epoch)

    for batch_idx, batch in enumerate(loader):
        batch_size = batch["input_ids"].size(0)
        batch_loss = torch.tensor(0.0, device=device)
        r_loss_value = 0.0
        h_loss_value = 0.0
        h_acc_value: Optional[float] = None
        posterior_answer_loss_value = 0.0
        aux_step_answer_loss_value = 0.0
        depth_consistency_loss_value = 0.0
        latent_shortcut_consistency_loss_value = 0.0
        final_readout_loss_value = 0.0
        trajectory_anchor_loss_value = 0.0
        correction_feedback_loss_value = 0.0
        operation_supervision_loss_value = 0.0
        prior_answer_loss_value = 0.0
        prior_posterior_logit_distill_loss_value = 0.0
        prior_acc_value: Optional[float] = None
        stochastic_posterior_kl_loss_value = 0.0
        gram_lprm_loss_value = 0.0
        gram_lprm_selected_acc_value = None
        gram_lprm_oracle_acc_value = None
        gram_lprm_target_spread_value = None
        gram_oracle_ce_loss_value = 0.0
        gram_oracle_ce_selected_acc_value = None
        gram_oracle_ce_oracle_acc_value = None
        gram_oracle_ce_true_prob_value = None
        lattice_candidate_loss_value = 0.0
        semantic_lm_alignment_loss_value = 0.0
        semantic_step_alignment_loss_value = 0.0
        readout_gate_value = None
        correction_gate_value = None
        readout_entropy_value = None
        lattice_true_alive_value = None
        lattice_false_alive_value = None
        stochastic_mu_norm_value = None
        stochastic_std_mean_value = None
        stochastic_noise_norm_value = None
        working_register_norm_value = None
        working_register_gate_value = None
        working_register_role_cosine_value = None
        semantic_token_feedback_gate_value = None
        semantic_token_feedback_entropy_value = None
        state_norm_mean = None
        transition_norm_mean = None
        state_cosine_mean = None
        n_steps = args.n_steps
        if args.depth_sample_exact_batch_max:
            if "depths" in batch and batch["depths"].numel() > 0:
                n_steps = max(1, min(int(batch["depths"].max().item()), int(args.n_steps)))
        elif args.depth_sample_min > 0:
            min_steps = int(args.depth_sample_min)
            if "depths" in batch and batch["depths"].numel() > 0:
                min_steps = max(min_steps, int(batch["depths"].max().item()))
            min_steps = min(min_steps, int(args.n_steps))
            n_steps = random.randint(min_steps, args.n_steps)

        if "r_indices" in batch:
            idx = batch["r_indices"]
            input_ids = batch["input_ids"][idx].to(device)
            attention_mask = batch["attention_mask"][idx].to(device)
            operation_ids = fit_step_sequence(batch["operation_ids"].to(device), n_steps)
            operation_arg_ids = fit_step_sequence(batch["operation_arg_ids"].to(device), n_steps)
            initial_labels = batch["initial_labels"].to(device)
            answer_labels = batch["reasoning_labels"].to(device)
            state_labels = fit_step_sequence(batch["state_labels"].to(device), n_steps)

            if args.stochastic_posterior_guidance and args.stochastic_transition_mode == "true_gram":
                posterior_labels = state_labels
            elif args.stochastic_posterior_guidance:
                posterior_labels = answer_labels
            else:
                posterior_labels = None
            conditioned_operation_ids = operation_ids if args.condition_on_operation_ids else None
            conditioned_operation_arg_ids = operation_arg_ids if args.condition_on_operation_ids else None
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                operation_ids=conditioned_operation_ids,
                operation_arg_ids=conditioned_operation_arg_ids,
                initial_labels=initial_labels,
                n_steps=n_steps,
                posterior_labels=posterior_labels,
            )
            if "qtrm_readout_gate" in out:
                readout_gate_value = float(out["qtrm_readout_gate"].detach().mean().item())
                total_readout_gate += readout_gate_value * len(idx)
                n_readout_gate += len(idx)
            if "qtrm_readout_attention_entropy" in out:
                readout_entropy_value = float(out["qtrm_readout_attention_entropy"].detach().mean().item())
                total_readout_entropy += readout_entropy_value * len(idx)
                n_readout_entropy += len(idx)
            if out.get("qtrm_correction_gate") is not None:
                correction_gate_value = float(out["qtrm_correction_gate"].detach().mean().item())
                total_correction_gate += correction_gate_value * len(idx)
                n_correction_gate += len(idx)
            if out.get("qtrm_stochastic_mu_norms") is not None:
                stochastic_mu_norm_value = float(out["qtrm_stochastic_mu_norms"].detach().mean().item())
                stochastic_std_mean_value = float(out["qtrm_stochastic_std_means"].detach().mean().item())
                stochastic_noise_norm_value = float(out["qtrm_stochastic_noise_norms"].detach().mean().item())
                total_stochastic_mu_norm += stochastic_mu_norm_value * len(idx)
                total_stochastic_std_mean += stochastic_std_mean_value * len(idx)
                total_stochastic_noise_norm += stochastic_noise_norm_value * len(idx)
                n_stochastic_guidance += len(idx)
            if out.get("qtrm_working_register_norms") is not None:
                working_register_norm_value = float(out["qtrm_working_register_norms"].detach().mean().item())
                working_register_gate_value = float(out["qtrm_working_register_gate_means"].detach().mean().item())
                working_register_role_cosine_value = (
                    float(out["qtrm_working_register_role_cosines"].detach().mean().item())
                    if out.get("qtrm_working_register_role_cosines") is not None
                    else None
                )
                total_working_register_norm += working_register_norm_value * len(idx)
                total_working_register_gate += working_register_gate_value * len(idx)
                if working_register_role_cosine_value is not None:
                    total_working_register_role_cosine += working_register_role_cosine_value * len(idx)
                n_working_register += len(idx)
            if out.get("qtrm_semantic_token_feedback_gate_means") is not None:
                semantic_token_feedback_gate_value = float(
                    out["qtrm_semantic_token_feedback_gate_means"].detach().mean().item()
                )
                semantic_token_feedback_entropy_value = float(
                    out["qtrm_semantic_token_feedback_entropies"].detach().mean().item()
                )
                total_semantic_token_feedback_gate += semantic_token_feedback_gate_value * len(idx)
                total_semantic_token_feedback_entropy += semantic_token_feedback_entropy_value * len(idx)
                n_semantic_token_feedback += len(idx)
            state_norm_mean = float(out["state_norms"].mean().item()) if "state_norms" in out and out["state_norms"] is not None else 0.0
            transition_norm_mean = float(out["transition_norms"].mean().item()) if "transition_norms" in out and out["transition_norms"] is not None else 0.0
            state_cosine_mean = float(out["state_cosines"].mean().item()) if "state_cosines" in out and out["state_cosines"] is not None else 0.0
            total_state_norm += state_norm_mean * len(idx)
            total_transition_norm += transition_norm_mean * len(idx)
            total_state_cosine += state_cosine_mean * len(idx)
            n_norms += len(idx)
            answer_loss = F.cross_entropy(out["answer_logits"], answer_labels)
            state_loss = F.cross_entropy(out["state_digit_logits"][:, 1:, :].reshape(-1, 10), state_labels.reshape(-1))
            correction_feedback_loss = torch.tensor(0.0, device=device)
            if args.correction_feedback_loss_weight > 0 and out.get("qtrm_correction_error_logits") is not None:
                first_pred = out["qtrm_first_answer_logits"].detach().argmax(dim=-1)
                correction_targets = (answer_labels - first_pred).remainder(10)
                correction_feedback_loss = F.cross_entropy(out["qtrm_correction_error_logits"], correction_targets)
            aux_step_answer_loss = torch.tensor(0.0, device=device)
            if args.aux_step_answer_weight > 0:
                aux_step_answer_loss, _ = compute_step_answer_loss(
                    model,
                    out["qtrm_core_step_states"],
                    state_labels,
                )
            prior_aux_step_answer_loss = torch.tensor(0.0, device=device)
            operation_supervision_loss = torch.tensor(0.0, device=device)
            if args.operation_supervision_weight > 0:
                operation_supervision_loss = compute_operation_supervision_loss(
                    out["operation_logits"],
                    operation_ids,
                )
            prior_answer_loss = torch.tensor(0.0, device=device)
            prior_posterior_logit_distill_loss = torch.tensor(0.0, device=device)
            prior_out = None
            needs_prior_forward = (
                args.prior_answer_weight > 0
                or args.prior_posterior_logit_distill_weight > 0
                or args.semantic_step_alignment_weight > 0
                or args.prior_aux_step_answer_weight > 0
            )
            if needs_prior_forward:
                # True-GRAM train accuracy can be misleading when the forward pass
                # uses posterior labels. This extra pass trains and measures the
                # same prior-only path used at evaluation time.
                prior_out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    operation_ids=conditioned_operation_ids,
                    operation_arg_ids=conditioned_operation_arg_ids,
                    initial_labels=initial_labels,
                    n_steps=n_steps,
                    posterior_labels=None,
                )
                prior_answer_loss = F.cross_entropy(prior_out["answer_logits"], answer_labels)
                with torch.no_grad():
                    prior_matches = prior_out["answer_logits"].argmax(-1).eq(answer_labels)
                    total_prior_acc += float(prior_matches.float().sum().item())
                    n_prior_reasoning += len(idx)
                    prior_acc_value = float(prior_matches.float().mean().item())
                if args.prior_posterior_logit_distill_weight > 0:
                    distill_temp = max(float(args.prior_posterior_distill_temperature), 1e-6)
                    teacher_probs = F.softmax(out["answer_logits"].detach().float() / distill_temp, dim=-1)
                    student_log_probs = F.log_softmax(prior_out["answer_logits"].float() / distill_temp, dim=-1)
                    prior_posterior_logit_distill_loss = (
                        F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
                        * (distill_temp * distill_temp)
                    )
                if args.prior_aux_step_answer_weight > 0:
                    prior_aux_step_answer_loss, _ = compute_step_answer_loss(
                        model,
                        prior_out["qtrm_core_step_states"],
                        state_labels,
                    )
            stochastic_posterior_kl_loss = torch.tensor(0.0, device=device)
            if args.stochastic_posterior_kl_weight > 0 and out.get("qtrm_stochastic_posterior_kls") is not None:
                stochastic_posterior_kl_loss = out["qtrm_stochastic_posterior_kls"]
                if args.stochastic_posterior_kl_free_bits > 0:
                    stochastic_posterior_kl_loss = torch.clamp(
                        stochastic_posterior_kl_loss - float(args.stochastic_posterior_kl_free_bits),
                        min=0.0,
                    )
                stochastic_posterior_kl_loss = stochastic_posterior_kl_loss.mean()
                if args.stochastic_posterior_kl_warmup_steps > 0:
                    global_step = (epoch - 1) * len(loader) + batch_idx
                    warmup = min(1.0, float(global_step) / float(args.stochastic_posterior_kl_warmup_steps))
                    stochastic_posterior_kl_loss = stochastic_posterior_kl_loss * warmup
            gram_lprm_loss = torch.tensor(0.0, device=device)
            if args.gram_lprm_weight > 0:
                if int(args.gram_lprm_train_samples) > 1:
                    gram_lprm_loss, gram_lprm_metrics = compute_multi_trajectory_lprm_loss(
                        model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        operation_ids=conditioned_operation_ids,
                        operation_arg_ids=conditioned_operation_arg_ids,
                        initial_labels=initial_labels,
                        answer_labels=answer_labels,
                        n_steps=n_steps,
                        args=args,
                    )
                    gram_lprm_selected_acc_value = gram_lprm_metrics["selected_accuracy"]
                    gram_lprm_oracle_acc_value = gram_lprm_metrics["oracle_accuracy"]
                    gram_lprm_target_spread_value = gram_lprm_metrics["target_spread"]
                    total_gram_lprm_selected_acc += gram_lprm_selected_acc_value * len(idx)
                    total_gram_lprm_oracle_acc += gram_lprm_oracle_acc_value * len(idx)
                    total_gram_lprm_target_spread += gram_lprm_target_spread_value * len(idx)
                    n_gram_lprm_multi += len(idx)
                else:
                    reward_logits = trajectory_reward_logits_from_output(
                        model,
                        out,
                        detach_state=bool(args.gram_lprm_detach_state),
                    )
                    if args.gram_lprm_target == "correct":
                        reward_targets = (out["answer_logits"].detach().argmax(dim=-1) == answer_labels).to(reward_logits.dtype)
                    else:
                        reward_targets = torch.softmax(out["answer_logits"].detach().float(), dim=-1)
                        reward_targets = reward_targets.gather(1, answer_labels.unsqueeze(1)).squeeze(1).to(reward_logits.dtype)
                    gram_lprm_loss = F.binary_cross_entropy_with_logits(reward_logits, reward_targets)
            gram_oracle_ce_loss = torch.tensor(0.0, device=device)
            if args.gram_oracle_ce_weight > 0:
                gram_oracle_ce_loss, gram_oracle_ce_metrics = compute_multi_trajectory_oracle_ce_loss(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    operation_ids=conditioned_operation_ids,
                    operation_arg_ids=conditioned_operation_arg_ids,
                    initial_labels=initial_labels,
                    answer_labels=answer_labels,
                    n_steps=n_steps,
                    args=args,
                )
                gram_oracle_ce_selected_acc_value = gram_oracle_ce_metrics["selected_accuracy"]
                gram_oracle_ce_oracle_acc_value = gram_oracle_ce_metrics["oracle_accuracy"]
                gram_oracle_ce_true_prob_value = gram_oracle_ce_metrics["selected_true_prob"]
                total_gram_oracle_ce_selected_acc += gram_oracle_ce_selected_acc_value * len(idx)
                total_gram_oracle_ce_oracle_acc += gram_oracle_ce_oracle_acc_value * len(idx)
                total_gram_oracle_ce_true_prob += gram_oracle_ce_true_prob_value * len(idx)
                n_gram_oracle_ce_multi += len(idx)
            semantic_lm_alignment_loss = torch.tensor(0.0, device=device)
            if args.semantic_lm_alignment_weight > 0 and getattr(model, "answer_path", "state_head") == "lm_head":
                label_token_ids = model.label_token_ids.to(device=device, dtype=torch.long)
                answer_token_ids = label_token_ids.index_select(0, answer_labels.to(torch.long)).unsqueeze(1)
                semantic_lm_alignment_loss = _lm_target_embedding_alignment_loss(
                    model,
                    out["qtrm_readout_state"].unsqueeze(1),
                    answer_token_ids,
                )
            semantic_step_alignment_loss = torch.tensor(0.0, device=device)
            if args.semantic_step_alignment_weight > 0 and getattr(model, "answer_path", "state_head") == "lm_head":
                label_token_ids = model.label_token_ids.to(device=device, dtype=torch.long)
                step_token_ids = label_token_ids.index_select(0, state_labels.reshape(-1).to(torch.long))
                step_token_ids = step_token_ids.reshape_as(state_labels)
                step_alignment_out = prior_out if prior_out is not None else out
                target_steps = min(state_labels.size(1), step_alignment_out["qtrm_core_step_states"].size(1) - 1)
                semantic_step_alignment_loss = _lm_target_embedding_alignment_loss(
                    model,
                    step_alignment_out["qtrm_core_step_states"][:, 1 : 1 + target_steps, :],
                    step_token_ids[:, :target_steps],
                )
            lattice_candidate_loss = torch.tensor(0.0, device=device)
            if args.lattice_candidate_weight > 0:
                lattice_targets = state_labels if args.lattice_candidate_target == "state_labels" else answer_labels
                lattice_candidate_loss, lattice_metrics = compute_lattice_candidate_loss(
                    out["state_digit_logits"],
                    lattice_targets,
                    min_step=args.lattice_candidate_min_step,
                    positive_weight=args.lattice_candidate_positive_weight,
                    negative_weight=args.lattice_candidate_negative_weight,
                )
                lattice_true_alive_value = lattice_metrics["true_alive_prob"]
                lattice_false_alive_value = lattice_metrics["false_alive_prob"]
                total_lattice_true_alive += lattice_true_alive_value * len(idx)
                total_lattice_false_alive += lattice_false_alive_value * len(idx)
                n_lattice_metrics += len(idx)
            depth_consistency_loss = torch.tensor(0.0, device=device)
            latent_shortcut_consistency_loss = torch.tensor(0.0, device=device)
            if (
                (args.depth_consistency_weight > 0 or args.latent_shortcut_consistency_weight > 0)
                and n_steps > 1
            ):
                consistency_steps = n_steps - 1
                if args.consistency_min_steps > 0 and args.consistency_min_steps <= n_steps - 1:
                    consistency_steps = random.randint(args.consistency_min_steps, n_steps - 1)
                shallow_operation_ids = fit_step_sequence(batch["operation_ids"].to(device), consistency_steps)
                shallow_operation_arg_ids = fit_step_sequence(batch["operation_arg_ids"].to(device), consistency_steps)
                conditioned_shallow_operation_ids = shallow_operation_ids if args.condition_on_operation_ids else None
                conditioned_shallow_operation_arg_ids = (
                    shallow_operation_arg_ids if args.condition_on_operation_ids else None
                )
                shallow_out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    operation_ids=conditioned_shallow_operation_ids,
                    operation_arg_ids=conditioned_shallow_operation_arg_ids,
                    initial_labels=initial_labels,
                    n_steps=consistency_steps,
                )
                if args.depth_consistency_weight > 0:
                    depth_consistency_loss = compute_depth_consistency_loss(
                        teacher_logits=out["answer_logits"],
                        student_logits=shallow_out["answer_logits"],
                        temperature=args.depth_consistency_temperature,
                    )
                if args.latent_shortcut_consistency_weight > 0:
                    latent_shortcut_consistency_loss = compute_latent_shortcut_consistency_loss(
                        short_state_trajectory=shallow_out["qtrm_core_step_states"],
                        long_state_trajectory=out["qtrm_core_step_states"],
                        min_step=args.latent_shortcut_consistency_min_step,
                    )
            final_readout_loss = torch.tensor(0.0, device=device)
            if args.final_readout_answer_weight > 0 and getattr(model, "recurrent_readout_pooling", "final") != "final":
                final_readout_logits = compute_final_readout_logits(model, out["qtrm_core_step_states"])
                final_readout_loss = F.cross_entropy(final_readout_logits, answer_labels)
            trajectory_anchor_loss = torch.tensor(0.0, device=device)
            if args.trajectory_anchor_weight > 0:
                trajectory_anchor_loss = compute_trajectory_anchor_loss(
                    state_trajectory=out["qtrm_core_step_states"],
                    teacher_state=out["qtrm_workspace"][:, 0, :],
                    min_step=args.trajectory_anchor_min_step,
                )
            reasoning_loss = (
                answer_loss * float(args.posterior_answer_weight)
                + state_loss * state_supervision_weight
                + aux_step_answer_loss * float(args.aux_step_answer_weight)
                + prior_aux_step_answer_loss * float(args.prior_aux_step_answer_weight)
                + depth_consistency_loss * float(args.depth_consistency_weight)
                + latent_shortcut_consistency_loss * float(args.latent_shortcut_consistency_weight)
                + final_readout_loss * float(args.final_readout_answer_weight)
                + trajectory_anchor_loss * float(args.trajectory_anchor_weight)
                + correction_feedback_loss * float(args.correction_feedback_loss_weight)
                + operation_supervision_loss * float(args.operation_supervision_weight)
                + prior_answer_loss * float(args.prior_answer_weight)
                + prior_posterior_logit_distill_loss * float(args.prior_posterior_logit_distill_weight)
                + stochastic_posterior_kl_loss * float(args.stochastic_posterior_kl_weight)
                + gram_lprm_loss * float(args.gram_lprm_weight)
                + gram_oracle_ce_loss * float(args.gram_oracle_ce_weight)
                + semantic_lm_alignment_loss * float(args.semantic_lm_alignment_weight)
                + semantic_step_alignment_loss * float(args.semantic_step_alignment_weight)
                + lattice_candidate_loss * float(args.lattice_candidate_weight)
            ) * float(args.reasoning_weight) * (len(idx) / batch_size)
            batch_loss = batch_loss + reasoning_loss
            r_loss_value = reasoning_loss.item()
            posterior_answer_loss_value = (
                answer_loss * float(args.posterior_answer_weight) * float(args.reasoning_weight) * (len(idx) / batch_size)
            ).item()
            aux_step_answer_loss_value = (
                (
                    aux_step_answer_loss * float(args.aux_step_answer_weight)
                    + prior_aux_step_answer_loss * float(args.prior_aux_step_answer_weight)
                )
                * float(args.reasoning_weight)
                * (len(idx) / batch_size)
            ).item()
            depth_consistency_loss_value = (
                depth_consistency_loss * float(args.depth_consistency_weight) * float(args.reasoning_weight) * (len(idx) / batch_size)
            ).item()
            latent_shortcut_consistency_loss_value = (
                latent_shortcut_consistency_loss
                * float(args.latent_shortcut_consistency_weight)
                * float(args.reasoning_weight)
                * (len(idx) / batch_size)
            ).item()
            final_readout_loss_value = (
                final_readout_loss * float(args.final_readout_answer_weight) * float(args.reasoning_weight) * (len(idx) / batch_size)
            ).item()
            trajectory_anchor_loss_value = (
                trajectory_anchor_loss * float(args.trajectory_anchor_weight) * float(args.reasoning_weight) * (len(idx) / batch_size)
            ).item()
            correction_feedback_loss_value = (
                correction_feedback_loss * float(args.correction_feedback_loss_weight) * float(args.reasoning_weight) * (len(idx) / batch_size)
            ).item()
            operation_supervision_loss_value = (
                operation_supervision_loss * float(args.operation_supervision_weight) * float(args.reasoning_weight) * (len(idx) / batch_size)
            ).item()
            prior_answer_loss_value = (
                prior_answer_loss * float(args.prior_answer_weight) * float(args.reasoning_weight) * (len(idx) / batch_size)
            ).item()
            prior_posterior_logit_distill_loss_value = (
                prior_posterior_logit_distill_loss
                * float(args.prior_posterior_logit_distill_weight)
                * float(args.reasoning_weight)
                * (len(idx) / batch_size)
            ).item()
            stochastic_posterior_kl_loss_value = (
                stochastic_posterior_kl_loss * float(args.stochastic_posterior_kl_weight) * float(args.reasoning_weight) * (len(idx) / batch_size)
            ).item()
            gram_lprm_loss_value = (
                gram_lprm_loss * float(args.gram_lprm_weight) * float(args.reasoning_weight) * (len(idx) / batch_size)
            ).item()
            gram_oracle_ce_loss_value = (
                gram_oracle_ce_loss * float(args.gram_oracle_ce_weight) * float(args.reasoning_weight) * (len(idx) / batch_size)
            ).item()
            semantic_lm_alignment_loss_value = (
                semantic_lm_alignment_loss
                * float(args.semantic_lm_alignment_weight)
                * float(args.reasoning_weight)
                * (len(idx) / batch_size)
            ).item()
            semantic_step_alignment_loss_value = (
                semantic_step_alignment_loss
                * float(args.semantic_step_alignment_weight)
                * float(args.reasoning_weight)
                * (len(idx) / batch_size)
            ).item()
            lattice_candidate_loss_value = (
                lattice_candidate_loss * float(args.lattice_candidate_weight) * float(args.reasoning_weight) * (len(idx) / batch_size)
            ).item()
            total_acc += (out["answer_logits"].argmax(-1) == answer_labels).float().sum().item()
            n_reasoning += len(idx)

        if "h_indices" in batch:
            idx = batch["h_indices"]
            input_ids = batch["input_ids"][idx].to(device)
            attention_mask = batch["attention_mask"][idx].to(device)
            targets = batch["healing_target_ids"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, n_steps=n_steps)
            target_steps = min(targets.size(1), out["qtrm_core_step_states"].size(1) - 1)
            target_slice = targets[:, :target_steps]
            step_states = out["qtrm_core_step_states"][:, 1 : 1 + target_steps, :]
            healing_token_loss, healing_alignment_loss, logits = compute_lm_token_loss_from_states(
                model,
                step_states,
                target_slice,
            )
            healing_loss = (
                (
                    healing_token_loss
                    + healing_alignment_loss * float(args.semantic_lm_alignment_weight)
                )
                * float(args.healing_weight)
                * (len(idx) / batch_size)
            )
            batch_loss = batch_loss + healing_loss
            h_loss_value = healing_loss.item()
            semantic_lm_alignment_loss_value += (
                healing_alignment_loss
                * float(args.semantic_lm_alignment_weight)
                * float(args.healing_weight)
                * (len(idx) / batch_size)
            ).item()
            valid_targets = target_slice.ne(IGNORE_INDEX)
            if bool(valid_targets.any().item()):
                healing_matches = logits.argmax(dim=-1).eq(target_slice) & valid_targets
                total_healing_correct += int(healing_matches.sum().item())
                total_healing_tokens += int(valid_targets.sum().item())
                h_acc_value = float(healing_matches.sum().item() / valid_targets.sum().item())

        optimizer.zero_grad()
        batch_loss.backward()
        grad_norm = 0.0
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        total_loss += batch_loss.item() * batch_size
        total_r_loss += r_loss_value * batch_size
        total_h_loss += h_loss_value * batch_size
        total_posterior_answer_loss += posterior_answer_loss_value * batch_size
        total_aux_step_answer_loss += aux_step_answer_loss_value * batch_size
        total_depth_consistency_loss += depth_consistency_loss_value * batch_size
        total_latent_shortcut_consistency_loss += latent_shortcut_consistency_loss_value * batch_size
        total_final_readout_loss += final_readout_loss_value * batch_size
        total_trajectory_anchor_loss += trajectory_anchor_loss_value * batch_size
        total_correction_feedback_loss += correction_feedback_loss_value * batch_size
        total_operation_supervision_loss += operation_supervision_loss_value * batch_size
        total_prior_answer_loss += prior_answer_loss_value * batch_size
        total_prior_posterior_logit_distill_loss += prior_posterior_logit_distill_loss_value * batch_size
        total_stochastic_posterior_kl_loss += stochastic_posterior_kl_loss_value * batch_size
        total_gram_lprm_loss += gram_lprm_loss_value * batch_size
        total_gram_oracle_ce_loss += gram_oracle_ce_loss_value * batch_size
        total_lattice_candidate_loss += lattice_candidate_loss_value * batch_size
        total_semantic_lm_alignment_loss += semantic_lm_alignment_loss_value * batch_size
        total_semantic_step_alignment_loss += semantic_step_alignment_loss_value * batch_size

        if batch_idx % int(args.log_every) == 0:
            step = (epoch - 1) * len(loader) + batch_idx
            writer.add_scalar("Step/Loss_Total", batch_loss.item(), step)
            writer.add_scalar("Step/Loss_Reasoning", r_loss_value, step)
            writer.add_scalar("Step/Loss_Healing", h_loss_value, step)
            writer.add_scalar("Step/Loss_PosteriorAnswer", posterior_answer_loss_value, step)
            if h_acc_value is not None:
                writer.add_scalar("Step/Accuracy_HealingTargetTokens", h_acc_value, step)
                writer.add_scalar("Train/Step/Accuracy_HealingTargetTokens", h_acc_value, step)
            writer.add_scalar("Step/Loss_AuxStepAnswer", aux_step_answer_loss_value, step)
            writer.add_scalar("Step/Loss_DepthConsistency", depth_consistency_loss_value, step)
            writer.add_scalar("Step/Loss_LatentShortcutConsistency", latent_shortcut_consistency_loss_value, step)
            writer.add_scalar("Step/Loss_FinalReadoutAnswer", final_readout_loss_value, step)
            writer.add_scalar("Step/Loss_TrajectoryAnchor", trajectory_anchor_loss_value, step)
            writer.add_scalar("Step/Loss_CorrectionFeedback", correction_feedback_loss_value, step)
            writer.add_scalar("Step/Loss_OperationSupervision", operation_supervision_loss_value, step)
            writer.add_scalar("Step/Loss_PriorAnswer", prior_answer_loss_value, step)
            writer.add_scalar("Step/Loss_PriorPosteriorLogitDistill", prior_posterior_logit_distill_loss_value, step)
            writer.add_scalar("Step/Loss_StochasticPosteriorKL", stochastic_posterior_kl_loss_value, step)
            writer.add_scalar("Step/Loss_GRAM_LPRM", gram_lprm_loss_value, step)
            writer.add_scalar("Step/Loss_GRAM_OracleCE", gram_oracle_ce_loss_value, step)
            if gram_lprm_selected_acc_value is not None:
                writer.add_scalar("Step/GRAM_LPRM_SelectedAccuracy", gram_lprm_selected_acc_value, step)
                writer.add_scalar("Step/GRAM_LPRM_OracleAccuracy", gram_lprm_oracle_acc_value, step)
                writer.add_scalar("Step/GRAM_LPRM_TargetSpread", gram_lprm_target_spread_value, step)
                writer.add_scalar("Train/Step/GRAM_LPRM_SelectedAccuracy", gram_lprm_selected_acc_value, step)
                writer.add_scalar("Train/Step/GRAM_LPRM_OracleAccuracy", gram_lprm_oracle_acc_value, step)
                writer.add_scalar("Train/Step/GRAM_LPRM_TargetSpread", gram_lprm_target_spread_value, step)
            if gram_oracle_ce_selected_acc_value is not None:
                writer.add_scalar("Step/GRAM_OracleCE_SelectedAccuracy", gram_oracle_ce_selected_acc_value, step)
                writer.add_scalar("Step/GRAM_OracleCE_OracleAccuracy", gram_oracle_ce_oracle_acc_value, step)
                writer.add_scalar("Step/GRAM_OracleCE_SelectedTrueProb", gram_oracle_ce_true_prob_value, step)
                writer.add_scalar("Train/Step/GRAM_OracleCE_SelectedAccuracy", gram_oracle_ce_selected_acc_value, step)
                writer.add_scalar("Train/Step/GRAM_OracleCE_OracleAccuracy", gram_oracle_ce_oracle_acc_value, step)
                writer.add_scalar("Train/Step/GRAM_OracleCE_SelectedTrueProb", gram_oracle_ce_true_prob_value, step)
            writer.add_scalar("Step/Loss_SemanticLMAlignment", semantic_lm_alignment_loss_value, step)
            writer.add_scalar("Step/Loss_SemanticStepAlignment", semantic_step_alignment_loss_value, step)
            writer.add_scalar("Step/Loss_LatticeCandidate", lattice_candidate_loss_value, step)
            writer.add_scalar("Step/Grad_Norm", float(grad_norm), step)
            writer.add_scalar("Step/N_Steps", n_steps, step)
            writer.add_scalar("Step/StateSupervisionWeight", state_supervision_weight, step)
            if state_norm_mean is not None:
                writer.add_scalar("Train/Step/StateNorm_Mean", state_norm_mean, step)
                writer.add_scalar("Train/Step/TransitionNorm_Mean", transition_norm_mean, step)
                writer.add_scalar("Train/Step/StateCosine_Mean", state_cosine_mean, step)
            if lattice_true_alive_value is not None:
                writer.add_scalar("Step/Lattice_TrueAliveProb", lattice_true_alive_value, step)
                writer.add_scalar("Step/Lattice_FalseAliveProb", lattice_false_alive_value, step)
                writer.add_scalar("Train/Step/Lattice_TrueAliveProb", lattice_true_alive_value, step)
                writer.add_scalar("Train/Step/Lattice_FalseAliveProb", lattice_false_alive_value, step)
            if readout_gate_value is not None:
                writer.add_scalar("Step/Readout_Gate_AttentionWeight", readout_gate_value, step)
                writer.add_scalar("Train/Step/Readout_Gate_AttentionWeight", readout_gate_value, step)
            if readout_entropy_value is not None:
                writer.add_scalar("Step/Readout_Attention_Entropy", readout_entropy_value, step)
                writer.add_scalar("Train/Step/Readout_Attention_Entropy", readout_entropy_value, step)
            if correction_gate_value is not None:
                writer.add_scalar("Step/Correction_Feedback_Gate", correction_gate_value, step)
                writer.add_scalar("Train/Step/Correction_Feedback_Gate", correction_gate_value, step)
            if stochastic_mu_norm_value is not None:
                writer.add_scalar("Step/StochasticHighLevel_MuNorm", stochastic_mu_norm_value, step)
                writer.add_scalar("Step/StochasticHighLevel_StdMean", stochastic_std_mean_value, step)
                writer.add_scalar("Step/StochasticHighLevel_NoiseNorm", stochastic_noise_norm_value, step)
                writer.add_scalar("Train/Step/StochasticHighLevel_MuNorm", stochastic_mu_norm_value, step)
                writer.add_scalar("Train/Step/StochasticHighLevel_StdMean", stochastic_std_mean_value, step)
                writer.add_scalar("Train/Step/StochasticHighLevel_NoiseNorm", stochastic_noise_norm_value, step)
            if working_register_norm_value is not None:
                writer.add_scalar("Step/WorkingRegister_Norm", working_register_norm_value, step)
                writer.add_scalar("Step/WorkingRegister_GateMean", working_register_gate_value, step)
                writer.add_scalar("Train/Step/WorkingRegister_Norm", working_register_norm_value, step)
                writer.add_scalar("Train/Step/WorkingRegister_GateMean", working_register_gate_value, step)
                if working_register_role_cosine_value is not None:
                    writer.add_scalar("Step/WorkingRegister_RoleCosine", working_register_role_cosine_value, step)
                    writer.add_scalar("Train/Step/WorkingRegister_RoleCosine", working_register_role_cosine_value, step)
            if semantic_token_feedback_gate_value is not None:
                writer.add_scalar("Step/SemanticTokenFeedback_GateMean", semantic_token_feedback_gate_value, step)
                writer.add_scalar("Step/SemanticTokenFeedback_Entropy", semantic_token_feedback_entropy_value, step)
                writer.add_scalar("Train/Step/SemanticTokenFeedback_GateMean", semantic_token_feedback_gate_value, step)
                writer.add_scalar("Train/Step/SemanticTokenFeedback_Entropy", semantic_token_feedback_entropy_value, step)
            writer.add_scalar("Train/Step/Loss_Total", batch_loss.item(), step)
            writer.add_scalar("Train/Step/Loss_Reasoning", r_loss_value, step)
            writer.add_scalar("Train/Step/Loss_Healing", h_loss_value, step)
            writer.add_scalar("Train/Step/Loss_PosteriorAnswer", posterior_answer_loss_value, step)
            writer.add_scalar("Train/Step/Loss_AuxStepAnswer", aux_step_answer_loss_value, step)
            writer.add_scalar("Train/Step/Loss_DepthConsistency", depth_consistency_loss_value, step)
            writer.add_scalar("Train/Step/Loss_LatentShortcutConsistency", latent_shortcut_consistency_loss_value, step)
            writer.add_scalar("Train/Step/Loss_FinalReadoutAnswer", final_readout_loss_value, step)
            writer.add_scalar("Train/Step/Loss_TrajectoryAnchor", trajectory_anchor_loss_value, step)
            writer.add_scalar("Train/Step/Loss_CorrectionFeedback", correction_feedback_loss_value, step)
            writer.add_scalar("Train/Step/Loss_OperationSupervision", operation_supervision_loss_value, step)
            writer.add_scalar("Train/Step/Loss_PriorAnswer", prior_answer_loss_value, step)
            writer.add_scalar("Train/Step/Loss_PriorPosteriorLogitDistill", prior_posterior_logit_distill_loss_value, step)
            if prior_acc_value is not None:
                writer.add_scalar("Train/Step/Accuracy_Reasoning_PriorNoPosterior", prior_acc_value, step)
            writer.add_scalar("Train/Step/Loss_StochasticPosteriorKL", stochastic_posterior_kl_loss_value, step)
            writer.add_scalar("Train/Step/Loss_GRAM_LPRM", gram_lprm_loss_value, step)
            writer.add_scalar("Train/Step/Loss_GRAM_OracleCE", gram_oracle_ce_loss_value, step)
            if gram_lprm_selected_acc_value is not None:
                writer.add_scalar("Train/Step/GRAM_LPRM_SelectedAccuracy", gram_lprm_selected_acc_value, step)
                writer.add_scalar("Train/Step/GRAM_LPRM_OracleAccuracy", gram_lprm_oracle_acc_value, step)
                writer.add_scalar("Train/Step/GRAM_LPRM_TargetSpread", gram_lprm_target_spread_value, step)
            if gram_oracle_ce_selected_acc_value is not None:
                writer.add_scalar("Train/Step/GRAM_OracleCE_SelectedAccuracy", gram_oracle_ce_selected_acc_value, step)
                writer.add_scalar("Train/Step/GRAM_OracleCE_OracleAccuracy", gram_oracle_ce_oracle_acc_value, step)
                writer.add_scalar("Train/Step/GRAM_OracleCE_SelectedTrueProb", gram_oracle_ce_true_prob_value, step)
            writer.add_scalar("Train/Step/Loss_LatticeCandidate", lattice_candidate_loss_value, step)
            writer.add_scalar("Train/Step/Grad_Norm", float(grad_norm), step)
            writer.add_scalar("Train/Step/N_Steps", n_steps, step)
            writer.add_scalar("Train/Step/StateSupervisionWeight", state_supervision_weight, step)
            step_context = {"phase": "train", "granularity": "step"}
            track_aim_scalar(aim_run, batch_loss.item(), name="loss_total", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, r_loss_value, name="loss_reasoning", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, h_loss_value, name="loss_healing", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, posterior_answer_loss_value, name="loss_posterior_answer", step=step, epoch=epoch, context=step_context)
            if state_norm_mean is not None:
                track_aim_scalar(aim_run, state_norm_mean, name="state_norm_mean", step=step, epoch=epoch, context=step_context)
                track_aim_scalar(aim_run, transition_norm_mean, name="transition_norm_mean", step=step, epoch=epoch, context=step_context)
                track_aim_scalar(aim_run, state_cosine_mean, name="state_cosine_mean", step=step, epoch=epoch, context=step_context)
            if working_register_norm_value is not None:
                track_aim_scalar(aim_run, working_register_norm_value, name="working_register_norm", step=step, epoch=epoch, context=step_context)
                track_aim_scalar(aim_run, working_register_gate_value, name="working_register_gate_mean", step=step, epoch=epoch, context=step_context)
                if working_register_role_cosine_value is not None:
                    track_aim_scalar(aim_run, working_register_role_cosine_value, name="working_register_role_cosine", step=step, epoch=epoch, context=step_context)
            if semantic_token_feedback_gate_value is not None:
                track_aim_scalar(
                    aim_run,
                    semantic_token_feedback_gate_value,
                    name="semantic_token_feedback_gate_mean",
                    step=step,
                    epoch=epoch,
                    context=step_context,
                )
                track_aim_scalar(
                    aim_run,
                    semantic_token_feedback_entropy_value,
                    name="semantic_token_feedback_entropy",
                    step=step,
                    epoch=epoch,
                    context=step_context,
                )
            if h_acc_value is not None:
                track_aim_scalar(
                    aim_run,
                    h_acc_value,
                    name="accuracy_healing_target_tokens",
                    step=step,
                    epoch=epoch,
                    context=step_context,
                )
            track_aim_scalar(aim_run, aux_step_answer_loss_value, name="loss_aux_step_answer", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, depth_consistency_loss_value, name="loss_depth_consistency", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(
                aim_run,
                latent_shortcut_consistency_loss_value,
                name="loss_latent_shortcut_consistency",
                step=step,
                epoch=epoch,
                context=step_context,
            )
            track_aim_scalar(aim_run, final_readout_loss_value, name="loss_final_readout_answer", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, trajectory_anchor_loss_value, name="loss_trajectory_anchor", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, correction_feedback_loss_value, name="loss_correction_feedback", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, operation_supervision_loss_value, name="loss_operation_supervision", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, prior_answer_loss_value, name="loss_prior_answer", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(
                aim_run,
                prior_posterior_logit_distill_loss_value,
                name="loss_prior_posterior_logit_distill",
                step=step,
                epoch=epoch,
                context=step_context,
            )
            if prior_acc_value is not None:
                track_aim_scalar(
                    aim_run,
                    prior_acc_value,
                    name="accuracy_reasoning_prior_no_posterior",
                    step=step,
                    epoch=epoch,
                    context=step_context,
                )
            track_aim_scalar(aim_run, stochastic_posterior_kl_loss_value, name="loss_stochastic_posterior_kl", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, gram_lprm_loss_value, name="loss_gram_lprm", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, gram_oracle_ce_loss_value, name="loss_gram_oracle_ce", step=step, epoch=epoch, context=step_context)
            if gram_lprm_selected_acc_value is not None:
                track_aim_scalar(aim_run, gram_lprm_selected_acc_value, name="gram_lprm_selected_accuracy", step=step, epoch=epoch, context=step_context)
                track_aim_scalar(aim_run, gram_lprm_oracle_acc_value, name="gram_lprm_oracle_accuracy", step=step, epoch=epoch, context=step_context)
                track_aim_scalar(aim_run, gram_lprm_target_spread_value, name="gram_lprm_target_spread", step=step, epoch=epoch, context=step_context)
            if gram_oracle_ce_selected_acc_value is not None:
                track_aim_scalar(aim_run, gram_oracle_ce_selected_acc_value, name="gram_oracle_ce_selected_accuracy", step=step, epoch=epoch, context=step_context)
                track_aim_scalar(aim_run, gram_oracle_ce_oracle_acc_value, name="gram_oracle_ce_oracle_accuracy", step=step, epoch=epoch, context=step_context)
                track_aim_scalar(aim_run, gram_oracle_ce_true_prob_value, name="gram_oracle_ce_selected_true_prob", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, lattice_candidate_loss_value, name="loss_lattice_candidate", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, float(grad_norm), name="grad_norm", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, n_steps, name="n_steps", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(
                aim_run,
                state_supervision_weight,
                name="state_supervision_weight",
                step=step,
                epoch=epoch,
                context=step_context,
            )
            if lattice_true_alive_value is not None:
                track_aim_scalar(aim_run, lattice_true_alive_value, name="lattice_true_alive_prob", step=step, epoch=epoch, context=step_context)
                track_aim_scalar(aim_run, lattice_false_alive_value, name="lattice_false_alive_prob", step=step, epoch=epoch, context=step_context)
            if readout_gate_value is not None:
                track_aim_scalar(aim_run, readout_gate_value, name="readout_gate_attention_weight", step=step, epoch=epoch, context=step_context)
            if readout_entropy_value is not None:
                track_aim_scalar(aim_run, readout_entropy_value, name="readout_attention_entropy", step=step, epoch=epoch, context=step_context)
            if correction_gate_value is not None:
                track_aim_scalar(aim_run, correction_gate_value, name="correction_feedback_gate", step=step, epoch=epoch, context=step_context)
            if stochastic_mu_norm_value is not None:
                track_aim_scalar(aim_run, stochastic_mu_norm_value, name="stochastic_high_level_mu_norm", step=step, epoch=epoch, context=step_context)
                track_aim_scalar(aim_run, stochastic_std_mean_value, name="stochastic_high_level_std_mean", step=step, epoch=epoch, context=step_context)
                track_aim_scalar(aim_run, stochastic_noise_norm_value, name="stochastic_high_level_noise_norm", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, batch_loss.item(), name="train_loss_total", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, r_loss_value, name="train_loss_reasoning", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, h_loss_value, name="train_loss_healing", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, posterior_answer_loss_value, name="train_loss_posterior_answer", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, aux_step_answer_loss_value, name="train_loss_aux_step_answer", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, depth_consistency_loss_value, name="train_loss_depth_consistency", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(
                aim_run,
                latent_shortcut_consistency_loss_value,
                name="train_loss_latent_shortcut_consistency",
                step=step,
                epoch=epoch,
                context=step_context,
            )
            track_aim_scalar(aim_run, final_readout_loss_value, name="train_loss_final_readout_answer", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, trajectory_anchor_loss_value, name="train_loss_trajectory_anchor", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, correction_feedback_loss_value, name="train_loss_correction_feedback", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, operation_supervision_loss_value, name="train_loss_operation_supervision", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, prior_answer_loss_value, name="train_loss_prior_answer", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(
                aim_run,
                prior_posterior_logit_distill_loss_value,
                name="train_loss_prior_posterior_logit_distill",
                step=step,
                epoch=epoch,
                context=step_context,
            )
            track_aim_scalar(aim_run, stochastic_posterior_kl_loss_value, name="train_loss_stochastic_posterior_kl", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, gram_lprm_loss_value, name="train_loss_gram_lprm", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, gram_oracle_ce_loss_value, name="train_loss_gram_oracle_ce", step=step, epoch=epoch, context=step_context)
            if gram_lprm_selected_acc_value is not None:
                track_aim_scalar(aim_run, gram_lprm_selected_acc_value, name="train_gram_lprm_selected_accuracy", step=step, epoch=epoch, context=step_context)
                track_aim_scalar(aim_run, gram_lprm_oracle_acc_value, name="train_gram_lprm_oracle_accuracy", step=step, epoch=epoch, context=step_context)
                track_aim_scalar(aim_run, gram_lprm_target_spread_value, name="train_gram_lprm_target_spread", step=step, epoch=epoch, context=step_context)
            if gram_oracle_ce_selected_acc_value is not None:
                track_aim_scalar(aim_run, gram_oracle_ce_selected_acc_value, name="train_gram_oracle_ce_selected_accuracy", step=step, epoch=epoch, context=step_context)
                track_aim_scalar(aim_run, gram_oracle_ce_oracle_acc_value, name="train_gram_oracle_ce_oracle_accuracy", step=step, epoch=epoch, context=step_context)
                track_aim_scalar(aim_run, gram_oracle_ce_true_prob_value, name="train_gram_oracle_ce_selected_true_prob", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, semantic_lm_alignment_loss_value, name="train_loss_semantic_lm_alignment", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, semantic_step_alignment_loss_value, name="train_loss_semantic_step_alignment", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, lattice_candidate_loss_value, name="train_loss_lattice_candidate", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, float(grad_norm), name="train_grad_norm", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(aim_run, n_steps, name="train_n_steps", step=step, epoch=epoch, context=step_context)
            track_aim_scalar(
                aim_run,
                state_supervision_weight,
                name="train_state_supervision_weight",
                step=step,
                epoch=epoch,
                context=step_context,
            )
            if readout_gate_value is not None:
                track_aim_scalar(aim_run, readout_gate_value, name="train_readout_gate_attention_weight", step=step, epoch=epoch, context=step_context)
            if readout_entropy_value is not None:
                track_aim_scalar(aim_run, readout_entropy_value, name="train_readout_attention_entropy", step=step, epoch=epoch, context=step_context)
            if correction_gate_value is not None:
                track_aim_scalar(aim_run, correction_gate_value, name="train_correction_feedback_gate", step=step, epoch=epoch, context=step_context)
            if stochastic_mu_norm_value is not None:
                track_aim_scalar(aim_run, stochastic_mu_norm_value, name="train_stochastic_high_level_mu_norm", step=step, epoch=epoch, context=step_context)
                track_aim_scalar(aim_run, stochastic_std_mean_value, name="train_stochastic_high_level_std_mean", step=step, epoch=epoch, context=step_context)
                track_aim_scalar(aim_run, stochastic_noise_norm_value, name="train_stochastic_high_level_noise_norm", step=step, epoch=epoch, context=step_context)
            if semantic_token_feedback_gate_value is not None:
                track_aim_scalar(
                    aim_run,
                    semantic_token_feedback_gate_value,
                    name="train_semantic_token_feedback_gate_mean",
                    step=step,
                    epoch=epoch,
                    context=step_context,
                )
                track_aim_scalar(
                    aim_run,
                    semantic_token_feedback_entropy_value,
                    name="train_semantic_token_feedback_entropy",
                    step=step,
                    epoch=epoch,
                    context=step_context,
                )

    avg_loss = total_loss / len(loader.dataset)
    avg_r_loss = total_r_loss / len(loader.dataset)
    avg_h_loss = total_h_loss / len(loader.dataset)
    avg_posterior_answer_loss = total_posterior_answer_loss / len(loader.dataset)
    avg_aux_step_answer_loss = total_aux_step_answer_loss / len(loader.dataset)
    avg_depth_consistency_loss = total_depth_consistency_loss / len(loader.dataset)
    avg_latent_shortcut_consistency_loss = total_latent_shortcut_consistency_loss / len(loader.dataset)
    avg_final_readout_loss = total_final_readout_loss / len(loader.dataset)
    avg_trajectory_anchor_loss = total_trajectory_anchor_loss / len(loader.dataset)
    avg_correction_feedback_loss = total_correction_feedback_loss / len(loader.dataset)
    avg_operation_supervision_loss = total_operation_supervision_loss / len(loader.dataset)
    avg_prior_answer_loss = total_prior_answer_loss / len(loader.dataset)
    avg_prior_posterior_logit_distill_loss = total_prior_posterior_logit_distill_loss / len(loader.dataset)
    avg_stochastic_posterior_kl_loss = total_stochastic_posterior_kl_loss / len(loader.dataset)
    avg_gram_lprm_loss = total_gram_lprm_loss / len(loader.dataset)
    avg_gram_lprm_selected_acc = total_gram_lprm_selected_acc / n_gram_lprm_multi if n_gram_lprm_multi else None
    avg_gram_lprm_oracle_acc = total_gram_lprm_oracle_acc / n_gram_lprm_multi if n_gram_lprm_multi else None
    avg_gram_lprm_target_spread = total_gram_lprm_target_spread / n_gram_lprm_multi if n_gram_lprm_multi else None
    avg_gram_oracle_ce_loss = total_gram_oracle_ce_loss / len(loader.dataset)
    avg_gram_oracle_ce_selected_acc = (
        total_gram_oracle_ce_selected_acc / n_gram_oracle_ce_multi if n_gram_oracle_ce_multi else None
    )
    avg_gram_oracle_ce_oracle_acc = (
        total_gram_oracle_ce_oracle_acc / n_gram_oracle_ce_multi if n_gram_oracle_ce_multi else None
    )
    avg_gram_oracle_ce_true_prob = (
        total_gram_oracle_ce_true_prob / n_gram_oracle_ce_multi if n_gram_oracle_ce_multi else None
    )
    avg_lattice_candidate_loss = total_lattice_candidate_loss / len(loader.dataset)
    avg_semantic_lm_alignment_loss = total_semantic_lm_alignment_loss / len(loader.dataset)
    avg_semantic_step_alignment_loss = total_semantic_step_alignment_loss / len(loader.dataset)
    avg_readout_gate = total_readout_gate / n_readout_gate if n_readout_gate else None
    avg_correction_gate = total_correction_gate / n_correction_gate if n_correction_gate else None
    avg_readout_entropy = total_readout_entropy / n_readout_entropy if n_readout_entropy else None
    avg_lattice_true_alive = total_lattice_true_alive / n_lattice_metrics if n_lattice_metrics else None
    avg_lattice_false_alive = total_lattice_false_alive / n_lattice_metrics if n_lattice_metrics else None
    avg_stochastic_mu_norm = total_stochastic_mu_norm / n_stochastic_guidance if n_stochastic_guidance else None
    avg_stochastic_std_mean = total_stochastic_std_mean / n_stochastic_guidance if n_stochastic_guidance else None
    avg_stochastic_noise_norm = total_stochastic_noise_norm / n_stochastic_guidance if n_stochastic_guidance else None
    avg_working_register_norm = total_working_register_norm / n_working_register if n_working_register else None
    avg_working_register_gate = total_working_register_gate / n_working_register if n_working_register else None
    avg_working_register_role_cosine = total_working_register_role_cosine / n_working_register if n_working_register else None
    avg_semantic_token_feedback_gate = (
        total_semantic_token_feedback_gate / n_semantic_token_feedback if n_semantic_token_feedback else None
    )
    avg_semantic_token_feedback_entropy = (
        total_semantic_token_feedback_entropy / n_semantic_token_feedback if n_semantic_token_feedback else None
    )
    avg_acc = total_acc / n_reasoning if n_reasoning else 0.0
    avg_prior_acc = total_prior_acc / n_prior_reasoning if n_prior_reasoning else 0.0
    avg_healing_acc = total_healing_correct / total_healing_tokens if total_healing_tokens else 0.0
    avg_state_norm = total_state_norm / n_norms if n_norms else None
    avg_transition_norm = total_transition_norm / n_norms if n_norms else None
    avg_state_cosine = total_state_cosine / n_norms if n_norms else None

    writer.add_scalar("Epoch/Loss_Total", avg_loss, epoch)
    writer.add_scalar("Epoch/Loss_Reasoning", avg_r_loss, epoch)
    writer.add_scalar("Epoch/Loss_Healing", avg_h_loss, epoch)
    writer.add_scalar("Epoch/Loss_PosteriorAnswer", avg_posterior_answer_loss, epoch)
    writer.add_scalar("Epoch/Loss_AuxStepAnswer", avg_aux_step_answer_loss, epoch)
    writer.add_scalar("Epoch/Loss_DepthConsistency", avg_depth_consistency_loss, epoch)
    writer.add_scalar("Epoch/Loss_LatentShortcutConsistency", avg_latent_shortcut_consistency_loss, epoch)
    writer.add_scalar("Epoch/Loss_FinalReadoutAnswer", avg_final_readout_loss, epoch)
    writer.add_scalar("Epoch/Loss_TrajectoryAnchor", avg_trajectory_anchor_loss, epoch)
    writer.add_scalar("Epoch/Loss_CorrectionFeedback", avg_correction_feedback_loss, epoch)
    writer.add_scalar("Epoch/Loss_OperationSupervision", avg_operation_supervision_loss, epoch)
    writer.add_scalar("Epoch/Loss_PriorAnswer", avg_prior_answer_loss, epoch)
    writer.add_scalar("Epoch/Loss_PriorPosteriorLogitDistill", avg_prior_posterior_logit_distill_loss, epoch)
    writer.add_scalar("Epoch/Loss_StochasticPosteriorKL", avg_stochastic_posterior_kl_loss, epoch)
    writer.add_scalar("Epoch/Loss_GRAM_LPRM", avg_gram_lprm_loss, epoch)
    writer.add_scalar("Epoch/Loss_GRAM_OracleCE", avg_gram_oracle_ce_loss, epoch)
    if avg_gram_lprm_selected_acc is not None:
        writer.add_scalar("Epoch/GRAM_LPRM_SelectedAccuracy", avg_gram_lprm_selected_acc, epoch)
        writer.add_scalar("Epoch/GRAM_LPRM_OracleAccuracy", avg_gram_lprm_oracle_acc, epoch)
        writer.add_scalar("Epoch/GRAM_LPRM_TargetSpread", avg_gram_lprm_target_spread, epoch)
        writer.add_scalar("Train/Epoch/GRAM_LPRM_SelectedAccuracy", avg_gram_lprm_selected_acc, epoch)
        writer.add_scalar("Train/Epoch/GRAM_LPRM_OracleAccuracy", avg_gram_lprm_oracle_acc, epoch)
        writer.add_scalar("Train/Epoch/GRAM_LPRM_TargetSpread", avg_gram_lprm_target_spread, epoch)
    if avg_gram_oracle_ce_selected_acc is not None:
        writer.add_scalar("Epoch/GRAM_OracleCE_SelectedAccuracy", avg_gram_oracle_ce_selected_acc, epoch)
        writer.add_scalar("Epoch/GRAM_OracleCE_OracleAccuracy", avg_gram_oracle_ce_oracle_acc, epoch)
        writer.add_scalar("Epoch/GRAM_OracleCE_SelectedTrueProb", avg_gram_oracle_ce_true_prob, epoch)
        writer.add_scalar("Train/Epoch/GRAM_OracleCE_SelectedAccuracy", avg_gram_oracle_ce_selected_acc, epoch)
        writer.add_scalar("Train/Epoch/GRAM_OracleCE_OracleAccuracy", avg_gram_oracle_ce_oracle_acc, epoch)
        writer.add_scalar("Train/Epoch/GRAM_OracleCE_SelectedTrueProb", avg_gram_oracle_ce_true_prob, epoch)
    writer.add_scalar("Epoch/Loss_SemanticLMAlignment", avg_semantic_lm_alignment_loss, epoch)
    writer.add_scalar("Epoch/Loss_SemanticStepAlignment", avg_semantic_step_alignment_loss, epoch)
    writer.add_scalar("Epoch/Loss_LatticeCandidate", avg_lattice_candidate_loss, epoch)
    writer.add_scalar("Epoch/Accuracy_Reasoning", avg_acc, epoch)
    writer.add_scalar("Epoch/Accuracy_Reasoning_PriorNoPosterior", avg_prior_acc, epoch)
    writer.add_scalar("Epoch/Accuracy_HealingTargetTokens", avg_healing_acc, epoch)
    writer.add_scalar("Epoch/Learning_Rate", optimizer.param_groups[0]["lr"], epoch)
    writer.add_scalar("Epoch/StateSupervisionWeight", state_supervision_weight, epoch)
    if avg_state_norm is not None:
        writer.add_scalar("Epoch/StateNorm_Mean", avg_state_norm, epoch)
        writer.add_scalar("Epoch/TransitionNorm_Mean", avg_transition_norm, epoch)
        writer.add_scalar("Epoch/StateCosine_Mean", avg_state_cosine, epoch)
        writer.add_scalar("Train/Epoch/StateNorm_Mean", avg_state_norm, epoch)
        writer.add_scalar("Train/Epoch/TransitionNorm_Mean", avg_transition_norm, epoch)
        writer.add_scalar("Train/Epoch/StateCosine_Mean", avg_state_cosine, epoch)
    if avg_readout_gate is not None:
        writer.add_scalar("Epoch/Readout_Gate_AttentionWeight", avg_readout_gate, epoch)
        writer.add_scalar("Train/Epoch/Readout_Gate_AttentionWeight", avg_readout_gate, epoch)
    if avg_readout_entropy is not None:
        writer.add_scalar("Epoch/Readout_Attention_Entropy", avg_readout_entropy, epoch)
        writer.add_scalar("Train/Epoch/Readout_Attention_Entropy", avg_readout_entropy, epoch)
    if avg_lattice_true_alive is not None:
        writer.add_scalar("Epoch/Lattice_TrueAliveProb", avg_lattice_true_alive, epoch)
        writer.add_scalar("Epoch/Lattice_FalseAliveProb", avg_lattice_false_alive, epoch)
        writer.add_scalar("Train/Epoch/Lattice_TrueAliveProb", avg_lattice_true_alive, epoch)
        writer.add_scalar("Train/Epoch/Lattice_FalseAliveProb", avg_lattice_false_alive, epoch)
    if avg_correction_gate is not None:
        writer.add_scalar("Epoch/Correction_Feedback_Gate", avg_correction_gate, epoch)
        writer.add_scalar("Train/Epoch/Correction_Feedback_Gate", avg_correction_gate, epoch)
    if avg_stochastic_mu_norm is not None:
        writer.add_scalar("Epoch/StochasticHighLevel_MuNorm", avg_stochastic_mu_norm, epoch)
        writer.add_scalar("Epoch/StochasticHighLevel_StdMean", avg_stochastic_std_mean, epoch)
        writer.add_scalar("Epoch/StochasticHighLevel_NoiseNorm", avg_stochastic_noise_norm, epoch)
        writer.add_scalar("Train/Epoch/StochasticHighLevel_MuNorm", avg_stochastic_mu_norm, epoch)
        writer.add_scalar("Train/Epoch/StochasticHighLevel_StdMean", avg_stochastic_std_mean, epoch)
        writer.add_scalar("Train/Epoch/StochasticHighLevel_NoiseNorm", avg_stochastic_noise_norm, epoch)
    if avg_working_register_norm is not None:
        writer.add_scalar("Epoch/WorkingRegister_Norm", avg_working_register_norm, epoch)
        writer.add_scalar("Epoch/WorkingRegister_GateMean", avg_working_register_gate, epoch)
        writer.add_scalar("Train/Epoch/WorkingRegister_Norm", avg_working_register_norm, epoch)
        writer.add_scalar("Train/Epoch/WorkingRegister_GateMean", avg_working_register_gate, epoch)
        if avg_working_register_role_cosine is not None:
            writer.add_scalar("Epoch/WorkingRegister_RoleCosine", avg_working_register_role_cosine, epoch)
            writer.add_scalar("Train/Epoch/WorkingRegister_RoleCosine", avg_working_register_role_cosine, epoch)
    if avg_semantic_token_feedback_gate is not None:
        writer.add_scalar("Epoch/SemanticTokenFeedback_GateMean", avg_semantic_token_feedback_gate, epoch)
        writer.add_scalar("Epoch/SemanticTokenFeedback_Entropy", avg_semantic_token_feedback_entropy, epoch)
        writer.add_scalar("Train/Epoch/SemanticTokenFeedback_GateMean", avg_semantic_token_feedback_gate, epoch)
        writer.add_scalar("Train/Epoch/SemanticTokenFeedback_Entropy", avg_semantic_token_feedback_entropy, epoch)
    writer.add_scalar("Train/Epoch/Loss_Total", avg_loss, epoch)
    writer.add_scalar("Train/Epoch/Loss_Reasoning", avg_r_loss, epoch)
    writer.add_scalar("Train/Epoch/Loss_Healing", avg_h_loss, epoch)
    writer.add_scalar("Train/Epoch/Loss_PosteriorAnswer", avg_posterior_answer_loss, epoch)
    writer.add_scalar("Train/Epoch/Loss_AuxStepAnswer", avg_aux_step_answer_loss, epoch)
    writer.add_scalar("Train/Epoch/Loss_DepthConsistency", avg_depth_consistency_loss, epoch)
    writer.add_scalar("Train/Epoch/Loss_LatentShortcutConsistency", avg_latent_shortcut_consistency_loss, epoch)
    writer.add_scalar("Train/Epoch/Loss_FinalReadoutAnswer", avg_final_readout_loss, epoch)
    writer.add_scalar("Train/Epoch/Loss_TrajectoryAnchor", avg_trajectory_anchor_loss, epoch)
    writer.add_scalar("Train/Epoch/Loss_CorrectionFeedback", avg_correction_feedback_loss, epoch)
    writer.add_scalar("Train/Epoch/Loss_OperationSupervision", avg_operation_supervision_loss, epoch)
    writer.add_scalar("Train/Epoch/Loss_PriorAnswer", avg_prior_answer_loss, epoch)
    writer.add_scalar("Train/Epoch/Loss_PriorPosteriorLogitDistill", avg_prior_posterior_logit_distill_loss, epoch)
    writer.add_scalar("Train/Epoch/Loss_StochasticPosteriorKL", avg_stochastic_posterior_kl_loss, epoch)
    writer.add_scalar("Train/Epoch/Loss_GRAM_LPRM", avg_gram_lprm_loss, epoch)
    writer.add_scalar("Train/Epoch/Loss_GRAM_OracleCE", avg_gram_oracle_ce_loss, epoch)
    if avg_gram_lprm_selected_acc is not None:
        writer.add_scalar("Train/Epoch/GRAM_LPRM_SelectedAccuracy", avg_gram_lprm_selected_acc, epoch)
        writer.add_scalar("Train/Epoch/GRAM_LPRM_OracleAccuracy", avg_gram_lprm_oracle_acc, epoch)
        writer.add_scalar("Train/Epoch/GRAM_LPRM_TargetSpread", avg_gram_lprm_target_spread, epoch)
    if avg_gram_oracle_ce_selected_acc is not None:
        writer.add_scalar("Train/Epoch/GRAM_OracleCE_SelectedAccuracy", avg_gram_oracle_ce_selected_acc, epoch)
        writer.add_scalar("Train/Epoch/GRAM_OracleCE_OracleAccuracy", avg_gram_oracle_ce_oracle_acc, epoch)
        writer.add_scalar("Train/Epoch/GRAM_OracleCE_SelectedTrueProb", avg_gram_oracle_ce_true_prob, epoch)
    writer.add_scalar("Train/Epoch/Loss_SemanticLMAlignment", avg_semantic_lm_alignment_loss, epoch)
    writer.add_scalar("Train/Epoch/Loss_SemanticStepAlignment", avg_semantic_step_alignment_loss, epoch)
    writer.add_scalar("Train/Epoch/Loss_LatticeCandidate", avg_lattice_candidate_loss, epoch)
    writer.add_scalar("Train/Epoch/Accuracy_Reasoning", avg_acc, epoch)
    writer.add_scalar("Train/Epoch/Accuracy_Reasoning_PriorNoPosterior", avg_prior_acc, epoch)
    writer.add_scalar("Train/Epoch/Accuracy_HealingTargetTokens", avg_healing_acc, epoch)
    writer.add_scalar("Train/Epoch/Learning_Rate", optimizer.param_groups[0]["lr"], epoch)
    writer.add_scalar("Train/Epoch/StateSupervisionWeight", state_supervision_weight, epoch)
    writer.flush()
    epoch_context = {"phase": "train", "granularity": "epoch"}
    track_aim_scalar(aim_run, avg_loss, name="loss_total", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_r_loss, name="loss_reasoning", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_h_loss, name="loss_healing", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_posterior_answer_loss, name="loss_posterior_answer", epoch=epoch, context=epoch_context)
    if avg_state_norm is not None:
        track_aim_scalar(aim_run, avg_state_norm, name="epoch_state_norm_mean", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_transition_norm, name="epoch_transition_norm_mean", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_state_cosine, name="epoch_state_cosine_mean", epoch=epoch, context=epoch_context)
    if avg_working_register_norm is not None:
        track_aim_scalar(aim_run, avg_working_register_norm, name="epoch_working_register_norm", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_working_register_gate, name="epoch_working_register_gate_mean", epoch=epoch, context=epoch_context)
        if avg_working_register_role_cosine is not None:
            track_aim_scalar(aim_run, avg_working_register_role_cosine, name="epoch_working_register_role_cosine", epoch=epoch, context=epoch_context)
    if avg_semantic_token_feedback_gate is not None:
        track_aim_scalar(
            aim_run,
            avg_semantic_token_feedback_gate,
            name="epoch_semantic_token_feedback_gate_mean",
            epoch=epoch,
            context=epoch_context,
        )
        track_aim_scalar(
            aim_run,
            avg_semantic_token_feedback_entropy,
            name="epoch_semantic_token_feedback_entropy",
            epoch=epoch,
            context=epoch_context,
        )
    track_aim_scalar(aim_run, avg_aux_step_answer_loss, name="loss_aux_step_answer", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_depth_consistency_loss, name="loss_depth_consistency", epoch=epoch, context=epoch_context)
    track_aim_scalar(
        aim_run,
        avg_latent_shortcut_consistency_loss,
        name="loss_latent_shortcut_consistency",
        epoch=epoch,
        context=epoch_context,
    )
    track_aim_scalar(aim_run, avg_final_readout_loss, name="loss_final_readout_answer", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_trajectory_anchor_loss, name="loss_trajectory_anchor", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_correction_feedback_loss, name="loss_correction_feedback", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_operation_supervision_loss, name="loss_operation_supervision", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_prior_answer_loss, name="loss_prior_answer", epoch=epoch, context=epoch_context)
    track_aim_scalar(
        aim_run,
        avg_prior_posterior_logit_distill_loss,
        name="loss_prior_posterior_logit_distill",
        epoch=epoch,
        context=epoch_context,
    )
    track_aim_scalar(aim_run, avg_stochastic_posterior_kl_loss, name="loss_stochastic_posterior_kl", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_gram_lprm_loss, name="loss_gram_lprm", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_gram_oracle_ce_loss, name="loss_gram_oracle_ce", epoch=epoch, context=epoch_context)
    if avg_gram_lprm_selected_acc is not None:
        track_aim_scalar(aim_run, avg_gram_lprm_selected_acc, name="gram_lprm_selected_accuracy", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_gram_lprm_oracle_acc, name="gram_lprm_oracle_accuracy", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_gram_lprm_target_spread, name="gram_lprm_target_spread", epoch=epoch, context=epoch_context)
    if avg_gram_oracle_ce_selected_acc is not None:
        track_aim_scalar(aim_run, avg_gram_oracle_ce_selected_acc, name="gram_oracle_ce_selected_accuracy", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_gram_oracle_ce_oracle_acc, name="gram_oracle_ce_oracle_accuracy", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_gram_oracle_ce_true_prob, name="gram_oracle_ce_selected_true_prob", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_semantic_lm_alignment_loss, name="loss_semantic_lm_alignment", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_semantic_step_alignment_loss, name="loss_semantic_step_alignment", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_lattice_candidate_loss, name="loss_lattice_candidate", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_acc, name="accuracy_reasoning", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_prior_acc, name="accuracy_reasoning_prior_no_posterior", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_healing_acc, name="accuracy_healing_target_tokens", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, optimizer.param_groups[0]["lr"], name="learning_rate", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, state_supervision_weight, name="state_supervision_weight", epoch=epoch, context=epoch_context)
    if avg_readout_gate is not None:
        track_aim_scalar(aim_run, avg_readout_gate, name="readout_gate_attention_weight", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_readout_gate, name="train_readout_gate_attention_weight", epoch=epoch, context=epoch_context)
    if avg_readout_entropy is not None:
        track_aim_scalar(aim_run, avg_readout_entropy, name="readout_attention_entropy", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_readout_entropy, name="train_readout_attention_entropy", epoch=epoch, context=epoch_context)
    if avg_lattice_true_alive is not None:
        track_aim_scalar(aim_run, avg_lattice_true_alive, name="lattice_true_alive_prob", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_lattice_false_alive, name="lattice_false_alive_prob", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_lattice_true_alive, name="train_lattice_true_alive_prob", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_lattice_false_alive, name="train_lattice_false_alive_prob", epoch=epoch, context=epoch_context)
    if avg_correction_gate is not None:
        track_aim_scalar(aim_run, avg_correction_gate, name="correction_feedback_gate", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_correction_gate, name="train_correction_feedback_gate", epoch=epoch, context=epoch_context)
    if avg_stochastic_mu_norm is not None:
        track_aim_scalar(aim_run, avg_stochastic_mu_norm, name="stochastic_high_level_mu_norm", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_stochastic_std_mean, name="stochastic_high_level_std_mean", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_stochastic_noise_norm, name="stochastic_high_level_noise_norm", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_stochastic_mu_norm, name="train_stochastic_high_level_mu_norm", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_stochastic_std_mean, name="train_stochastic_high_level_std_mean", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_stochastic_noise_norm, name="train_stochastic_high_level_noise_norm", epoch=epoch, context=epoch_context)
    if avg_semantic_token_feedback_gate is not None:
        track_aim_scalar(
            aim_run,
            avg_semantic_token_feedback_gate,
            name="semantic_token_feedback_gate_mean",
            epoch=epoch,
            context=epoch_context,
        )
        track_aim_scalar(
            aim_run,
            avg_semantic_token_feedback_entropy,
            name="semantic_token_feedback_entropy",
            epoch=epoch,
            context=epoch_context,
        )
        track_aim_scalar(
            aim_run,
            avg_semantic_token_feedback_gate,
            name="train_semantic_token_feedback_gate_mean",
            epoch=epoch,
            context=epoch_context,
        )
        track_aim_scalar(
            aim_run,
            avg_semantic_token_feedback_entropy,
            name="train_semantic_token_feedback_entropy",
            epoch=epoch,
            context=epoch_context,
        )
    track_aim_scalar(aim_run, avg_loss, name="train_loss_total", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_r_loss, name="train_loss_reasoning", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_h_loss, name="train_loss_healing", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_posterior_answer_loss, name="train_loss_posterior_answer", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_aux_step_answer_loss, name="train_loss_aux_step_answer", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_depth_consistency_loss, name="train_loss_depth_consistency", epoch=epoch, context=epoch_context)
    track_aim_scalar(
        aim_run,
        avg_latent_shortcut_consistency_loss,
        name="train_loss_latent_shortcut_consistency",
        epoch=epoch,
        context=epoch_context,
    )
    track_aim_scalar(aim_run, avg_final_readout_loss, name="train_loss_final_readout_answer", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_trajectory_anchor_loss, name="train_loss_trajectory_anchor", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_correction_feedback_loss, name="train_loss_correction_feedback", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_operation_supervision_loss, name="train_loss_operation_supervision", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_prior_answer_loss, name="train_loss_prior_answer", epoch=epoch, context=epoch_context)
    track_aim_scalar(
        aim_run,
        avg_prior_posterior_logit_distill_loss,
        name="train_loss_prior_posterior_logit_distill",
        epoch=epoch,
        context=epoch_context,
    )
    track_aim_scalar(aim_run, avg_stochastic_posterior_kl_loss, name="train_loss_stochastic_posterior_kl", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_gram_lprm_loss, name="train_loss_gram_lprm", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_gram_oracle_ce_loss, name="train_loss_gram_oracle_ce", epoch=epoch, context=epoch_context)
    if avg_gram_lprm_selected_acc is not None:
        track_aim_scalar(aim_run, avg_gram_lprm_selected_acc, name="train_gram_lprm_selected_accuracy", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_gram_lprm_oracle_acc, name="train_gram_lprm_oracle_accuracy", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_gram_lprm_target_spread, name="train_gram_lprm_target_spread", epoch=epoch, context=epoch_context)
    if avg_gram_oracle_ce_selected_acc is not None:
        track_aim_scalar(aim_run, avg_gram_oracle_ce_selected_acc, name="train_gram_oracle_ce_selected_accuracy", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_gram_oracle_ce_oracle_acc, name="train_gram_oracle_ce_oracle_accuracy", epoch=epoch, context=epoch_context)
        track_aim_scalar(aim_run, avg_gram_oracle_ce_true_prob, name="train_gram_oracle_ce_selected_true_prob", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_semantic_lm_alignment_loss, name="train_loss_semantic_lm_alignment", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_semantic_step_alignment_loss, name="train_loss_semantic_step_alignment", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_lattice_candidate_loss, name="train_loss_lattice_candidate", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_acc, name="train_accuracy_reasoning", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, avg_prior_acc, name="train_accuracy_reasoning_prior_no_posterior", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, optimizer.param_groups[0]["lr"], name="train_learning_rate", epoch=epoch, context=epoch_context)
    track_aim_scalar(aim_run, state_supervision_weight, name="train_state_supervision_weight", epoch=epoch, context=epoch_context)

    if avg_gram_lprm_selected_acc is not None:
        print(
            f"GRAM_LPRM epoch {epoch:3d} | "
            f"selected_acc={avg_gram_lprm_selected_acc:.4f} | "
            f"oracle_acc={avg_gram_lprm_oracle_acc:.4f} | "
            f"target_spread={avg_gram_lprm_target_spread:.4f}",
            flush=True,
        )
    if avg_gram_oracle_ce_selected_acc is not None:
        print(
            f"GRAM_OracleCE epoch {epoch:3d} | "
            f"selected_acc={avg_gram_oracle_ce_selected_acc:.4f} | "
            f"oracle_acc={avg_gram_oracle_ce_oracle_acc:.4f} | "
            f"selected_true_prob={avg_gram_oracle_ce_true_prob:.4f}",
            flush=True,
        )

    return (
        avg_loss,
        avg_acc,
        avg_r_loss,
        avg_h_loss,
        avg_posterior_answer_loss,
        avg_aux_step_answer_loss,
        avg_depth_consistency_loss,
        avg_latent_shortcut_consistency_loss,
        avg_final_readout_loss,
        avg_trajectory_anchor_loss,
        avg_correction_feedback_loss,
        avg_operation_supervision_loss,
        avg_prior_answer_loss,
        avg_prior_posterior_logit_distill_loss,
        avg_prior_acc,
        avg_stochastic_posterior_kl_loss,
        avg_gram_lprm_loss,
        avg_gram_oracle_ce_loss,
        avg_semantic_lm_alignment_loss,
        avg_semantic_step_alignment_loss,
        avg_lattice_candidate_loss,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-model-id", type=str, default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--reasoning-count", type=int, default=2800)
    parser.add_argument("--healing-count", type=int, default=1200)
    parser.add_argument("--healing-data-path", type=str, default=None)
    parser.add_argument("--healing-include-glob", type=str, action="append", default=None)
    parser.add_argument("--healing-rows-per-file-cap", type=int, default=2048)
    parser.add_argument("--healing-target-tokens", type=int, default=1)
    parser.add_argument("--source-eval-data-path", type=str, default=None)
    parser.add_argument("--source-eval-include-glob", type=str, action="append", default=None)
    parser.add_argument("--source-eval-count", type=int, default=512)
    parser.add_argument("--source-eval-rows-per-file-cap", type=int, default=2048)
    parser.add_argument("--source-eval-target-tokens", type=int, default=0)
    parser.add_argument("--source-eval-batch-size", type=int, default=16)
    parser.add_argument("--synthetic-schema", choices=("legacy", "generalized"), default="legacy")
    parser.add_argument("--train-depths", type=int, nargs="+", default=[4])
    parser.add_argument(
        "--synthetic-family-mix",
        choices=("chain2_checksum1", "balanced", "checksum2_chain1"),
        default="chain2_checksum1",
    )
    parser.add_argument(
        "--synthetic-sampling-strategy",
        choices=("random", "stratified"),
        default="random",
    )
    parser.add_argument(
        "--train-surface-mode",
        choices=("canonical", "ledger", "prose", "heldout", "mixed", "mixed_all"),
        default="canonical",
        help="Prompt-surface curriculum for generalized synthetic training rows.",
    )
    parser.add_argument(
        "--eval-surface-mode",
        choices=("canonical", "ledger", "prose", "heldout", "mixed", "mixed_all"),
        default="canonical",
        help="Prompt-surface mode for held-out synthetic generalization eval.",
    )
    parser.add_argument(
        "--synthetic-depth-family-pattern",
        type=str,
        nargs="+",
        default=None,
        help="Optional repeated family:depth entries, e.g. chain:4 checksum:8 checksum:8.",
    )
    parser.add_argument("--reasoning-condition-prefix", type=str, default="synth")
    parser.add_argument("--reasoning-weight", type=float, default=2.0)
    parser.add_argument("--healing-weight", type=float, default=0.4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--workspace-pooling", choices=("mean", "last", "attention", "sequence", "none"), default="mean")
    parser.add_argument("--core-impl", choices=("state_transition", "hybrid_state_transition"), default="state_transition")
    parser.add_argument("--core-update", choices=("mlp", "mini_gated_delta"), default="mlp")
    parser.add_argument("--answer-path", choices=("state_head", "lm_head"), default="state_head")
    parser.add_argument(
        "--recurrent-readout-pooling",
        choices=("final", "mean", "attention", "sharp_attention", "hybrid_gate"),
        default="final",
    )
    parser.add_argument("--recurrent-readout-temperature", type=float, default=1.0)
    parser.add_argument("--freeze-qwen", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--save-last-every-epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-best-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-best-generalization-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-trainable-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-steps", type=int, default=4)
    parser.add_argument("--depth-sample-min", type=int, default=0)
    parser.add_argument(
        "--depth-sample-exact-batch-max",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the current reasoning batch's maximum synthetic depth as n_steps. "
            "This is a curriculum/data-contract diagnostic that prevents shallow "
            "rows from being trained with extra padded copy steps."
        ),
    )
    parser.add_argument(
        "--posterior-answer-weight",
        type=float,
        default=1.0,
        help=(
            "Weight for the answer loss computed on the posterior-guided True-GRAM "
            "path. Lower this when prior-only evaluation path training should dominate."
        ),
    )
    parser.add_argument("--aux-step-answer-weight", type=float, default=0.0)
    parser.add_argument(
        "--prior-aux-step-answer-weight",
        type=float,
        default=0.0,
        help=(
            "Apply LM-head step-answer CE to the prior/no-posterior trajectory "
            "used at evaluation time."
        ),
    )
    parser.add_argument("--state-supervision-weight", type=float, default=1.0)
    parser.add_argument("--state-supervision-decay-rate", type=float, default=1.0)
    parser.add_argument("--state-supervision-min-weight", type=float, default=0.0)
    parser.add_argument("--final-readout-answer-weight", type=float, default=0.0)
    parser.add_argument("--depth-consistency-weight", type=float, default=0.0)
    parser.add_argument("--depth-consistency-temperature", type=float, default=1.0)
    parser.add_argument("--consistency-min-steps", type=int, default=0)
    parser.add_argument("--latent-shortcut-consistency-weight", type=float, default=0.0)
    parser.add_argument("--latent-shortcut-consistency-min-step", type=int, default=1)
    parser.add_argument("--trajectory-anchor-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-anchor-min-step", type=int, default=1)
    parser.add_argument("--correction-feedback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--correction-feedback-scale", type=float, default=1.0)
    parser.add_argument("--correction-feedback-gate-init-bias", type=float, default=-1.0)
    parser.add_argument("--correction-feedback-loss-weight", type=float, default=0.0)
    parser.add_argument("--operation-supervision-weight", type=float, default=0.0)
    parser.add_argument(
        "--prior-answer-weight",
        type=float,
        default=0.0,
        help=(
            "Optional answer loss on the prior-only True-GRAM path used at evaluation. "
            "This keeps posterior teacher-forcing from becoming the only trained path."
        ),
    )
    parser.add_argument(
        "--prior-posterior-logit-distill-weight",
        type=float,
        default=0.0,
        help=(
            "Optional KL distillation from posterior-guided answer logits into the "
            "prior-only evaluation path."
        ),
    )
    parser.add_argument("--prior-posterior-distill-temperature", type=float, default=1.0)
    parser.add_argument("--condition-on-operation-ids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--operation-arg-conditioning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--continuous-time", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--working-register-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--working-register-slots", type=int, default=4)
    parser.add_argument("--working-register-update-scale", type=float, default=0.25)
    parser.add_argument("--working-register-feedback-scale", type=float, default=1.0)
    parser.add_argument("--working-register-gate-init-bias", type=float, default=-2.0)
    parser.add_argument("--working-register-summary-mode", choices=("mean", "query_attention", "query_dot"), default="mean")
    parser.add_argument("--working-register-role-conditioning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--working-register-role-anchor-scale", type=float, default=0.0)
    parser.add_argument("--working-register-update-mode", choices=("all", "cyclic"), default="all")
    parser.add_argument("--stochastic-high-level-guidance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stochastic-high-level-scale", type=float, default=0.05)
    parser.add_argument("--stochastic-high-level-min-std", type=float, default=1e-4)
    parser.add_argument("--stochastic-high-level-max-std", type=float, default=0.2)
    parser.add_argument("--stochastic-high-level-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stochastic-posterior-guidance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stochastic-transition-mode", choices=("delta", "true_gram"), default="delta")
    parser.add_argument("--stochastic-posterior-kl-weight", type=float, default=0.0)
    parser.add_argument("--stochastic-posterior-kl-free-bits", type=float, default=0.0)
    parser.add_argument("--stochastic-posterior-kl-warmup-steps", type=int, default=0)
    parser.add_argument("--gram-lprm-weight", type=float, default=0.0)
    parser.add_argument("--gram-lprm-target", choices=("true_prob", "correct"), default="true_prob")
    parser.add_argument("--gram-lprm-train-samples", type=int, default=1)
    parser.add_argument("--gram-lprm-bce-weight", type=float, default=1.0)
    parser.add_argument("--gram-lprm-listwise-weight", type=float, default=1.0)
    parser.add_argument("--gram-lprm-pairwise-weight", type=float, default=0.5)
    parser.add_argument("--gram-lprm-pairwise-margin", type=float, default=0.02)
    parser.add_argument("--gram-lprm-listwise-temperature", type=float, default=1.0)
    parser.add_argument("--gram-lprm-detach-state", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--gram-oracle-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Train the generator through the best sampled prior trajectory. "
            "This targets candidate quality, unlike LPRM which mainly trains the checker."
        ),
    )
    parser.add_argument("--gram-oracle-ce-train-samples", type=int, default=1)
    parser.add_argument(
        "--gram-oracle-ce-selection",
        choices=("true_prob", "correct_then_prob"),
        default="correct_then_prob",
    )
    parser.add_argument("--trajectory-reward-mode", choices=("final", "rich"), default="final")
    parser.add_argument("--semantic-lm-alignment-weight", type=float, default=0.0)
    parser.add_argument(
        "--semantic-step-alignment-weight",
        type=float,
        default=0.0,
        help=(
            "Align intermediate recurrent thought states to Qwen LM-head digit-token "
            "directions. This keeps step supervision on the universal LM-head path "
            "instead of relying only on a separate 10-way state head."
        ),
    )
    parser.add_argument(
        "--semantic-token-feedback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "At each recurrent step, decode the thought state's belief over Qwen "
            "label-token directions and feed the expected token direction back into z_H."
        ),
    )
    parser.add_argument("--semantic-token-feedback-scale", type=float, default=0.0)
    parser.add_argument("--semantic-token-feedback-temperature", type=float, default=1.0)
    parser.add_argument("--semantic-token-feedback-gate-init-bias", type=float, default=-2.0)
    parser.add_argument("--semantic-token-feedback-score-mode", choices=("cosine", "dot"), default="cosine")
    parser.add_argument("--semantic-token-feedback-teacher-forcing", type=float, default=0.0)
    parser.add_argument("--train-qwen-embeddings", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-qwen-lm-head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-qwen-final-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qwen-embedding-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--qwen-lm-head-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--qwen-final-norm-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--trajectory-reward-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--train-only-trajectory-reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lattice-candidate-weight", type=float, default=0.0)
    parser.add_argument("--lattice-candidate-target", choices=("final_answer", "state_labels"), default="final_answer")
    parser.add_argument("--lattice-candidate-positive-weight", type=float, default=8.0)
    parser.add_argument("--lattice-candidate-negative-weight", type=float, default=1.0)
    parser.add_argument("--lattice-candidate-min-step", type=int, default=1)
    parser.add_argument("--lattice-candidate-threshold", type=float, default=0.5)
    parser.add_argument("--lattice-feedback-mode", choices=("none", "soft", "threshold"), default="none")
    parser.add_argument("--lattice-feedback-scale", type=float, default=0.0)
    parser.add_argument("--lattice-feedback-threshold", type=float, default=0.5)
    parser.add_argument("--init-lattice-feedback-from-readout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--transition-scale-init", type=float, default=1.0)
    parser.add_argument("--override-transition-scale", type=float, default=None)
    parser.add_argument("--override-injection-gate-logit", type=float, default=None)
    parser.add_argument("--zero-step-embeddings", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze-step-embeddings", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--step-embedding-std", type=float, default=None)
    parser.add_argument("--state-update-schedule", choices=("nested", "two_stream"), default="nested")
    parser.add_argument("--latent-feedback-passes", type=int, default=1)
    parser.add_argument("--timestamp-run-dir", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--aim-repo", type=str, default=os.environ.get("QTRM_AIM_REPO"))
    parser.add_argument("--aim-experiment", type=str, default="qwen35_hrmtext")
    parser.add_argument("--aim-run-name", type=str, default=None)
    parser.add_argument("--aim-description", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-count", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-seed", type=int, default=10042)
    parser.add_argument("--eval-depths", type=int, nargs="+", default=[4, 6, 8, 10])
    parser.add_argument("--lattice-eval-thresholds", type=float, nargs="+", default=[0.35, 0.5, 0.65, 0.8])
    parser.add_argument("--stochastic-eval-samples", type=int, nargs="+", default=[1])
    parser.add_argument("--stochastic-selection-mode", choices=("mean", "vote", "confidence", "lprm"), default="lprm")
    parser.add_argument("--generalization-early-stop-patience", type=int, default=0)
    parser.add_argument("--generalization-early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataloader-seed", type=int, default=None)
    parser.add_argument("--layerscale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--layerscale-init", type=float, default=1e-5)
    parser.add_argument("--gate-type", choices=("tanh", "sigmoid"), default="tanh")
    parser.add_argument("--gate-bias-init", type=float, default=0.5)
    args = parser.parse_args()
    if args.depth_sample_min < 0:
        raise ValueError("--depth-sample-min must be >= 0")
    if args.healing_target_tokens <= 0:
        raise ValueError("--healing-target-tokens must be positive")
    if args.source_eval_count <= 0:
        raise ValueError("--source-eval-count must be positive")
    if args.source_eval_batch_size <= 0:
        raise ValueError("--source-eval-batch-size must be positive")
    if args.source_eval_target_tokens < 0:
        raise ValueError("--source-eval-target-tokens must be >= 0")
    if args.depth_sample_min > args.n_steps:
        raise ValueError("--depth-sample-min must be <= --n-steps")
    if args.working_register_slots <= 0:
        raise ValueError("--working-register-slots must be positive")
    if args.working_register_update_scale < 0:
        raise ValueError("--working-register-update-scale must be >= 0")
    if args.working_register_feedback_scale < 0:
        raise ValueError("--working-register-feedback-scale must be >= 0")
    if args.working_register_role_anchor_scale < 0:
        raise ValueError("--working-register-role-anchor-scale must be >= 0")
    if args.state_supervision_weight < 0:
        raise ValueError("--state-supervision-weight must be >= 0")
    if args.prior_aux_step_answer_weight < 0:
        raise ValueError("--prior-aux-step-answer-weight must be >= 0")
    if not 0.0 <= args.state_supervision_decay_rate <= 1.0:
        raise ValueError("--state-supervision-decay-rate must be between 0 and 1")
    if args.state_supervision_min_weight < 0:
        raise ValueError("--state-supervision-min-weight must be >= 0")
    if args.state_supervision_min_weight > args.state_supervision_weight:
        raise ValueError("--state-supervision-min-weight must be <= --state-supervision-weight")
    if args.consistency_min_steps < 0:
        raise ValueError("--consistency-min-steps must be >= 0")
    if args.consistency_min_steps >= args.n_steps:
        raise ValueError("--consistency-min-steps must be < --n-steps")
    if args.latent_shortcut_consistency_weight < 0:
        raise ValueError("--latent-shortcut-consistency-weight must be >= 0")
    if args.latent_shortcut_consistency_min_step < 1:
        raise ValueError("--latent-shortcut-consistency-min-step must be >= 1")
    if args.trajectory_anchor_weight < 0:
        raise ValueError("--trajectory-anchor-weight must be >= 0")
    if args.trajectory_anchor_min_step < 1:
        raise ValueError("--trajectory-anchor-min-step must be >= 1")
    if args.correction_feedback_loss_weight < 0:
        raise ValueError("--correction-feedback-loss-weight must be >= 0")
    if args.operation_supervision_weight < 0:
        raise ValueError("--operation-supervision-weight must be >= 0")
    if args.stochastic_posterior_kl_free_bits < 0:
        raise ValueError("--stochastic-posterior-kl-free-bits must be >= 0")
    if args.stochastic_posterior_kl_warmup_steps < 0:
        raise ValueError("--stochastic-posterior-kl-warmup-steps must be >= 0")
    if any(int(value) <= 0 for value in args.stochastic_eval_samples):
        raise ValueError("--stochastic-eval-samples values must be positive")
    if args.stochastic_high_level_scale < 0:
        raise ValueError("--stochastic-high-level-scale must be >= 0")
    if args.stochastic_high_level_min_std < 0:
        raise ValueError("--stochastic-high-level-min-std must be >= 0")
    if args.stochastic_high_level_max_std < args.stochastic_high_level_min_std:
        raise ValueError("--stochastic-high-level-max-std must be >= --stochastic-high-level-min-std")
    if args.stochastic_posterior_kl_weight < 0:
        raise ValueError("--stochastic-posterior-kl-weight must be >= 0")
    if args.gram_lprm_weight < 0:
        raise ValueError("--gram-lprm-weight must be >= 0")
    if args.gram_lprm_train_samples <= 0:
        raise ValueError("--gram-lprm-train-samples must be positive")
    if args.gram_oracle_ce_weight < 0:
        raise ValueError("--gram-oracle-ce-weight must be >= 0")
    if args.gram_oracle_ce_train_samples <= 0:
        raise ValueError("--gram-oracle-ce-train-samples must be positive")
    if args.gram_lprm_bce_weight < 0:
        raise ValueError("--gram-lprm-bce-weight must be >= 0")
    if args.gram_lprm_listwise_weight < 0:
        raise ValueError("--gram-lprm-listwise-weight must be >= 0")
    if args.gram_lprm_pairwise_weight < 0:
        raise ValueError("--gram-lprm-pairwise-weight must be >= 0")
    if args.gram_lprm_pairwise_margin < 0:
        raise ValueError("--gram-lprm-pairwise-margin must be >= 0")
    if args.gram_lprm_listwise_temperature <= 0:
        raise ValueError("--gram-lprm-listwise-temperature must be > 0")
    if args.semantic_lm_alignment_weight < 0:
        raise ValueError("--semantic-lm-alignment-weight must be >= 0")
    if args.semantic_step_alignment_weight < 0:
        raise ValueError("--semantic-step-alignment-weight must be >= 0")
    if args.semantic_token_feedback and args.answer_path != "lm_head":
        raise ValueError("--semantic-token-feedback requires --answer-path lm_head")
    if args.semantic_token_feedback_scale < 0:
        raise ValueError("--semantic-token-feedback-scale must be >= 0")
    if args.semantic_token_feedback_temperature <= 0:
        raise ValueError("--semantic-token-feedback-temperature must be > 0")
    if not 0.0 <= args.semantic_token_feedback_teacher_forcing <= 1.0:
        raise ValueError("--semantic-token-feedback-teacher-forcing must be in [0, 1]")
    if args.qwen_embedding_lr_multiplier <= 0:
        raise ValueError("--qwen-embedding-lr-multiplier must be > 0")
    if args.qwen_lm_head_lr_multiplier <= 0:
        raise ValueError("--qwen-lm-head-lr-multiplier must be > 0")
    if args.qwen_final_norm_lr_multiplier <= 0:
        raise ValueError("--qwen-final-norm-lr-multiplier must be > 0")
    if args.trajectory_reward_lr_multiplier <= 0:
        raise ValueError("--trajectory-reward-lr-multiplier must be > 0")
    if args.train_only_trajectory_reward and args.gram_lprm_weight <= 0:
        raise ValueError("--train-only-trajectory-reward requires --gram-lprm-weight > 0")
    if args.train_only_trajectory_reward and args.gram_lprm_train_samples <= 1:
        raise ValueError("--train-only-trajectory-reward requires --gram-lprm-train-samples > 1")
    if args.lattice_candidate_weight < 0:
        raise ValueError("--lattice-candidate-weight must be >= 0")
    if args.lattice_candidate_positive_weight < 0:
        raise ValueError("--lattice-candidate-positive-weight must be >= 0")
    if args.lattice_candidate_negative_weight < 0:
        raise ValueError("--lattice-candidate-negative-weight must be >= 0")
    if args.lattice_candidate_min_step < 1:
        raise ValueError("--lattice-candidate-min-step must be >= 1")
    if not 0.0 <= args.lattice_candidate_threshold <= 1.0:
        raise ValueError("--lattice-candidate-threshold must be in [0, 1]")
    if args.lattice_feedback_scale < 0:
        raise ValueError("--lattice-feedback-scale must be >= 0")
    if not 0.0 <= args.lattice_feedback_threshold <= 1.0:
        raise ValueError("--lattice-feedback-threshold must be in [0, 1]")
    if args.override_transition_scale is not None and args.override_transition_scale < 0:
        raise ValueError("--override-transition-scale must be >= 0")
    if args.stochastic_posterior_guidance and not args.stochastic_high_level_guidance:
        raise ValueError("--stochastic-posterior-guidance requires --stochastic-high-level-guidance")
    if args.eval_every < 0:
        raise ValueError("--eval-every must be >= 0")
    if args.eval_count <= 0:
        raise ValueError("--eval-count must be positive")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    if any(depth <= 0 for depth in args.eval_depths):
        raise ValueError("--eval-depths must all be positive")
    if any(threshold < 0.0 or threshold > 1.0 for threshold in args.lattice_eval_thresholds):
        raise ValueError("--lattice-eval-thresholds must all be in [0, 1]")
    if args.generalization_early_stop_patience < 0:
        raise ValueError("--generalization-early-stop-patience must be >= 0")
    if args.generalization_early_stop_min_delta < 0:
        raise ValueError("--generalization-early-stop-min-delta must be >= 0")
    if any(depth <= 0 for depth in args.train_depths):
        raise ValueError("--train-depths must all be positive")
    if args.synthetic_schema == "generalized" and max(args.train_depths) > args.n_steps:
        raise ValueError("--train-depths must be <= --n-steps for generalized synthetic schema")
    depth_family_pattern = parse_depth_family_pattern(args.synthetic_depth_family_pattern)

    dataloader_generator = configure_reproducibility(args)

    if args.timestamp_run_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = args.run_name or os.path.basename(os.path.normpath(args.out_dir))
        args.out_dir = os.path.join(args.out_dir, f"{timestamp}_{run_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = build_qwen_state_transition_model(
        args.qwen_model_id,
        freeze_qwen=args.freeze_qwen,
        device=device,
        core_impl=args.core_impl,
        core_update=args.core_update,
        answer_path=args.answer_path,
        workspace_pooling=args.workspace_pooling,
        recurrent_readout_pooling=args.recurrent_readout_pooling,
        recurrent_readout_temperature=args.recurrent_readout_temperature,
        n_steps=args.n_steps,
        transition_scale_init=args.transition_scale_init,
        step_embedding_std=args.step_embedding_std,
        state_update_schedule=args.state_update_schedule,
        latent_feedback_passes=args.latent_feedback_passes,
        correction_feedback=args.correction_feedback,
        correction_feedback_scale=args.correction_feedback_scale,
        correction_feedback_gate_init_bias=args.correction_feedback_gate_init_bias,
        stochastic_high_level_guidance=args.stochastic_high_level_guidance,
        stochastic_high_level_scale=args.stochastic_high_level_scale,
        stochastic_high_level_min_std=args.stochastic_high_level_min_std,
        stochastic_high_level_max_std=args.stochastic_high_level_max_std,
        stochastic_high_level_eval=args.stochastic_high_level_eval,
        stochastic_posterior_guidance=args.stochastic_posterior_guidance,
        stochastic_transition_mode=args.stochastic_transition_mode,
        lattice_feedback_mode=args.lattice_feedback_mode,
        lattice_feedback_scale=args.lattice_feedback_scale,
        lattice_feedback_threshold=args.lattice_feedback_threshold,
        operation_arg_conditioning=args.operation_arg_conditioning,
        continuous_time=args.continuous_time,
        working_register_enabled=args.working_register_enabled,
        working_register_slots=args.working_register_slots,
        working_register_update_scale=args.working_register_update_scale,
        working_register_feedback_scale=args.working_register_feedback_scale,
        working_register_gate_init_bias=args.working_register_gate_init_bias,
        working_register_summary_mode=args.working_register_summary_mode,
        working_register_role_conditioning=args.working_register_role_conditioning,
        working_register_role_anchor_scale=args.working_register_role_anchor_scale,
        working_register_update_mode=args.working_register_update_mode,
        semantic_token_feedback=args.semantic_token_feedback,
        semantic_token_feedback_scale=args.semantic_token_feedback_scale,
        semantic_token_feedback_temperature=args.semantic_token_feedback_temperature,
        semantic_token_feedback_gate_init_bias=args.semantic_token_feedback_gate_init_bias,
        semantic_token_feedback_score_mode=args.semantic_token_feedback_score_mode,
        semantic_token_feedback_teacher_forcing=args.semantic_token_feedback_teacher_forcing,
        trajectory_reward_mode=args.trajectory_reward_mode,
        layerscale=args.layerscale,
        layerscale_init=args.layerscale_init,
        gate_type=args.gate_type,
        gate_bias_init=args.gate_bias_init,
    )

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        load_stats = load_flexible_checkpoint(model, args.resume, device)
        print(f"[resume] load stats: {load_stats}", flush=True)

    override_stats = apply_recurrent_identity_overrides(model, args)
    if any(override_stats.values()):
        print(f"[recurrent-identity-overrides] {override_stats}", flush=True)
    lattice_init_stats = initialize_lattice_feedback_from_readout(
        model,
        enabled=bool(args.init_lattice_feedback_from_readout),
    )
    if lattice_init_stats.get("initialized", 0.0):
        print(f"[lattice-feedback-init] {lattice_init_stats}", flush=True)
    partial_qwen_stats = configure_qwen_partial_training(model, args)
    if partial_qwen_stats.get("enabled", False):
        print(f"[qwen-partial-training] {partial_qwen_stats}", flush=True)
    trajectory_reward_only_stats = configure_trajectory_reward_only_training(
        model,
        enabled=bool(args.train_only_trajectory_reward),
    )
    if trajectory_reward_only_stats.get("enabled", False):
        print(f"[trajectory-reward-only-training] {trajectory_reward_only_stats}", flush=True)

    healing_rows = None
    if args.healing_data_path:
        healing_rows = load_hrm_text_rows_from_path(
            args.healing_data_path,
            count=args.healing_count,
            seed=args.seed,
            include_globs=args.healing_include_glob,
            rows_per_file_cap=args.healing_rows_per_file_cap,
        )
        print(f"Loaded HRM-Text healing rows: {len(healing_rows)} from {args.healing_data_path}", flush=True)

    train_ds = ConcatDataset(
        [
            SyntheticDataset(
                tokenizer,
                count=args.reasoning_count,
                seed=args.seed,
                max_length=args.max_length,
                schema=args.synthetic_schema,
                depths=args.train_depths,
                max_operation_steps=args.n_steps,
                condition_prefix=args.reasoning_condition_prefix,
                family_mix=args.synthetic_family_mix,
                sampling_strategy=args.synthetic_sampling_strategy,
                depth_family_pattern=depth_family_pattern,
                surface_mode=args.train_surface_mode,
            ),
            HRMTextHealingDataset(
                tokenizer,
                rows=healing_rows,
                count=args.healing_count,
                seed=args.seed,
                max_length=args.max_length,
                target_tokens=args.healing_target_tokens,
            ),
        ]
    )
    source_eval_ds = None
    if args.source_eval_data_path:
        source_eval_rows = load_hrm_text_rows_from_path(
            args.source_eval_data_path,
            count=args.source_eval_count,
            seed=args.seed,
            include_globs=args.source_eval_include_glob,
            rows_per_file_cap=args.source_eval_rows_per_file_cap,
        )
        source_eval_target_tokens = (
            args.source_eval_target_tokens if args.source_eval_target_tokens > 0 else args.healing_target_tokens
        )
        source_eval_ds = HRMTextHealingDataset(
            tokenizer,
            rows=source_eval_rows,
            count=len(source_eval_rows),
            seed=args.seed,
            max_length=args.max_length,
            target_tokens=source_eval_target_tokens,
        )
        print(
            f"Loaded verified-source eval rows: {len(source_eval_rows)} from {args.source_eval_data_path}",
            flush=True,
        )
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=dataloader_generator,
    )
    optimizer = build_optimizer(model, args)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "run_info.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"created_at={datetime.now().isoformat(timespec='seconds')}\n")
        handle.write(f"out_dir={args.out_dir}\n")
        for key, value in sorted(vars(args).items()):
            handle.write(f"{key}={value}\n")
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "logs"))
    aim_run = init_aim_run(args)
    best_acc = float("-inf")
    best_epoch = 0
    best_generalization_acc = float("-inf")
    best_generalization_epoch = 0
    best_stochastic_oracle_acc = float("-inf")
    best_stochastic_oracle_epoch = 0
    best_stochastic_oracle_samples = 0
    epochs_since_generalization_improvement = 0

    try:
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            (
                loss,
                acc,
                r_loss,
                h_loss,
                posterior_answer_loss,
                aux_loss,
                consistency_loss,
                latent_shortcut_consistency_loss,
                final_readout_loss,
                trajectory_anchor_loss,
                correction_feedback_loss,
                operation_supervision_loss,
                prior_answer_loss,
                prior_posterior_logit_distill_loss,
                prior_acc,
                stochastic_posterior_kl_loss,
                gram_lprm_loss,
                gram_oracle_ce_loss,
                semantic_lm_alignment_loss,
                semantic_step_alignment_loss,
                lattice_candidate_loss,
            ) = train_one_epoch(
                model,
                loader,
                optimizer,
                device,
                args,
                writer,
                epoch,
                aim_run=aim_run,
            )
            duration = time.time() - start
            print(
                f"Epoch {epoch:3d} | loss={loss:.4f} | r_loss={r_loss:.4f} | "
                f"h_loss={h_loss:.4f} | post={posterior_answer_loss:.4f} | "
                f"aux_step={aux_loss:.4f} | "
                f"depth_cons={consistency_loss:.4f} | latent_shortcut={latent_shortcut_consistency_loss:.4f} | "
                f"final_readout={final_readout_loss:.4f} | "
                f"traj_anchor={trajectory_anchor_loss:.4f} | "
                f"corr_fb={correction_feedback_loss:.4f} | "
                f"op_sup={operation_supervision_loss:.4f} | "
                f"prior={prior_answer_loss:.4f} | prior_distill={prior_posterior_logit_distill_loss:.4f} | "
                f"prior_acc={prior_acc:.4f} | "
                f"stoch_kl={stochastic_posterior_kl_loss:.4f} | "
                f"lprm={gram_lprm_loss:.4f} | oracle_ce={gram_oracle_ce_loss:.4f} | "
                f"semantic_lm={semantic_lm_alignment_loss:.4f} | "
                f"semantic_step={semantic_step_alignment_loss:.4f} | "
                f"lattice={lattice_candidate_loss:.4f} | "
                f"acc={acc:.4f} | time={duration:.1f}s",
                flush=True,
            )
            track_aim_scalar(
                aim_run,
                duration,
                name="epoch_duration_seconds",
                epoch=epoch,
                context={"phase": "train", "granularity": "epoch"},
            )

            if args.save_best_checkpoint and acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                torch.save(
                    checkpoint_state_dict(model, trainable_only=args.save_trainable_only),
                    os.path.join(args.out_dir, "best.pt"),
                )
                with open(os.path.join(args.out_dir, "best_info.txt"), "w", encoding="utf-8") as handle:
                    handle.write(f"best_epoch={best_epoch}\n")
                    handle.write(f"best_accuracy={best_acc:.8f}\n")
                    handle.write(f"best_train_accuracy={best_acc:.8f}\n")
                    handle.write(f"loss={loss:.8f}\n")
                    handle.write(f"reasoning_loss={r_loss:.8f}\n")
                    handle.write(f"healing_loss={h_loss:.8f}\n")
                    handle.write(f"posterior_answer_loss={posterior_answer_loss:.8f}\n")
                    handle.write(f"aux_step_answer_loss={aux_loss:.8f}\n")
                    handle.write(f"depth_consistency_loss={consistency_loss:.8f}\n")
                    handle.write(f"latent_shortcut_consistency_loss={latent_shortcut_consistency_loss:.8f}\n")
                    handle.write(f"final_readout_answer_loss={final_readout_loss:.8f}\n")
                    handle.write(f"trajectory_anchor_loss={trajectory_anchor_loss:.8f}\n")
                    handle.write(f"correction_feedback_loss={correction_feedback_loss:.8f}\n")
                    handle.write(f"operation_supervision_loss={operation_supervision_loss:.8f}\n")
                    handle.write(f"prior_answer_loss={prior_answer_loss:.8f}\n")
                    handle.write(f"prior_posterior_logit_distill_loss={prior_posterior_logit_distill_loss:.8f}\n")
                    handle.write(f"prior_no_posterior_accuracy={prior_acc:.8f}\n")
                    handle.write(f"semantic_lm_alignment_loss={semantic_lm_alignment_loss:.8f}\n")
                    handle.write(f"semantic_step_alignment_loss={semantic_step_alignment_loss:.8f}\n")
                    handle.write(f"lattice_candidate_loss={lattice_candidate_loss:.8f}\n")
                writer.add_scalar("Epoch/Best_Accuracy_Reasoning", best_acc, epoch)
                writer.add_scalar("Train/Epoch/Best_Accuracy_Reasoning", best_acc, epoch)
                writer.flush()
                track_aim_scalar(
                    aim_run,
                    best_acc,
                    name="best_accuracy_reasoning",
                    epoch=epoch,
                    context={"phase": "train", "granularity": "epoch"},
                )
                track_aim_scalar(
                    aim_run,
                    best_acc,
                    name="best_train_accuracy_reasoning",
                    epoch=epoch,
                    context={"phase": "train", "granularity": "epoch"},
                )
                if aim_run is not None:
                    aim_run["best"] = {
                        "epoch": best_epoch,
                        "accuracy_reasoning": best_acc,
                        "train_accuracy_reasoning": best_acc,
                        "checkpoint": os.path.join(args.out_dir, "best.pt"),
                    }

            if args.eval_every > 0 and epoch % args.eval_every == 0:
                heldout = evaluate_heldout_reasoning(
                    model,
                    tokenizer,
                    args=args,
                    device=device,
                    writer=writer,
                    epoch=epoch,
                    aim_run=aim_run,
                )
                heldout_acc = float(heldout["mean_accuracy"])
                stochastic_summary = heldout.get("stochastic_eval_samples", {})
                stochastic_text = ""
                if stochastic_summary:
                    stochastic_text = " | " + " ".join(
                        f"K{samples}=acc:{float(metrics['accuracy']):.4f}"
                        f"/oracle:{float(metrics.get('oracle_accuracy', 0.0)):.4f}"
                        f"/cos:{float(metrics['trajectory_cosine']):.3f}"
                        for samples, metrics in sorted(
                            stochastic_summary.items(),
                            key=lambda item: int(item[0]),
                        )
                    )
                    primary_stochastic_samples, primary_stochastic_metrics = max(
                        stochastic_summary.items(),
                        key=lambda item: int(item[0]),
                    )
                    primary_stochastic_oracle_acc = float(
                        primary_stochastic_metrics.get("oracle_accuracy", 0.0)
                    )
                    if primary_stochastic_oracle_acc > best_stochastic_oracle_acc:
                        best_stochastic_oracle_acc = primary_stochastic_oracle_acc
                        best_stochastic_oracle_epoch = epoch
                        best_stochastic_oracle_samples = int(primary_stochastic_samples)
                        if args.save_best_generalization_checkpoint:
                            torch.save(
                                checkpoint_state_dict(model, trainable_only=args.save_trainable_only),
                                os.path.join(args.out_dir, "best_stochastic_oracle.pt"),
                            )
                        with open(
                            os.path.join(args.out_dir, "best_stochastic_oracle_info.txt"),
                            "w",
                            encoding="utf-8",
                        ) as handle:
                            handle.write(f"best_stochastic_oracle_epoch={best_stochastic_oracle_epoch}\n")
                            handle.write(f"best_stochastic_oracle_samples={best_stochastic_oracle_samples}\n")
                            handle.write(f"best_stochastic_oracle_accuracy={best_stochastic_oracle_acc:.8f}\n")
                            handle.write(f"train_accuracy_at_epoch={acc:.8f}\n")
                            handle.write(f"heldout_mean_accuracy_at_epoch={heldout_acc:.8f}\n")
                            handle.write(f"checkpoint={os.path.join(args.out_dir, 'best_stochastic_oracle.pt')}\n")
                            handle.write(json.dumps(heldout, indent=2, sort_keys=True))
                            handle.write("\n")
                        writer.add_scalar(
                            "Generalization/HeldOut/BestStochasticOracleAccuracy_TRM",
                            best_stochastic_oracle_acc,
                            epoch,
                        )
                        writer.flush()
                        track_aim_scalar(
                            aim_run,
                            best_stochastic_oracle_acc,
                            name="best_generalization_trm_stochastic_oracle_accuracy",
                            epoch=epoch,
                            context={
                                "phase": "generalization",
                                "split": "held_out",
                                "samples": str(best_stochastic_oracle_samples),
                            },
                        )
                        if aim_run is not None:
                            aim_run["best_stochastic_oracle"] = {
                                "epoch": best_stochastic_oracle_epoch,
                                "samples": best_stochastic_oracle_samples,
                                "oracle_accuracy": best_stochastic_oracle_acc,
                                "train_accuracy_at_epoch": acc,
                                "heldout_mean_accuracy_at_epoch": heldout_acc,
                                "checkpoint": os.path.join(args.out_dir, "best_stochastic_oracle.pt"),
                            }
                print(
                    f"Generalization epoch {epoch:3d} | heldout_mean_acc={heldout_acc:.4f} | "
                    + " ".join(
                        f"depth{depth}={heldout['depths'][str(depth)]['accuracy']:.4f}"
                        for depth in args.eval_depths
                    )
                    + (
                        " | "
                        + " ".join(
                            f"LDT@{threshold}=submit:{metrics['submit_rate']:.3f}/"
                            f"sound:{metrics['soundness']:.3f}/alive:{metrics['mean_alive_count']:.2f}"
                            for threshold, metrics in heldout.get("lattice_thresholds_mean", {}).items()
                        )
                    )
                    + stochastic_text,
                    flush=True,
                )
                if source_eval_ds is not None:
                    source_heldout = evaluate_heldout_source_mix(
                        model,
                        source_eval_ds,
                        args=args,
                        device=device,
                        writer=writer,
                        epoch=epoch,
                        aim_run=aim_run,
                    )
                    print(
                        f"VerifiedSource epoch {epoch:3d} | "
                        f"target_token_acc={float(source_heldout['accuracy']):.4f} | "
                        f"target_token_loss={float(source_heldout['loss']):.4f} | "
                        f"tokens={int(source_heldout['total'])}",
                        flush=True,
                    )
                    with open(os.path.join(args.out_dir, "latest_verified_source_eval.json"), "w", encoding="utf-8") as handle:
                        json.dump(source_heldout, handle, indent=2, sort_keys=True)
                        handle.write("\n")
                    if aim_run is not None:
                        aim_run["latest_verified_source_eval"] = source_heldout
                if heldout_acc > best_generalization_acc:
                    improvement = heldout_acc - best_generalization_acc
                    best_generalization_acc = heldout_acc
                    best_generalization_epoch = epoch
                    epochs_since_generalization_improvement = 0
                    if args.save_best_generalization_checkpoint:
                        torch.save(
                            checkpoint_state_dict(model, trainable_only=args.save_trainable_only),
                            os.path.join(args.out_dir, "best_generalization.pt"),
                        )
                    with open(os.path.join(args.out_dir, "best_generalization_info.txt"), "w", encoding="utf-8") as handle:
                        handle.write(f"best_generalization_epoch={best_generalization_epoch}\n")
                        handle.write(f"best_generalization_mean_accuracy={best_generalization_acc:.8f}\n")
                        handle.write(f"train_accuracy_at_epoch={acc:.8f}\n")
                        handle.write(f"checkpoint={os.path.join(args.out_dir, 'best_generalization.pt')}\n")
                        handle.write(json.dumps(heldout, indent=2, sort_keys=True))
                        handle.write("\n")
                    writer.add_scalar(
                        "Generalization/HeldOut/BestMeanAccuracy_TRM",
                        best_generalization_acc,
                        epoch,
                    )
                    track_aim_scalar(
                        aim_run,
                        best_generalization_acc,
                        name="best_generalization_trm_mean_accuracy",
                        epoch=epoch,
                        context={"phase": "generalization", "split": "held_out"},
                    )
                    if aim_run is not None:
                        aim_run["best_generalization"] = {
                            "epoch": best_generalization_epoch,
                            "mean_accuracy": best_generalization_acc,
                            "train_accuracy_at_epoch": acc,
                            "checkpoint": os.path.join(args.out_dir, "best_generalization.pt"),
                        }
                else:
                    improvement = heldout_acc - best_generalization_acc
                    if improvement > float(args.generalization_early_stop_min_delta):
                        epochs_since_generalization_improvement = 0
                    else:
                        epochs_since_generalization_improvement += 1
                if (
                    args.generalization_early_stop_patience > 0
                    and epochs_since_generalization_improvement >= args.generalization_early_stop_patience
                ):
                    message = (
                        f"Early stopping at epoch {epoch}: heldout_mean_acc={heldout_acc:.4f}, "
                        f"best={best_generalization_acc:.4f} at epoch {best_generalization_epoch}"
                    )
                    print(message, flush=True)
                    with open(os.path.join(args.out_dir, "early_stop_info.txt"), "w", encoding="utf-8") as handle:
                        handle.write(message + "\n")
                    if aim_run is not None:
                        aim_run["early_stop"] = {
                            "epoch": epoch,
                            "best_generalization_epoch": best_generalization_epoch,
                            "best_generalization_mean_accuracy": best_generalization_acc,
                            "patience": args.generalization_early_stop_patience,
                        }
                    break

            if args.checkpoint_every > 0:
                if epoch % args.checkpoint_every == 0:
                    torch.save(
                        checkpoint_state_dict(model, trainable_only=args.save_trainable_only),
                        os.path.join(args.out_dir, f"epoch_{epoch}.pt"),
                    )
                if args.save_last_every_epoch:
                    torch.save(
                        checkpoint_state_dict(model, trainable_only=args.save_trainable_only),
                        os.path.join(args.out_dir, "last.pt"),
                    )
    finally:
        writer.close()
        if aim_run is not None:
            aim_run.close()
    if args.checkpoint_every > 0 and not args.save_last_every_epoch:
        torch.save(
            checkpoint_state_dict(model, trainable_only=args.save_trainable_only),
            os.path.join(args.out_dir, "last.pt"),
        )


if __name__ == "__main__":
    main()
