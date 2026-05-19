#!/usr/bin/env python3
"""Train/evaluate the Qwen-backbone QTRM recurrent core gate."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from qtrm_mm.qwen_backbone_qtrm import QwenBackboneQTRM


@dataclass(frozen=True)
class SyntheticCase:
    prompt: str
    label: str
    family: str


def parse_int_list(value: str) -> tuple[int, ...] | None:
    text = str(value).strip()
    if text == "":
        return None
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def parse_float_map(value: str) -> dict[str, float]:
    text = str(value).strip()
    if text == "":
        return {}
    result: dict[str, float] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"expected name=value in float map item: {part!r}")
        name, raw_value = part.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty name in float map item: {part!r}")
        result[name] = float(raw_value.strip())
    return result


def _dtype(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _load_ouro_model(args: argparse.Namespace, *, dtype: torch.dtype, device: torch.device):
    if str(args.core_impl) != "ouro_weight_wrapped":
        return None
    layer_indices = parse_int_list(str(args.ouro_core_layer_indices))
    if bool(args.ouro_partial_safetensors):
        from qtrm_mm.ouro_partial import build_partial_ouro_model_from_safetensors

        return build_partial_ouro_model_from_safetensors(
            str(args.ouro_model_id),
            layer_indices=layer_indices or (18,),
            dtype=dtype,
            device=device,
        )
    try:
        from transformers import AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required to load Ouro") from exc
    model = AutoModelForCausalLM.from_pretrained(
        str(args.ouro_model_id),
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    return model.to(device)


def build_synthetic_cases(
    *,
    count: int,
    seed: int,
    case_mode: str = "standard",
) -> list[SyntheticCase]:
    rng = random.Random(int(seed))
    cases: list[SyntheticCase] = []
    mode = str(case_mode)
    if mode == "standard":
        families = ("checksum", "chain", "select")
    elif mode == "hard_v1":
        families = ("checksum4", "chain5", "select_pair")
    elif mode == "hard_repair_v1":
        families = ("select_pair", "checksum4", "select_pair", "chain5")
    elif mode == "mixed_v1":
        families = ("checksum", "chain", "select", "checksum4", "chain5", "select_pair")
    else:
        raise ValueError(f"unsupported case_mode: {case_mode}")
    for idx in range(int(count)):
        family = families[idx % len(families)]
        if family == "checksum":
            a, b, c = (rng.randrange(10) for _ in range(3))
            answer = (a + 2 * b + 3 * c) % 10
            prompt = (
                "Compute the checksum mod 10. "
                f"Rule: (a + 2*b + 3*c) mod 10. a={a}, b={b}, c={c}. "
                "Answer with one digit. Answer: "
            )
        elif family == "chain":
            start = rng.randrange(10)
            add = rng.randrange(10)
            sub = rng.randrange(10)
            mul = rng.choice((1, 3, 7, 9))
            answer = ((start + add - sub) * mul) % 10
            prompt = (
                "Follow the digit chain mod 10. "
                f"Start {start}; add {add}; subtract {sub}; multiply by {mul}. "
                "Answer with one digit. Answer: "
            )
        elif family == "select":
            digits = [rng.randrange(10) for _ in range(5)]
            pos = rng.randrange(len(digits))
            answer = (digits[pos] + pos) % 10
            prompt = (
                "Read the digit list and answer mod 10. "
                f"Digits: {digits}. Take index {pos}, add the index, mod 10. "
                "Answer with one digit. Answer: "
            )
        elif family == "checksum4":
            a, b, c, d = (rng.randrange(10) for _ in range(4))
            answer = (a + 2 * b + 3 * c + 4 * d) % 10
            prompt = (
                "Compute the extended checksum mod 10. "
                f"Rule: (a + 2*b + 3*c + 4*d) mod 10. "
                f"a={a}, b={b}, c={c}, d={d}. "
                "Answer with one digit. Answer: "
            )
        elif family == "chain5":
            start = rng.randrange(10)
            add_a = rng.randrange(10)
            mul = rng.choice((1, 3, 7, 9))
            sub = rng.randrange(10)
            add_b = rng.randrange(10)
            answer = (((start + add_a) * mul - sub) + add_b) % 10
            prompt = (
                "Follow the five-step digit chain mod 10. "
                f"Start {start}; add {add_a}; multiply by {mul}; "
                f"subtract {sub}; add {add_b}. "
                "Answer with one digit. Answer: "
            )
        elif family == "select_pair":
            digits = [rng.randrange(10) for _ in range(7)]
            first = rng.randrange(len(digits))
            second = rng.randrange(len(digits))
            answer = (digits[first] + digits[second] + first + second) % 10
            prompt = (
                "Read the digit list and answer mod 10. "
                f"Digits: {digits}. Take indices {first} and {second}; "
                "add both selected digits and both indices, mod 10. "
                "Answer with one digit. Answer: "
            )
        else:  # pragma: no cover - guarded by case_mode family lists
            raise AssertionError(f"unknown family: {family}")
        cases.append(SyntheticCase(prompt=prompt, label=str(answer), family=family))
    return cases


_CHECKSUM4_RE = re.compile(r"a=(\d+), b=(\d+), c=(\d+), d=(\d+)")


def checksum4_counterfactual_cases(
    case: SyntheticCase,
    *,
    variants: int,
) -> list[SyntheticCase]:
    if case.family != "checksum4" or int(variants) <= 0:
        return []
    match = _CHECKSUM4_RE.search(case.prompt)
    if match is None:
        return []
    values = [int(match.group(index)) for index in range(1, 5)]
    weights = [1, 2, 3, 4]
    result: list[SyntheticCase] = []
    for variant in range(int(variants)):
        index = 3 - (variant % 4)
        delta = 1 + (variant // 4)
        edited = list(values)
        edited[index] = (edited[index] + delta) % 10
        a, b, c, d = edited
        answer = sum(weight * value for weight, value in zip(weights, edited)) % 10
        prompt = (
            "Compute the extended checksum mod 10. "
            "Rule: (a + 2*b + 3*c + 4*d) mod 10. "
            f"a={a}, b={b}, c={c}, d={d}. "
            "Answer with one digit. Answer: "
        )
        result.append(SyntheticCase(prompt=prompt, label=str(answer), family="checksum4"))
    return result


def checksum4_residue_targets(case: SyntheticCase) -> list[int]:
    if case.family != "checksum4":
        return []
    match = _CHECKSUM4_RE.search(case.prompt)
    if match is None:
        return []
    a, b, c, d = (int(match.group(index)) for index in range(1, 5))
    return [
        a % 10,
        (a + 2 * b) % 10,
        (a + 2 * b + 3 * c) % 10,
        (a + 2 * b + 3 * c + 4 * d) % 10,
    ]


BASIC_LANGUAGE_PROBE_PROMPTS = (
    "User: Explain why evidence should be checked.\nAssistant: ",
    "User: 양자 컴퓨팅이란 무엇인가요?\nAssistant: ",
    "User: Write one clear sentence about careful reasoning.\nAssistant: ",
    "User: What should a model do when it is uncertain?\nAssistant: ",
    "User: Explain quantum entanglement in one simple sentence.\nAssistant: ",
    "User: Translate to Korean: Careful reasoning reduces mistakes.\nAssistant: ",
    "User: 좋은 연구 노트를 쓰는 방법을 짧게 말해 주세요.\nAssistant: ",
    "User: 사실과 의견의 차이를 설명해 주세요.\nAssistant: ",
)


EXTENDED_LANGUAGE_PROBE_PROMPTS = BASIC_LANGUAGE_PROBE_PROMPTS + (
    "User: Summarize this in one sentence: Evidence can be incomplete, so conclusions should stay calibrated.\nAssistant: ",
    "User: Give a concise reason why repeated experiments matter.\nAssistant: ",
    "User: What is the difference between correlation and causation?\nAssistant: ",
    "User: Translate to English: 신중한 판단은 실수를 줄입니다.\nAssistant: ",
    "User: 다음 문장을 영어로 번역하세요: 증거가 부족하면 결론을 보류해야 합니다.\nAssistant: ",
    "User: 불확실할 때 좋은 답변은 어떻게 해야 하나요?\nAssistant: ",
    "User: 인과관계와 상관관계의 차이를 짧게 설명해 주세요.\nAssistant: ",
    "User: Write a short definition of a hypothesis.\nAssistant: ",
    "User: Explain why a model should not invent citations.\nAssistant: ",
    "User: 데이터 품질이 모델 성능에 중요한 이유를 한 문장으로 말해 주세요.\nAssistant: ",
    "User: What should happen if two sources conflict?\nAssistant: ",
    "User: 두 자료가 서로 모순될 때 먼저 확인할 것은 무엇인가요?\nAssistant: ",
    "User: Write one sentence about scientific humility.\nAssistant: ",
    "User: 과학적 겸손이 왜 필요한지 짧게 말해 주세요.\nAssistant: ",
    "User: Explain a checksum in simple terms.\nAssistant: ",
    "User: 체크섬이 무엇인지 쉽게 설명해 주세요.\nAssistant: ",
    "User: Give a short answer: should an AI guess when it lacks evidence?\nAssistant: ",
    "User: 근거가 없을 때 AI가 추측해야 하나요?\nAssistant: ",
    "User: Describe a careful debugging process in one sentence.\nAssistant: ",
    "User: 버그를 찾을 때 로그가 왜 중요한가요?\nAssistant: ",
    "User: What does it mean to verify an answer?\nAssistant: ",
    "User: 답변을 검증한다는 것은 무슨 뜻인가요?\nAssistant: ",
    "User: Explain memory in a language model at a high level.\nAssistant: ",
    "User: 언어 모델의 기억을 높은 수준에서 설명해 주세요.\nAssistant: ",
)


LANGUAGE_HEALING_EXAMPLES = (
    (
        "User: Explain why evidence should be checked.\nAssistant: ",
        "Evidence should be checked because weak or outdated evidence can lead to wrong conclusions.",
    ),
    (
        "User: 양자 컴퓨팅이란 무엇인가요?\nAssistant: ",
        "양자 컴퓨팅은 양자 상태를 이용해 특정 계산을 더 효율적으로 처리하려는 컴퓨팅 방식입니다.",
    ),
    (
        "User: Write one clear sentence about careful reasoning.\nAssistant: ",
        "Careful reasoning compares evidence, alternatives, and uncertainty before reaching a conclusion.",
    ),
    (
        "User: What should a model do when it is uncertain?\nAssistant: ",
        "It should say what is uncertain, avoid guessing, and seek better evidence when possible.",
    ),
    (
        "User: Translate to Korean: Careful reasoning reduces mistakes.\nAssistant: ",
        "신중한 추론은 실수를 줄입니다.",
    ),
    (
        "User: Translate to English: 신중한 판단은 실수를 줄입니다.\nAssistant: ",
        "Careful judgment reduces mistakes.",
    ),
    (
        "User: 불확실할 때 좋은 답변은 어떻게 해야 하나요?\nAssistant: ",
        "좋은 답변은 불확실한 부분을 밝히고, 근거를 확인하며, 단정적인 추측을 피해야 합니다.",
    ),
    (
        "User: What is the difference between correlation and causation?\nAssistant: ",
        "Correlation means two things vary together; causation means one thing directly helps produce the other.",
    ),
    (
        "User: 인과관계와 상관관계의 차이를 짧게 설명해 주세요.\nAssistant: ",
        "상관관계는 함께 변한다는 뜻이고, 인과관계는 한쪽이 다른 쪽의 원인이 된다는 뜻입니다.",
    ),
    (
        "User: What should happen if two sources conflict?\nAssistant: ",
        "The answer should compare source quality, dates, methods, and direct evidence before deciding.",
    ),
    (
        "User: 두 자료가 서로 모순될 때 먼저 확인할 것은 무엇인가요?\nAssistant: ",
        "먼저 출처의 신뢰도, 작성 시점, 근거의 직접성, 측정 방법을 확인해야 합니다.",
    ),
    (
        "User: Explain a checksum in simple terms.\nAssistant: ",
        "A checksum is a small value used to check whether data was copied or transmitted correctly.",
    ),
    (
        "User: 체크섬이 무엇인지 쉽게 설명해 주세요.\nAssistant: ",
        "체크섬은 데이터가 중간에 바뀌었는지 확인하기 위해 계산하는 작은 확인값입니다.",
    ),
    (
        "User: Give a short answer: should an AI guess when it lacks evidence?\nAssistant: ",
        "No. It should state the uncertainty and look for reliable evidence.",
    ),
    (
        "User: 근거가 없을 때 AI가 추측해야 하나요?\nAssistant: ",
        "아니요. 근거가 부족하다고 말하고 필요한 정보를 더 찾아야 합니다.",
    ),
    (
        "User: Describe a careful debugging process in one sentence.\nAssistant: ",
        "A careful debugging process reproduces the failure, isolates the cause, changes one thing, and verifies the fix.",
    ),
)


def language_probe_prompts(probe_set: str = "basic") -> list[str]:
    if str(probe_set) == "extended":
        return list(EXTENDED_LANGUAGE_PROBE_PROMPTS)
    if str(probe_set) != "basic":
        raise ValueError(f"unknown language probe set: {probe_set}")
    return list(BASIC_LANGUAGE_PROBE_PROMPTS)


def language_healing_examples() -> list[tuple[str, str]]:
    return list(LANGUAGE_HEALING_EXAMPLES)


def _label_token_ids(tokenizer) -> dict[str, int]:
    result: dict[str, int] = {}
    for digit in "0123456789":
        ids = tokenizer.encode(digit, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"digit label is not a single token: {digit} -> {ids}")
        result[digit] = int(ids[0])
    return result


def _digit_choice_predictions(logits: torch.Tensor, label_ids: dict[str, int]) -> torch.Tensor:
    digits = list("0123456789")
    token_ids = torch.tensor([label_ids[digit] for digit in digits], device=logits.device)
    choice_index = logits.index_select(dim=-1, index=token_ids).argmax(dim=-1)
    return choice_index


def _batch(items: list[SyntheticCase], batch_size: int) -> Iterable[list[SyntheticCase]]:
    for start in range(0, len(items), int(batch_size)):
        yield items[start : start + int(batch_size)]


def _encode_prompts(tokenizer, prompts: list[str], *, max_seq_len: int, device: torch.device):
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_seq_len),
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def _encode_prefix_response_examples(
    tokenizer,
    examples: list[tuple[str, str]],
    *,
    max_seq_len: int,
    device: torch.device,
):
    texts = [prompt + response for prompt, response in examples]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_seq_len),
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    labels = torch.full_like(input_ids, -100)
    for row, (prompt, _response) in enumerate(examples):
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        row_len = (
            int(attention_mask[row].sum().item())
            if attention_mask is not None
            else int(input_ids.shape[1])
        )
        start = min(prompt_len, row_len)
        if start < row_len:
            labels[row, start:row_len] = input_ids[row, start:row_len]
    return input_ids, attention_mask, labels


def _response_only_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.float().view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=-100,
    )


def _response_only_kl_loss(
    core_logits: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    mask = labels[:, 1:].ne(-100)
    if not bool(mask.any().item()):
        return core_logits.new_tensor(0.0, dtype=torch.float32)
    core_selected = core_logits[:, :-1, :][mask]
    base_selected = base_logits[:, :-1, :][mask]
    return F.kl_div(
        F.log_softmax(core_selected.float(), dim=-1),
        F.softmax(base_selected.float(), dim=-1),
        reduction="batchmean",
    )


def _last_token_logits(model, input_ids, attention_mask, *, force_core_off: bool = False):
    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        force_core_off=force_core_off,
    )
    return outputs.logits[:, -1, :]


def _last_token_state(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return hidden[:, -1, :]
    index = attention_mask.long().sum(dim=1).clamp_min(1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), index]


def _last_token_step_states(
    step_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if attention_mask is None:
        return step_states[:, :, -1, :]
    index = attention_mask.long().sum(dim=1).clamp_min(1) - 1
    return step_states[
        torch.arange(step_states.shape[0], device=step_states.device),
        :,
        index,
        :,
    ]


@torch.no_grad()
def evaluate_cases(model, tokenizer, cases: list[SyntheticCase], args, label_ids: dict[str, int]):
    device = next(model.parameters()).device
    total = 0
    core_correct = 0
    base_correct = 0
    core_choice_correct = 0
    base_choice_correct = 0
    core_outer_iterations = []
    core_converged = []
    by_family: dict[str, dict[str, int]] = {}
    base_logits_finite = True
    core_logits_finite = True
    choice_targets = {digit: idx for idx, digit in enumerate("0123456789")}
    for chunk in _batch(cases, int(args.batch_size)):
        input_ids, attention_mask = _encode_prompts(
            tokenizer,
            [case.prompt for case in chunk],
            max_seq_len=int(args.max_seq_len),
            device=device,
        )
        targets = torch.tensor([label_ids[case.label] for case in chunk], device=device)
        choice_target = torch.tensor([choice_targets[case.label] for case in chunk], device=device)
        base_logits = _last_token_logits(model, input_ids, attention_mask, force_core_off=True)
        core_outputs = model(
            input_ids,
            attention_mask=attention_mask,
            force_trajectory_carry_off=bool(args.eval_force_trajectory_carry_off),
        )
        core_logits = core_outputs.logits[:, -1, :]
        base_logits_finite = bool(base_logits_finite and torch.isfinite(base_logits).all().item())
        core_logits_finite = bool(core_logits_finite and torch.isfinite(core_logits).all().item())
        if hasattr(core_outputs, "qtrm_core_outer_iterations"):
            outer = getattr(core_outputs, "qtrm_core_outer_iterations")
            if outer is not None:
                core_outer_iterations.extend(float(value) for value in outer.detach().cpu().view(-1))
        if hasattr(core_outputs, "qtrm_core_converged"):
            converged = getattr(core_outputs, "qtrm_core_converged")
            if converged is not None:
                core_converged.extend(bool(value) for value in converged.detach().cpu().view(-1))
        base_pred = base_logits.argmax(dim=-1)
        core_pred = core_logits.argmax(dim=-1)
        base_choice_pred = _digit_choice_predictions(base_logits, label_ids)
        core_choice_pred = _digit_choice_predictions(core_logits, label_ids)
        for case, base_item, core_item, base_choice, core_choice, target, choice_item in zip(
            chunk,
            base_pred,
            core_pred,
            base_choice_pred,
            core_choice_pred,
            targets,
            choice_target,
        ):
            fam = by_family.setdefault(
                case.family,
                {"total": 0, "base": 0, "core": 0, "base_choice": 0, "core_choice": 0},
            )
            fam["total"] += 1
            total += 1
            if int(base_item) == int(target):
                fam["base"] += 1
                base_correct += 1
            if int(core_item) == int(target):
                fam["core"] += 1
                core_correct += 1
            if int(base_choice) == int(choice_item):
                fam["base_choice"] += 1
                base_choice_correct += 1
            if int(core_choice) == int(choice_item):
                fam["core_choice"] += 1
                core_choice_correct += 1
    return {
        "total": total,
        "base_accuracy": base_correct / max(1, total),
        "core_accuracy": core_correct / max(1, total),
        "gain": (core_correct - base_correct) / max(1, total),
        "base_choice_accuracy": base_choice_correct / max(1, total),
        "core_choice_accuracy": core_choice_correct / max(1, total),
        "choice_gain": (core_choice_correct - base_choice_correct) / max(1, total),
        "mean_core_outer_iterations": (
            sum(core_outer_iterations) / len(core_outer_iterations)
            if core_outer_iterations
            else None
        ),
        "core_converged_fraction": (
            sum(1 for value in core_converged if value) / len(core_converged)
            if core_converged
            else None
        ),
        "base_logits_finite": bool(base_logits_finite),
        "core_logits_finite": bool(core_logits_finite),
        "by_family": {
            family: {
                "base_accuracy": row["base"] / max(1, row["total"]),
                "core_accuracy": row["core"] / max(1, row["total"]),
                "gain": (row["core"] - row["base"]) / max(1, row["total"]),
                "base_choice_accuracy": row["base_choice"] / max(1, row["total"]),
                "core_choice_accuracy": row["core_choice"] / max(1, row["total"]),
                "choice_gain": (row["core_choice"] - row["base_choice"]) / max(1, row["total"]),
                "total": row["total"],
            }
            for family, row in sorted(by_family.items())
        },
    }


def family_gain_summary(evaluation: dict[str, object], *, metric: str = "full_vocab") -> dict[str, object]:
    rows = evaluation.get("by_family", {})
    gains = {}
    core_accuracies = {}
    if metric == "full_vocab":
        base_key = "base_accuracy"
        core_key = "core_accuracy"
    elif metric == "label_choice":
        base_key = "base_choice_accuracy"
        core_key = "core_choice_accuracy"
    else:
        raise ValueError(f"unsupported family summary metric: {metric}")
    if isinstance(rows, dict):
        for family, row in rows.items():
            if not isinstance(row, dict):
                continue
            base = float(row.get(base_key, 0.0))
            core = float(row.get(core_key, 0.0))
            gains[str(family)] = core - base
            core_accuracies[str(family)] = core
    min_gain = min(gains.values()) if gains else 0.0
    min_core_accuracy = min(core_accuracies.values()) if core_accuracies else 0.0
    return {
        "gains": gains,
        "core_accuracies": core_accuracies,
        "min_gain": float(min_gain),
        "min_core_accuracy": float(min_core_accuracy),
        "metric": str(metric),
    }


def evaluation_acceptance_summary(evaluation: dict[str, object], args) -> dict[str, object]:
    metric = str(args.acceptance_metric)
    family_summary = family_gain_summary(evaluation, metric=metric)
    gain = (
        float(evaluation["gain"])
        if metric == "full_vocab"
        else float(evaluation["choice_gain"])
    )
    accepted_reasoning = gain >= float(args.min_reasoning_gain)
    accepted_family_gain = float(family_summary["min_gain"]) >= float(args.min_family_gain)
    accepted_family_accuracy = float(family_summary["min_core_accuracy"]) >= float(
        args.min_family_core_accuracy
    )
    accepted_finite_logits = bool(
        evaluation.get("base_logits_finite", True)
        and evaluation.get("core_logits_finite", True)
    )
    # Prefer threshold pass, then larger family floor, then larger aggregate gain.
    score = (
        (1.0 if accepted_reasoning else 0.0)
        + (1.0 if accepted_family_gain else 0.0)
        + (1.0 if accepted_family_accuracy else 0.0)
        + float(family_summary["min_core_accuracy"])
        + float(family_summary["min_gain"])
        + gain
    )
    return {
        "metric": metric,
        "gain": gain,
        "family_summary": family_summary,
        "accepted_reasoning_gain": bool(accepted_reasoning),
        "accepted_family_gain": bool(accepted_family_gain),
        "accepted_family_core_accuracy": bool(accepted_family_accuracy),
        "accepted_finite_logits": bool(accepted_finite_logits),
        "score": float(score),
    }


@torch.no_grad()
def evaluate_language_non_regression(model, tokenizer, args):
    device = next(model.parameters()).device
    prompts = language_probe_prompts(str(args.language_probe_set))
    input_ids, attention_mask = _encode_prompts(
        tokenizer,
        prompts,
        max_seq_len=int(args.max_seq_len),
        device=device,
    )
    base = _last_token_logits(model, input_ids, attention_mask, force_core_off=True)
    core = _last_token_logits(model, input_ids, attention_mask)
    finite = bool(torch.isfinite(base).all().item() and torch.isfinite(core).all().item())
    if finite:
        agreement = (base.argmax(dim=-1) == core.argmax(dim=-1)).float().mean().item()
        mean_abs_delta = (base.float() - core.float()).abs().mean().item()
    else:
        agreement = 0.0
        mean_abs_delta = float("inf")
    return {
        "top1_agreement": float(agreement),
        "mean_abs_delta": float(mean_abs_delta),
        "finite_logits": bool(finite),
        "probe_set": str(args.language_probe_set),
        "num_prompts": len(prompts),
    }


def train_core(
    model,
    tokenizer,
    train_cases: list[SyntheticCase],
    args,
    label_ids: dict[str, int],
    *,
    eval_cases: list[SyntheticCase] | None = None,
):
    device = next(model.parameters()).device
    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable = [parameter for _, parameter in named_trainable]
    qwen_trainable = [(name, parameter) for name, parameter in named_trainable if name.startswith("qwen.")]
    core_trainable = [
        (name, parameter) for name, parameter in named_trainable if not name.startswith("qwen.")
    ]
    param_groups = []
    if core_trainable:
        param_groups.append(
            {
                "params": [parameter for _, parameter in core_trainable],
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
            }
        )
    if qwen_trainable:
        param_groups.append(
            {
                "params": [parameter for _, parameter in qwen_trainable],
                "lr": float(args.qwen_lr),
                "weight_decay": float(args.qwen_weight_decay),
            }
        )
    if not param_groups:
        raise ValueError("no trainable parameters")
    optimizer = torch.optim.AdamW(param_groups)
    rng = random.Random(int(args.seed) + 17)
    family_loss_weights = parse_float_map(str(args.family_loss_weights))
    digit_labels = list("0123456789")
    label_token_ids = torch.tensor([label_ids[digit] for digit in digit_labels], device=device)
    latent_answer_head: torch.nn.Linear | None = None
    latent_answer_head_optimizer = None
    latent_answer_head_trainable = []
    if float(args.checksum_latent_answer_weight) > 0.0:
        latent_answer_head = torch.nn.Linear(int(model.report().hidden_size), 10).to(device=device)
        latent_answer_head_trainable = list(latent_answer_head.parameters())
        latent_answer_head_optimizer = torch.optim.AdamW(
            latent_answer_head_trainable,
            lr=float(args.checksum_latent_answer_lr),
            weight_decay=float(args.checksum_latent_answer_weight_decay),
        )
    losses = []
    checksum_counterfactual_losses = []
    checksum_latent_answer_losses = []
    checksum_trajectory_losses = []
    language_healing_losses = []
    language_healing_kl_losses = []
    best: dict[str, object] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    model.train()
    if any(parameter.requires_grad for parameter in model.qwen.parameters()):
        model.qwen.train()
    else:
        model.qwen.eval()
    if hasattr(model, "ouro_model"):
        model.ouro_model.eval()
    for step in range(1, int(args.steps) + 1):
        chunk = rng.sample(train_cases, k=min(int(args.batch_size), len(train_cases)))
        input_ids, attention_mask = _encode_prompts(
            tokenizer,
            [case.prompt for case in chunk],
            max_seq_len=int(args.max_seq_len),
            device=device,
        )
        target_labels = [case.label for case in chunk]
        targets = torch.tensor([label_ids[label] for label in target_labels], device=device)
        target_choice_indices = torch.tensor(
            [digit_labels.index(label) for label in target_labels],
            device=device,
            dtype=torch.long,
        )
        core_outputs = model(input_ids, attention_mask=attention_mask)
        core_logits = core_outputs.logits[:, -1, :]
        per_item_ce = F.cross_entropy(core_logits.float(), targets, reduction="none")
        if family_loss_weights:
            weights = torch.tensor(
                [family_loss_weights.get(case.family, 1.0) for case in chunk],
                device=device,
                dtype=per_item_ce.dtype,
            )
            ce = (per_item_ce * weights).sum() / weights.sum().clamp_min(1e-6)
        else:
            ce = per_item_ce.mean()
        loss = ce
        checksum_counterfactual_weight = float(args.checksum_counterfactual_weight)
        if checksum_counterfactual_weight > 0.0:
            counterfactual_cases: list[SyntheticCase] = []
            for case in chunk:
                counterfactual_cases.extend(
                    checksum4_counterfactual_cases(
                        case,
                        variants=int(args.checksum_counterfactual_variants),
                    )
                )
            if counterfactual_cases:
                cf_input_ids, cf_attention_mask = _encode_prompts(
                    tokenizer,
                    [case.prompt for case in counterfactual_cases],
                    max_seq_len=int(args.max_seq_len),
                    device=device,
                )
                cf_targets = torch.tensor(
                    [label_ids[case.label] for case in counterfactual_cases],
                    device=device,
                )
                cf_logits = _last_token_logits(model, cf_input_ids, cf_attention_mask)
                cf_ce = F.cross_entropy(cf_logits.float(), cf_targets)
                loss = loss + checksum_counterfactual_weight * cf_ce
                checksum_counterfactual_losses.append(float(cf_ce.detach().cpu()))
        base_logits = None
        if (
            float(args.kl_weight) > 0.0
            or float(args.core_advantage_weight) > 0.0
            or float(args.checksum_base_error_advantage_weight) > 0.0
        ):
            with torch.no_grad():
                base_logits = _last_token_logits(model, input_ids, attention_mask, force_core_off=True)
        if float(args.core_advantage_weight) > 0.0:
            if base_logits is None:
                raise RuntimeError("base logits were not computed for core advantage loss")
            row_index = torch.arange(targets.numel(), device=device)
            if str(args.core_advantage_mode) == "target_logp":
                core_log_probs = F.log_softmax(core_logits.float(), dim=-1)
                base_log_probs = F.log_softmax(base_logits.float(), dim=-1)
                core_target_logp = core_log_probs[row_index, targets]
                base_target_logp = base_log_probs[row_index, targets]
                target_margin = core_target_logp - base_target_logp
            elif str(args.core_advantage_mode) == "label_choice_margin":
                core_choice_logits = core_logits.float().index_select(dim=-1, index=label_token_ids)
                base_choice_logits = base_logits.float().index_select(dim=-1, index=label_token_ids)
                wrong_mask = torch.ones_like(core_choice_logits, dtype=torch.bool)
                wrong_mask[row_index, target_choice_indices] = False
                core_target = core_choice_logits[row_index, target_choice_indices]
                base_target = base_choice_logits[row_index, target_choice_indices]
                core_wrong = core_choice_logits.masked_fill(~wrong_mask, float("-inf")).amax(dim=-1)
                base_wrong = base_choice_logits.masked_fill(~wrong_mask, float("-inf")).amax(dim=-1)
                target_margin = (core_target - core_wrong) - (base_target - base_wrong)
            else:
                raise ValueError(f"unknown core advantage mode: {args.core_advantage_mode}")
            per_item_advantage = F.relu(float(args.core_advantage_margin) - target_margin)
            if family_loss_weights:
                advantage = (per_item_advantage * weights).sum() / weights.sum().clamp_min(1e-6)
            else:
                advantage = per_item_advantage.mean()
            loss = loss + float(args.core_advantage_weight) * advantage
        if float(args.checksum_base_error_advantage_weight) > 0.0:
            if base_logits is None:
                raise RuntimeError("base logits were not computed for checksum base-error loss")
            row_index = torch.arange(targets.numel(), device=device)
            core_choice_logits = core_logits.float().index_select(dim=-1, index=label_token_ids)
            base_choice_logits = base_logits.float().index_select(dim=-1, index=label_token_ids)
            wrong_mask = torch.ones_like(core_choice_logits, dtype=torch.bool)
            wrong_mask[row_index, target_choice_indices] = False
            core_target = core_choice_logits[row_index, target_choice_indices]
            base_target = base_choice_logits[row_index, target_choice_indices]
            core_wrong = core_choice_logits.masked_fill(~wrong_mask, float("-inf")).amax(dim=-1)
            base_wrong = base_choice_logits.masked_fill(~wrong_mask, float("-inf")).amax(dim=-1)
            base_margin = base_target - base_wrong
            core_margin = core_target - core_wrong
            family_mask = torch.tensor(
                [case.family == "checksum4" for case in chunk],
                device=device,
                dtype=torch.bool,
            )
            weak_base_mask = base_margin < float(args.checksum_base_error_base_margin_threshold)
            checksum_mask = family_mask & weak_base_mask
            if bool(checksum_mask.any().item()):
                checksum_margin_loss = F.relu(
                    float(args.checksum_base_error_margin) - core_margin[checksum_mask]
                ).mean()
                loss = loss + float(args.checksum_base_error_advantage_weight) * checksum_margin_loss
        if latent_answer_head is not None:
            latent_source_name = str(args.checksum_latent_answer_source)
            latent_source = (
                getattr(core_outputs, "qtrm_core_delta", None)
                if latent_source_name == "delta_h"
                else getattr(core_outputs, "qtrm_core_hidden", None)
            )
            if latent_source is None:
                raise RuntimeError(f"missing latent source for checksum latent answer: {latent_source_name}")
            checksum_mask = torch.tensor(
                [case.family == "checksum4" for case in chunk],
                device=device,
                dtype=torch.bool,
            )
            if bool(checksum_mask.any().item()):
                latent_last = _last_token_state(latent_source, attention_mask)
                latent_logits = latent_answer_head(latent_last.float())
                latent_loss = F.cross_entropy(
                    latent_logits[checksum_mask].float(),
                    target_choice_indices[checksum_mask],
                )
                loss = loss + float(args.checksum_latent_answer_weight) * latent_loss
                checksum_latent_answer_losses.append(float(latent_loss.detach().cpu()))
        if float(args.checksum_trajectory_weight) > 0.0:
            step_states = getattr(core_outputs, "qtrm_core_step_states", None)
            if step_states is None:
                raise RuntimeError("missing qtrm_core_step_states for checksum trajectory loss")
            residue_rows: list[list[int]] = []
            residue_indices: list[int] = []
            for item_index, case in enumerate(chunk):
                residues = checksum4_residue_targets(case)
                if residues:
                    residue_indices.append(item_index)
                    residue_rows.append(residues)
            if residue_rows:
                usable_steps = min(step_states.shape[1], len(residue_rows[0]))
                selected_states = step_states[residue_indices, :usable_steps]
                selected_last = _last_token_step_states(
                    selected_states,
                    attention_mask[residue_indices] if attention_mask is not None else None,
                )
                selected_last = model.core_out_norm(selected_last).to(dtype=core_logits.dtype)
                trajectory_logits = model._lm_head()(selected_last)
                trajectory_choice_logits = trajectory_logits.float().index_select(
                    dim=-1,
                    index=label_token_ids,
                )
                residue_targets = torch.tensor(
                    [row[:usable_steps] for row in residue_rows],
                    device=device,
                    dtype=torch.long,
                )
                trajectory_loss = F.cross_entropy(
                    trajectory_choice_logits.reshape(-1, trajectory_choice_logits.shape[-1]),
                    residue_targets.reshape(-1),
                )
                loss = loss + float(args.checksum_trajectory_weight) * trajectory_loss
                checksum_trajectory_losses.append(float(trajectory_loss.detach().cpu()))
        if float(args.kl_weight) > 0.0:
            if base_logits is None:
                raise RuntimeError("base logits were not computed for KL loss")
            kl = F.kl_div(
                F.log_softmax(core_logits.float(), dim=-1),
                F.softmax(base_logits.float(), dim=-1),
                reduction="batchmean",
            )
            loss = loss + float(args.kl_weight) * kl
        if float(args.language_kl_weight) > 0.0:
            language_prompts = language_probe_prompts(str(args.language_probe_set))
            language_chunk = [
                rng.choice(language_prompts)
                for _ in range(max(1, int(args.language_kl_batch_size)))
            ]
            lang_input_ids, lang_attention_mask = _encode_prompts(
                tokenizer,
                language_chunk,
                max_seq_len=int(args.max_seq_len),
                device=device,
            )
            lang_core_logits = _last_token_logits(model, lang_input_ids, lang_attention_mask)
            with torch.no_grad():
                lang_base_logits = _last_token_logits(
                    model,
                    lang_input_ids,
                    lang_attention_mask,
                    force_core_off=True,
                )
            lang_kl = F.kl_div(
                F.log_softmax(lang_core_logits.float(), dim=-1),
                F.softmax(lang_base_logits.float(), dim=-1),
                reduction="batchmean",
            )
            loss = loss + float(args.language_kl_weight) * lang_kl
        if float(args.language_healing_weight) > 0.0:
            healing_examples = language_healing_examples()
            healing_chunk = [
                rng.choice(healing_examples)
                for _ in range(max(1, int(args.language_healing_batch_size)))
            ]
            heal_input_ids, heal_attention_mask, heal_labels = _encode_prefix_response_examples(
                tokenizer,
                healing_chunk,
                max_seq_len=int(args.max_seq_len),
                device=device,
            )
            heal_core_outputs = model(heal_input_ids, attention_mask=heal_attention_mask)
            heal_ce = _response_only_ce_loss(heal_core_outputs.logits, heal_labels)
            loss = loss + float(args.language_healing_weight) * heal_ce
            language_healing_losses.append(float(heal_ce.detach().cpu()))
            if float(args.language_healing_kl_weight) > 0.0:
                with torch.no_grad():
                    heal_base_outputs = model(
                        heal_input_ids,
                        attention_mask=heal_attention_mask,
                        force_core_off=True,
                    )
                heal_kl = _response_only_kl_loss(
                    heal_core_outputs.logits,
                    heal_base_outputs.logits,
                    heal_labels,
                )
                loss = loss + float(args.language_healing_kl_weight) * heal_kl
                language_healing_kl_losses.append(float(heal_kl.detach().cpu()))
        if not torch.isfinite(loss.detach()):
            raise RuntimeError(
                f"non-finite training loss at step {step}: {float(loss.detach().cpu())}"
            )
        optimizer.zero_grad(set_to_none=True)
        if latent_answer_head_optimizer is not None:
            latent_answer_head_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
        if latent_answer_head_trainable:
            torch.nn.utils.clip_grad_norm_(latent_answer_head_trainable, float(args.grad_clip))
        optimizer.step()
        if latent_answer_head_optimizer is not None:
            latent_answer_head_optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % int(args.log_every) == 0 or step == 1 or step == int(args.steps):
            print(f"step={step} loss={losses[-1]:.4f}")
        if (
            eval_cases is not None
            and int(args.eval_every_steps) > 0
            and step % int(args.eval_every_steps) == 0
        ):
            model.eval()
            evaluation = evaluate_cases(model, tokenizer, eval_cases, args, label_ids)
            summary = evaluation_acceptance_summary(evaluation, args)
            if (
                float(args.selection_language_weight) > 0.0
                or float(args.selection_min_language_top1) > 0.0
            ):
                language_summary = evaluate_language_non_regression(model, tokenizer, args)
                language_top1 = float(language_summary["top1_agreement"])
                summary["language"] = language_summary
                summary["accepted_selection_language"] = (
                    language_top1 >= float(args.selection_min_language_top1)
                )
                summary["score"] = float(summary["score"]) + (
                    float(args.selection_language_weight) * language_top1
                )
                if language_top1 < float(args.selection_min_language_top1):
                    summary["score"] = float(summary["score"]) - 10.0
            summary["step"] = int(step)
            summary["loss"] = losses[-1]
            if best is None or float(summary["score"]) > float(best["score"]):
                best = summary
                best_state = _trainable_state_dict(model)
            print(
                "eval_step="
                f"{step} gain={summary['gain']:.4f} "
                f"min_family_gain={summary['family_summary']['min_gain']:.4f} "
                f"min_family_core_accuracy={summary['family_summary']['min_core_accuracy']:.4f}"
            )
            model.train()
            if any(parameter.requires_grad for parameter in model.qwen.parameters()):
                model.qwen.train()
            else:
                model.qwen.eval()
            if hasattr(model, "ouro_model"):
                model.ouro_model.eval()
    if best_state is not None and bool(args.restore_best_checkpoint):
        incompatible = model.load_state_dict(best_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected best checkpoint keys: {incompatible.unexpected_keys[:8]}")
    return {
        "last_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / max(1, len(losses)),
        "mean_checksum_counterfactual_loss": (
            sum(checksum_counterfactual_losses) / len(checksum_counterfactual_losses)
            if checksum_counterfactual_losses
            else None
        ),
        "mean_checksum_latent_answer_loss": (
            sum(checksum_latent_answer_losses) / len(checksum_latent_answer_losses)
            if checksum_latent_answer_losses
            else None
        ),
        "mean_checksum_trajectory_loss": (
            sum(checksum_trajectory_losses) / len(checksum_trajectory_losses)
            if checksum_trajectory_losses
            else None
        ),
        "mean_language_healing_loss": (
            sum(language_healing_losses) / len(language_healing_losses)
            if language_healing_losses
            else None
        ),
        "mean_language_healing_kl_loss": (
            sum(language_healing_kl_losses) / len(language_healing_kl_losses)
            if language_healing_kl_losses
            else None
        ),
        "best_periodic_eval": best,
        "restored_best_checkpoint": bool(best_state is not None and bool(args.restore_best_checkpoint)),
    }


def _trainable_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        key: parameter.detach().cpu()
        for key, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _load_trainable_checkpoint(model, checkpoint_path: str) -> dict[str, object]:
    if not str(checkpoint_path):
        return {"path": "", "loaded": False}
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    trainable_keys = {key for key, parameter in model.named_parameters() if parameter.requires_grad}
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"unexpected init checkpoint keys: {unexpected[:8]}")
    missing = list(incompatible.missing_keys)
    trainable_missing = sorted(key for key in missing if key in trainable_keys)
    trainable_loaded = sorted(key for key in state.keys() if key in trainable_keys)
    return {
        "path": str(checkpoint_path),
        "loaded": True,
        "missing_key_count": len(missing),
        "trainable_missing_key_count": len(trainable_missing),
        "trainable_loaded_key_count": len(trainable_loaded),
        "trainable_missing_keys": trainable_missing[:16],
        "unexpected_key_count": len(unexpected),
        "checkpoint_report": checkpoint.get("report", {}),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required") from exc

    device = torch.device(str(args.device))
    dtype = _dtype(str(args.dtype))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_id), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_ids = _label_token_ids(tokenizer)
    ouro_model = _load_ouro_model(args, dtype=dtype, device=device)
    partial_qwen_layer_indices = parse_int_list(str(args.unfreeze_qwen_layer_indices))
    partial_qwen_requested = bool(
        partial_qwen_layer_indices
        or args.unfreeze_qwen_embeddings
        or args.unfreeze_qwen_lm_head
        or args.unfreeze_qwen_final_norm
    )
    model = QwenBackboneQTRM.from_pretrained(
        str(args.model_id),
        dtype=dtype,
        device=device,
        max_seq_len=int(args.max_seq_len),
        freeze_qwen=not bool(args.train_qwen),
        core_gate_init=float(args.core_gate_init),
        residual_scale=float(args.residual_scale),
        core_impl=str(args.core_impl),
        mandatory_core=bool(args.mandatory_core),
        qwen_core_layer_indices=parse_int_list(str(args.qwen_core_layer_indices)),
        ouro_model=ouro_model,
        ouro_core_layer_indices=parse_int_list(str(args.ouro_core_layer_indices)),
        core_adapter_dim=int(args.core_adapter_dim),
        core_delta_adapter_mode=str(args.core_delta_adapter_mode),
        core_insertion_mode=str(args.core_insertion_mode),
        core_insert_after_layer=int(args.core_insert_after_layer),
        core_residual_gate_mode=str(args.core_residual_gate_mode),
        core_residual_gate_dim=int(args.core_residual_gate_dim),
        core_residual_gate_init=float(args.core_residual_gate_init),
        core_trajectory_carry_mode=str(args.core_trajectory_carry_mode),
        core_trajectory_carry_gate_init=float(args.core_trajectory_carry_gate_init),
        n_core_layers=int(args.n_core_layers),
        h_cycles=int(args.h_cycles),
        l_cycles=int(args.l_cycles),
        outer_steps=int(args.outer_steps),
        delta_backend=str(args.delta_backend),
        strict_backends=bool(args.strict_backends),
        core_convergence_halt_enabled=bool(args.core_convergence_halt_enabled),
        core_convergence_halt_threshold=float(args.core_convergence_halt_threshold),
        core_convergence_halt_min_outer=int(args.core_convergence_halt_min_outer),
        core_step_conditioning_enabled=bool(args.core_step_conditioning_enabled),
        core_step_conditioning_max_steps=int(args.core_step_conditioning_max_steps),
        core_step_conditioning_scale=float(args.core_step_conditioning_scale),
        core_causal=True,
    ).to(device)
    qwen_trainability: dict[str, object]
    if partial_qwen_requested:
        qwen_trainability = model.set_qwen_partial_trainable(
            layer_indices=partial_qwen_layer_indices,
            train_embeddings=bool(args.unfreeze_qwen_embeddings),
            train_lm_head=bool(args.unfreeze_qwen_lm_head),
            train_final_norm=bool(args.unfreeze_qwen_final_norm),
        )
        qwen_train_mode = "partial"
    elif bool(args.train_qwen):
        qwen_trainability = {
            "mode": "all",
            "qwen_trainable_parameters": sum(
                int(parameter.numel())
                for parameter in model.qwen.parameters()
                if parameter.requires_grad
            ),
        }
        qwen_train_mode = "all"
    else:
        model.qwen.eval()
        qwen_trainability = {
            "mode": "frozen",
            "qwen_trainable_parameters": 0,
        }
        qwen_train_mode = "frozen"
    init_checkpoint_info = _load_trainable_checkpoint(model, str(args.init_checkpoint))
    model.eval()

    train_case_mode = str(args.train_case_mode or args.case_mode)
    eval_case_mode = str(args.eval_case_mode or args.case_mode)
    train_cases = build_synthetic_cases(
        count=int(args.train_cases),
        seed=int(args.seed),
        case_mode=train_case_mode,
    )
    eval_cases = build_synthetic_cases(
        count=int(args.eval_cases),
        seed=int(args.seed) + 10000,
        case_mode=eval_case_mode,
    )
    before = evaluate_cases(model, tokenizer, eval_cases, args, label_ids)
    before_language = evaluate_language_non_regression(model, tokenizer, args)
    train_report = train_core(
        model,
        tokenizer,
        train_cases,
        args,
        label_ids,
        eval_cases=eval_cases if int(args.eval_every_steps) > 0 else None,
    )
    model.eval()
    after = evaluate_cases(model, tokenizer, eval_cases, args, label_ids)
    after_language = evaluate_language_non_regression(model, tokenizer, args)
    family_summary = family_gain_summary(after, metric="full_vocab")
    choice_family_summary = family_gain_summary(after, metric="label_choice")
    acceptance_metric = str(args.acceptance_metric)
    acceptance_summary = evaluation_acceptance_summary(after, args)
    accepted_family_summary = acceptance_summary["family_summary"]
    accepted_reasoning = bool(acceptance_summary["accepted_reasoning_gain"])
    accepted_finite_logits = bool(acceptance_summary["accepted_finite_logits"])
    accepted_language = bool(after_language["finite_logits"]) and float(
        after_language["top1_agreement"]
    ) >= float(args.min_language_top1_agreement)
    accepted_family_gain = bool(acceptance_summary["accepted_family_gain"])
    accepted_family_accuracy = bool(acceptance_summary["accepted_family_core_accuracy"])
    core_layer_indices = list(getattr(model.core, "layer_indices", []))
    qwen_core_layers = core_layer_indices if str(args.core_impl) in {
        "qwen_layer_wrapped",
        "qwen_shared_layer_wrapped",
        "ouro_shared_qwen_layer",
    } else []
    ouro_core_layers = core_layer_indices if str(args.core_impl) == "ouro_weight_wrapped" else []
    report = {
        "status": "complete",
        "accepted": bool(accepted_reasoning and accepted_language),
        "accepted_reasoning_gain": bool(accepted_reasoning),
        "accepted_language_non_regression": bool(accepted_language),
        "accepted_family_gain": bool(accepted_family_gain),
        "accepted_family_core_accuracy": bool(accepted_family_accuracy),
        "accepted_finite_logits": bool(accepted_finite_logits),
        "model_id": str(args.model_id),
        "qtrm_native_integrated": True,
        "standalone_graph": True,
        "runtime_donor": False,
        "canonical_path": (
            "chat_template/prompt_tokens -> Qwen3.5 tokenizer/embedding/backbone "
            "-> mandatory QTRM recurrent core -> Qwen3.5 LM head -> AR text"
        ),
        "core_impl": str(args.core_impl),
        "mandatory_core": bool(args.mandatory_core),
        "qwen_trainable": bool(any(parameter.requires_grad for parameter in model.qwen.parameters())),
        "qwen_train_mode": qwen_train_mode,
        "qwen_trainability": qwen_trainability,
        "core_layer_indices": core_layer_indices,
        "qwen_core_layer_indices": qwen_core_layers,
        "ouro_model_id": str(args.ouro_model_id) if ouro_model is not None else "",
        "ouro_core_layer_indices": ouro_core_layers,
        "core_adapter_dim": int(args.core_adapter_dim),
        "core_delta_adapter_mode": str(args.core_delta_adapter_mode),
        "core_insertion_mode": str(args.core_insertion_mode),
        "core_insert_after_layer": int(model.core_insert_after_layer),
        "core_residual_gate_mode": str(args.core_residual_gate_mode),
        "core_residual_gate_dim": int(args.core_residual_gate_dim),
        "core_residual_gate_init": float(args.core_residual_gate_init),
        "core_trajectory_carry_mode": str(args.core_trajectory_carry_mode),
        "core_trajectory_carry_gate_init": float(args.core_trajectory_carry_gate_init),
        "eval_force_trajectory_carry_off": bool(args.eval_force_trajectory_carry_off),
        "residual_scale": float(args.residual_scale),
        "h_cycles": int(args.h_cycles),
        "l_cycles": int(args.l_cycles),
        "outer_steps": int(args.outer_steps),
        "core_convergence_halt_enabled": bool(args.core_convergence_halt_enabled),
        "core_convergence_halt_threshold": float(args.core_convergence_halt_threshold),
        "core_convergence_halt_min_outer": int(args.core_convergence_halt_min_outer),
        "core_step_conditioning_enabled": bool(args.core_step_conditioning_enabled),
        "core_step_conditioning_max_steps": int(args.core_step_conditioning_max_steps),
        "core_step_conditioning_scale": float(args.core_step_conditioning_scale),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "train_cases": int(args.train_cases),
        "eval_cases": int(args.eval_cases),
        "seed": int(args.seed),
        "lr": float(args.lr),
        "qwen_lr": float(args.qwen_lr),
        "weight_decay": float(args.weight_decay),
        "qwen_weight_decay": float(args.qwen_weight_decay),
        "grad_clip": float(args.grad_clip),
        "kl_weight": float(args.kl_weight),
        "language_kl_weight": float(args.language_kl_weight),
        "language_kl_batch_size": int(args.language_kl_batch_size),
        "language_probe_set": str(args.language_probe_set),
        "language_healing_weight": float(args.language_healing_weight),
        "language_healing_kl_weight": float(args.language_healing_kl_weight),
        "language_healing_batch_size": int(args.language_healing_batch_size),
        "language_healing_examples": len(language_healing_examples()),
        "selection_language_weight": float(args.selection_language_weight),
        "selection_min_language_top1": float(args.selection_min_language_top1),
        "core_advantage_weight": float(args.core_advantage_weight),
        "core_advantage_margin": float(args.core_advantage_margin),
        "core_advantage_mode": str(args.core_advantage_mode),
        "checksum_counterfactual_weight": float(args.checksum_counterfactual_weight),
        "checksum_counterfactual_variants": int(args.checksum_counterfactual_variants),
        "checksum_base_error_advantage_weight": float(
            args.checksum_base_error_advantage_weight
        ),
        "checksum_base_error_margin": float(args.checksum_base_error_margin),
        "checksum_base_error_base_margin_threshold": float(
            args.checksum_base_error_base_margin_threshold
        ),
        "checksum_latent_answer_weight": float(args.checksum_latent_answer_weight),
        "checksum_latent_answer_source": str(args.checksum_latent_answer_source),
        "checksum_latent_answer_lr": float(args.checksum_latent_answer_lr),
        "checksum_latent_answer_weight_decay": float(
            args.checksum_latent_answer_weight_decay
        ),
        "checksum_trajectory_weight": float(args.checksum_trajectory_weight),
        "case_mode": str(args.case_mode),
        "train_case_mode": train_case_mode,
        "eval_case_mode": eval_case_mode,
        "acceptance_metric": acceptance_metric,
        "init_checkpoint": init_checkpoint_info,
        "model_report": model.report().__dict__,
        "core_gate_value": float(model.normal_core_gate_value()),
        "before_eval": before,
        "after_eval": after,
        "after_family_summary": family_summary,
        "after_choice_family_summary": choice_family_summary,
        "accepted_family_summary": accepted_family_summary,
        "acceptance_summary": acceptance_summary,
        "before_language": before_language,
        "after_language": after_language,
        "train": train_report,
        "family_loss_weights": parse_float_map(str(args.family_loss_weights)),
        "thresholds": {
            "min_reasoning_gain": float(args.min_reasoning_gain),
            "min_language_top1_agreement": float(args.min_language_top1_agreement),
            "min_family_gain": float(args.min_family_gain),
            "min_family_core_accuracy": float(args.min_family_core_accuracy),
        },
    }
    report["accepted"] = bool(
        accepted_reasoning
        and accepted_language
        and accepted_family_gain
        and accepted_family_accuracy
        and accepted_finite_logits
    )
    if str(args.save_checkpoint):
        torch.save(
            {"model": _trainable_state_dict(model), "report": report},
            str(args.save_checkpoint),
        )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--out-dir", default="local_eval/qwen_backbone_qtrm_core_gate")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument(
        "--core-impl",
        choices=[
            "qwen_layer_wrapped",
            "qwen_shared_layer_wrapped",
            "ouro_shared_qwen_layer",
            "ouro_weight_wrapped",
        ],
        default="qwen_layer_wrapped",
    )
    parser.add_argument("--qwen-core-layer-indices", default="3")
    parser.add_argument("--ouro-model-id", default="ByteDance/Ouro-2.6B-Thinking")
    parser.add_argument("--ouro-core-layer-indices", default="")
    parser.add_argument("--ouro-partial-safetensors", action="store_true")
    parser.add_argument("--train-qwen", action="store_true")
    parser.add_argument("--unfreeze-qwen-layer-indices", default="")
    parser.add_argument("--unfreeze-qwen-embeddings", action="store_true")
    parser.add_argument("--unfreeze-qwen-lm-head", action="store_true")
    parser.add_argument("--unfreeze-qwen-final-norm", action="store_true")
    parser.add_argument("--mandatory-core", action="store_true")
    parser.add_argument("--core-adapter-dim", type=int, default=64)
    parser.add_argument(
        "--core-delta-adapter-mode",
        choices=["add", "adapter_only"],
        default="add",
    )
    parser.add_argument(
        "--core-insertion-mode",
        choices=["final_residual", "mid_layer_suffix"],
        default="final_residual",
    )
    parser.add_argument("--core-insert-after-layer", type=int, default=-1)
    parser.add_argument(
        "--core-residual-gate-mode",
        choices=["constant", "token_mlp"],
        default="constant",
    )
    parser.add_argument("--core-residual-gate-dim", type=int, default=128)
    parser.add_argument("--core-residual-gate-init", type=float, default=-2.0)
    parser.add_argument(
        "--core-trajectory-carry-mode",
        choices=["none", "mean", "learned"],
        default="none",
    )
    parser.add_argument("--core-trajectory-carry-gate-init", type=float, default=0.0)
    parser.add_argument("--eval-force-trajectory-carry-off", action="store_true")
    parser.add_argument("--n-core-layers", type=int, default=1)
    parser.add_argument("--h-cycles", type=int, default=1)
    parser.add_argument("--l-cycles", type=int, default=1)
    parser.add_argument("--outer-steps", type=int, default=1)
    parser.add_argument("--core-convergence-halt-enabled", action="store_true")
    parser.add_argument("--core-convergence-halt-threshold", type=float, default=1.0e-3)
    parser.add_argument("--core-convergence-halt-min-outer", type=int, default=1)
    parser.add_argument("--core-step-conditioning-enabled", action="store_true")
    parser.add_argument("--core-step-conditioning-max-steps", type=int, default=64)
    parser.add_argument("--core-step-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--delta-backend", default="fla_gated_delta")
    parser.add_argument("--strict-backends", action="store_true")
    parser.add_argument("--core-gate-init", type=float, default=-4.0)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-cases", type=int, default=256)
    parser.add_argument("--eval-cases", type=int, default=96)
    parser.add_argument(
        "--case-mode",
        choices=["standard", "hard_v1", "hard_repair_v1", "mixed_v1"],
        default="standard",
    )
    parser.add_argument(
        "--train-case-mode",
        choices=["", "standard", "hard_v1", "hard_repair_v1", "mixed_v1"],
        default="",
    )
    parser.add_argument(
        "--eval-case-mode",
        choices=["", "standard", "hard_v1", "hard_repair_v1", "mixed_v1"],
        default="",
    )
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--qwen-lr", type=float, default=5.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--qwen-weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--language-kl-weight", type=float, default=0.0)
    parser.add_argument("--language-kl-batch-size", type=int, default=2)
    parser.add_argument(
        "--language-probe-set",
        choices=["basic", "extended"],
        default="basic",
    )
    parser.add_argument("--language-healing-weight", type=float, default=0.0)
    parser.add_argument("--language-healing-kl-weight", type=float, default=0.0)
    parser.add_argument("--language-healing-batch-size", type=int, default=2)
    parser.add_argument("--selection-language-weight", type=float, default=0.0)
    parser.add_argument("--selection-min-language-top1", type=float, default=0.0)
    parser.add_argument("--core-advantage-weight", type=float, default=0.0)
    parser.add_argument("--core-advantage-margin", type=float, default=0.0)
    parser.add_argument(
        "--core-advantage-mode",
        choices=["target_logp", "label_choice_margin"],
        default="target_logp",
    )
    parser.add_argument("--family-loss-weights", default="")
    parser.add_argument("--checksum-counterfactual-weight", type=float, default=0.0)
    parser.add_argument("--checksum-counterfactual-variants", type=int, default=1)
    parser.add_argument("--checksum-base-error-advantage-weight", type=float, default=0.0)
    parser.add_argument("--checksum-base-error-margin", type=float, default=0.05)
    parser.add_argument("--checksum-base-error-base-margin-threshold", type=float, default=0.0)
    parser.add_argument("--checksum-latent-answer-weight", type=float, default=0.0)
    parser.add_argument(
        "--checksum-latent-answer-source",
        choices=["z_h", "delta_h"],
        default="z_h",
    )
    parser.add_argument("--checksum-latent-answer-lr", type=float, default=1.0e-3)
    parser.add_argument("--checksum-latent-answer-weight-decay", type=float, default=0.01)
    parser.add_argument("--checksum-trajectory-weight", type=float, default=0.0)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--restore-best-checkpoint", action="store_true")
    parser.add_argument(
        "--acceptance-metric",
        choices=["full_vocab", "label_choice"],
        default="full_vocab",
    )
    parser.add_argument("--min-reasoning-gain", type=float, default=0.05)
    parser.add_argument("--min-language-top1-agreement", type=float, default=0.50)
    parser.add_argument("--min-family-gain", type=float, default=-1.0)
    parser.add_argument("--min-family-core-accuracy", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--save-checkpoint", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not str(args.save_checkpoint):
        args.save_checkpoint = str(out_dir / "last_core.pt")
    report = run(args)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
