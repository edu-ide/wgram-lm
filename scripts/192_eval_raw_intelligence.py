#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

DEFAULT_MODES = [
    "donor_only_no_evidence",
    "qtrm_core_off_no_evidence",
    "qtrm_core_steps_1_no_evidence",
    "qtrm_core_steps_2_no_evidence",
    "qtrm_core_steps_4_no_evidence",
    "qtrm_core_steps_8_no_evidence",
]
FORCED_CHOICE_TIE_EPS = 1.0e-6
FORCED_CHOICE_TIE_COMPLETION = "__FORCED_CHOICE_TIE__"
SCALE_TOKEN_RE = r"\d+(?:p\d+)?"


def _parse_scale_token(token: str) -> float:
    return float(token.replace("p", "."))


def _normalize_answer(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _canonical_answer_text(text: str) -> str:
    answer = str(text).strip()
    if answer.casefold().startswith("answer:"):
        answer = answer.split(":", 1)[1].strip()
    return answer


def _strip_terminal_punctuation(text: str) -> str:
    return text.strip().strip(" \t\r\n.。．:：;；")


def score_answer(
    text: str,
    aliases: Iterable[str],
    *,
    expected_unknown: bool = False,
) -> dict[str, Any]:
    alias_list = [str(alias) for alias in aliases]
    canonical = _canonical_answer_text(text)
    compact = _strip_terminal_punctuation(canonical)
    normalized_text = _normalize_answer(canonical)
    normalized_compact = _normalize_answer(compact)
    normalized_aliases = [
        (alias, _normalize_answer(alias), _normalize_answer(_strip_terminal_punctuation(alias)))
        for alias in alias_list
    ]
    matched_aliases = [
        alias
        for alias, normalized_alias, _ in normalized_aliases
        if normalized_alias and normalized_alias in normalized_text
    ]
    exact_match = any(compact == _strip_terminal_punctuation(alias) for alias in alias_list)
    normalized_exact = any(
        normalized_compact and normalized_compact == normalized_alias_compact
        for _, _, normalized_alias_compact in normalized_aliases
    )
    normalized_contains = bool(matched_aliases)
    unknown_contains = "unknown" in normalized_text
    unknown_exact = normalized_compact == "unknown"
    unknown_correct = bool(expected_unknown and unknown_contains)
    hit = unknown_correct if expected_unknown else normalized_contains
    if expected_unknown and unknown_exact:
        match_type = "unknown_exact"
    elif exact_match:
        match_type = "exact"
    elif normalized_exact:
        match_type = "normalized_exact"
    elif unknown_correct:
        match_type = "unknown_contains"
    elif normalized_contains:
        match_type = "normalized_contains"
    else:
        match_type = "none"

    audit_reasons: list[str] = []
    if hit and normalized_contains and not (exact_match or normalized_exact):
        audit_reasons.append("loose_contains_match")
    if expected_unknown and unknown_correct and not unknown_exact:
        audit_reasons.append("unknown_with_extra_text")
    if not hit:
        audit_reasons.append("answer_miss")
    return {
        "hit": hit,
        "exact_match": exact_match,
        "normalized_exact": normalized_exact,
        "normalized_contains": normalized_contains,
        "unknown_correct": unknown_correct,
        "match_type": match_type,
        "matched_aliases": matched_aliases,
        "canonical_answer": canonical,
        "needs_human_audit": bool(audit_reasons),
        "audit_reasons": audit_reasons,
        "judge_status": "not_run",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run no-retrieval raw-intelligence QTRM eval modes."
    )
    parser.add_argument("--config", default="configs/qwen35_2b_4090.yaml")
    parser.add_argument("--checkpoint", default="runs/qwen35_2b_4090/last.pt")
    parser.add_argument("--cases", default="data/eval/pure_recursive_reasoning_heldout_72.jsonl")
    parser.add_argument(
        "--mode",
        action="append",
        default=None,
        help="Eval mode. Can be repeated. Defaults to donor/core-off/core-depth sweep.",
    )
    parser.add_argument("--out", default="runs/eval/pure_recursive_reasoning_depth_sweep.jsonl")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument(
        "--scoring",
        default="forced_choice",
        choices=["forced_choice", "causal_forced_choice", "generation"],
        help=(
            "forced_choice scores candidate answers by teacher-forced logprob; "
            "causal_forced_choice recomputes each answer-token score from a prefix-only input; "
            "generation uses greedy autoregressive output."
        ),
    )
    parser.add_argument(
        "--choice-score-normalization",
        default="mean",
        choices=["sum", "mean"],
        help=(
            "How to rank forced-choice answers. 'mean' uses per-token average "
            "logprob and avoids a structural bias toward short answers such as EMPTY."
        ),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--qtrm-logits-scale", type=float, default=None)
    parser.add_argument("--donor-logits-scale", type=float, default=None)
    parser.add_argument(
        "--donor-qtrm-conflict-gate",
        action="store_true",
        help="Probe mode: downscale QTRM residual on donor/QTRM top-token conflict.",
    )
    parser.add_argument(
        "--donor-qtrm-conflict-qtrm-scale",
        type=float,
        default=None,
        help="QTRM residual scale used on donor/QTRM top-token conflict when the probe gate is enabled.",
    )
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--suppress-visible-reasoning-tokens", action="store_true")
    return parser


def resolve_modes(args: argparse.Namespace) -> list[str]:
    return list(args.mode or DEFAULT_MODES)


def apply_eval_model_overrides(model_cfg, args: argparse.Namespace) -> None:
    if bool(getattr(args, "donor_qtrm_conflict_gate", False)):
        model_cfg.donor_qtrm_conflict_gate_enabled = True
    conflict_scale = getattr(args, "donor_qtrm_conflict_qtrm_scale", None)
    if conflict_scale is not None:
        model_cfg.donor_qtrm_conflict_qtrm_scale = float(conflict_scale)


def mode_runtime(mode: str) -> dict[str, Any]:
    if mode == "donor_only_no_evidence":
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": None,
            "qtrm_logits_scale": 0.0,
            "donor_logits_scale": 1.0,
            "memoryos_used": False,
            "retrieval_used": False,
        }
    if mode == "qtrm_core_off_no_evidence":
        return {
            "mode": mode,
            "disable_core": True,
            "core_steps_override": None,
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
        }
    if mode in {"qtrm_core_off_qtrm_only_no_evidence", "qtrm_core_off_low_donor_no_evidence"}:
        return {
            "mode": mode,
            "disable_core": True,
            "core_steps_override": None,
            "qtrm_logits_scale": 1.0,
            "donor_logits_scale": 0.0 if "qtrm_only" in mode else 0.25,
            "memoryos_used": False,
            "retrieval_used": False,
        }
    match = re.fullmatch(r"qtrm_core_steps_(\d+)_no_evidence", mode)
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
        }
    match = re.fullmatch(r"qtrm_core_steps_(\d+)_temporal_spatial_off_no_evidence", mode)
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_temporal_spatial_context": True,
        }
    match = re.fullmatch(r"qtrm_core_steps_(\d+)_transition_state_off_no_evidence", mode)
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_transition_state": True,
        }
    match = re.fullmatch(
        rf"qtrm_core_steps_(\d+)_donor_scale_({SCALE_TOKEN_RE})_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": 1.0,
            "donor_logits_scale": _parse_scale_token(match.group(2)),
            "memoryos_used": False,
            "retrieval_used": False,
        }
    match = re.fullmatch(
        rf"qtrm_core_steps_(\d+)_qtrm_scale_({SCALE_TOKEN_RE})_donor_scale_({SCALE_TOKEN_RE})_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": _parse_scale_token(match.group(2)),
            "donor_logits_scale": _parse_scale_token(match.group(3)),
            "memoryos_used": False,
            "retrieval_used": False,
        }
    match = re.fullmatch(r"qtrm_core_steps_(\d+)_(low_donor|qtrm_only)_no_evidence", mode)
    if match:
        scale_mode = match.group(2)
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": 1.0,
            "donor_logits_scale": 0.0 if scale_mode == "qtrm_only" else 0.25,
            "memoryos_used": False,
            "retrieval_used": False,
        }
    raise ValueError(f"unknown raw-intelligence eval mode: {mode}")


def load_cases(path: str | Path, *, max_cases: int | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if not case.get("id"):
                case["id"] = f"case-{line_no}"
            if not case.get("prompt") and not case.get("question"):
                raise ValueError(f"{path}:{line_no}: missing prompt/question")
            if not case.get("answer_aliases"):
                raise ValueError(f"{path}:{line_no}: missing answer_aliases")
            if case.get("evidence"):
                raise ValueError(f"{path}:{line_no}: raw-intelligence cases must not include evidence")
            cases.append(case)
            if max_cases is not None and len(cases) >= int(max_cases):
                break
    return cases


def _case_temporal_spatial_context(case: dict[str, Any], *, device: str):
    value = case.get("temporal_spatial_context")
    if value is None:
        return None
    import torch

    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        return tensor.view(1, -1)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3 and int(tensor.shape[0]) == 1:
        return tensor
    raise ValueError(
        "temporal_spatial_context must be a vector, token list, or single-batch token list"
    )


def _case_temporal_spatial_context_token_count(case: dict[str, Any]) -> int:
    value = case.get("temporal_spatial_context")
    if value is None:
        return 0
    if not isinstance(value, list):
        return 1
    if not value:
        return 0
    first = value[0]
    if isinstance(first, list):
        return len(value)
    return 1


def score_case_record(
    case: dict[str, Any],
    *,
    mode: str,
    completion: str,
    runtime: dict[str, Any],
    generated_tokens: int,
) -> dict[str, Any]:
    score = score_answer(
        completion,
        case.get("answer_aliases", []),
        expected_unknown=bool(case.get("expected_unknown", False)),
    )
    disable_temporal_spatial_context = bool(
        runtime.get("disable_temporal_spatial_context", False)
    )
    temporal_spatial_context_available = case.get("temporal_spatial_context") is not None
    temporal_spatial_context_token_count = (
        0
        if disable_temporal_spatial_context
        else _case_temporal_spatial_context_token_count(case)
    )
    return {
        "id": case.get("id"),
        "mode": mode,
        "raw_intelligence_axis": case.get("raw_intelligence_axis", "pure_recursive_reasoning"),
        "category": case.get("category", "uncategorized"),
        "task_family": case.get("task_family", case.get("category", "uncategorized")),
        "reasoning_family": case.get("reasoning_family", case.get("task_family", case.get("category", "uncategorized"))),
        "expected_paradigm": case.get("expected_paradigm", "unknown"),
        "requires_stochasticity": bool(case.get("requires_stochasticity", False)),
        "parallel_depth_estimate": case.get("parallel_depth_estimate"),
        "serial_trace_length_estimate": case.get("serial_trace_length_estimate"),
        "question": case.get("question", ""),
        "prompt": case.get("prompt") or case.get("question", ""),
        "answer_aliases": case.get("answer_aliases", []),
        "expected_unknown": bool(case.get("expected_unknown", False)),
        "completion": completion,
        "generated_tokens": int(generated_tokens),
        "core_steps_requested": runtime.get("core_steps_override"),
        "disable_core": bool(runtime.get("disable_core", False)),
        "memoryos_used": False,
        "retrieval_used": False,
        "evidence_token_count": 0,
        "workspace_memory_token_count": 0,
        "temporal_spatial_context_available": temporal_spatial_context_available,
        "disable_temporal_spatial_context": disable_temporal_spatial_context,
        "temporal_spatial_context_token_count": temporal_spatial_context_token_count,
        "disable_transition_state": bool(runtime.get("disable_transition_state", False)),
        **score,
    }


def _choice_candidates(case: dict[str, Any]) -> list[str]:
    choices = [str(choice) for choice in case.get("choices", []) if str(choice).strip()]
    aliases = [str(alias) for alias in case.get("answer_aliases", []) if str(alias).strip()]
    if not choices:
        choices = aliases
    for alias in aliases:
        if alias not in choices:
            choices.insert(0, alias)
    return choices


def _choice_token_count(tokenizer, choice: str) -> int:
    if tokenizer is None:
        return 1
    token_ids = tokenizer.encode(f" {choice}", add_special_tokens=False)
    if not token_ids:
        token_ids = tokenizer.encode(str(choice), add_special_tokens=False)
    return max(1, len(token_ids))


def _normalized_choice_score(logprob_sum: float, token_count: int, normalization: str) -> float:
    mode = str(normalization or "sum").lower()
    if mode == "sum":
        return float(logprob_sum)
    if mode == "mean":
        return float(logprob_sum) / max(1, int(token_count))
    raise ValueError("choice score normalization must be 'sum' or 'mean'")


def _no_repeat_ngram_banned_tokens(generated: list[int], prompt_len: int, ngram_size: int) -> list[int]:
    n = int(ngram_size)
    if n <= 0:
        return []
    completion = generated[prompt_len:]
    if n == 1:
        return sorted(set(completion))
    if len(completion) < n - 1:
        return []
    prefix = tuple(completion[-(n - 1) :])
    banned: set[int] = set()
    for idx in range(0, len(completion) - n + 1):
        ngram = tuple(completion[idx : idx + n])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])
    return sorted(banned)


def _visible_reasoning_token_ids(tokenizer, *, enabled: bool) -> list[int]:
    if not enabled:
        return []
    ids: list[int] = []
    for marker in ("<think>", "</think>"):
        try:
            ids.extend(int(token_id) for token_id in tokenizer.encode(marker, add_special_tokens=False))
        except Exception:
            continue
    return sorted(set(ids))


def _completion_text(tokenizer, generated: list[int], *, prompt_len: int) -> str:
    full_text = tokenizer.decode(generated, skip_special_tokens=True)
    prompt_text = tokenizer.decode(generated[:prompt_len], skip_special_tokens=True)
    if prompt_text and full_text.startswith(prompt_text):
        return full_text[len(prompt_text) :].strip()
    return tokenizer.decode(generated[prompt_len:], skip_special_tokens=True).strip()


def _select_device(cfg_device: str, requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() and cfg_device in {"auto", "cuda"} else "cpu"


def _prepare_inputs(tokenizer, text: str, max_length: int, device: str):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    return {k: v.to(device) for k, v in enc.items()}


def _causal_choice_prefixes(
    tokenizer,
    prompt: str,
    choice: str,
    *,
    max_length: int,
    device: str,
):
    import torch

    prompt_inputs = _prepare_inputs(tokenizer, prompt, max_length, device)
    prompt_ids = prompt_inputs["input_ids"][0].detach().cpu().tolist()
    choice_ids = tokenizer.encode(f" {choice}", add_special_tokens=False)
    if not choice_ids:
        choice_ids = tokenizer.encode(str(choice), add_special_tokens=False)
    prefixes = []
    for pos, target_id in enumerate(choice_ids):
        prefix_ids = prompt_ids + [int(token_id) for token_id in choice_ids[:pos]]
        if len(prefix_ids) > int(max_length):
            break
        input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        prefixes.append((input_ids, attention_mask, int(target_id)))
    return prefixes


def _donor_kwargs(donor, input_ids, attention_mask, device: str, *, return_logits: bool):
    if donor is None:
        return {}
    encoded = donor.encode_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_logits=return_logits,
    )
    out = {"text_states": encoded["text_states"].to(device)}
    if return_logits and encoded.get("logits") is not None:
        out["donor_logits"] = encoded["logits"].to(device)
    return out


def _record_conflict_gate_mean(
    telemetry: dict[str, Any] | None,
    outputs: dict[str, Any],
    *,
    start: int | None = None,
    end: int | None = None,
) -> None:
    if telemetry is None:
        return
    gate = outputs.get("donor_qtrm_conflict_gate")
    if gate is None or getattr(gate, "numel", lambda: 0)() == 0:
        return
    gate_slice = gate
    if start is not None or end is not None:
        gate_slice = gate[:, start:end]
    if gate_slice.numel() == 0:
        return
    telemetry.setdefault("donor_qtrm_conflict_gate_mean_values", []).append(
        float(gate_slice.float().mean().detach().cpu().item())
    )


def _finalize_choice_telemetry(telemetry: dict[str, Any]) -> dict[str, Any]:
    values = [
        float(value)
        for value in telemetry.get("donor_qtrm_conflict_gate_mean_values", [])
    ]
    if not values:
        return {}
    return {
        "donor_qtrm_conflict_gate_mean": sum(values) / len(values),
        "donor_qtrm_conflict_gate_observations": len(values),
    }


def _answer_choice_logprob(
    model,
    donor,
    tokenizer,
    prompt: str,
    choice: str,
    *,
    runtime: dict[str, Any],
    max_length: int,
    device: str,
    telemetry: dict[str, Any] | None = None,
    temporal_spatial_context=None,
) -> float:
    import torch

    full_text = f"{prompt} {choice}"
    prompt_inputs = _prepare_inputs(tokenizer, prompt, max_length, device)
    inputs = _prepare_inputs(tokenizer, full_text, max_length, device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    full_len = int(input_ids.shape[1])
    if full_len <= prompt_len:
        return float("-inf")

    old_outer_steps = int(model.cfg.outer_steps)
    if runtime.get("core_steps_override") is not None:
        model.cfg.outer_steps = int(runtime["core_steps_override"])
    try:
        extra = _donor_kwargs(
            donor,
            input_ids,
            attention_mask,
            device,
            return_logits=bool(model.cfg.donor_logits_scale != 0.0),
        )
        with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                **extra,
                disable_core=bool(runtime.get("disable_core", False)),
                temporal_spatial_context=temporal_spatial_context,
                disable_temporal_spatial_context=bool(
                    runtime.get("disable_temporal_spatial_context", False)
                ),
                disable_transition_state=bool(runtime.get("disable_transition_state", False)),
            )
        logits = outputs["logits"].float()
        offset = logits.shape[1] - input_ids.shape[1]
        aligned = logits[:, offset + prompt_len - 1 : offset + full_len - 1, :]
        targets = input_ids[:, prompt_len:full_len].to(device=aligned.device)
        if aligned.shape[1] != targets.shape[1]:
            return float("-inf")
        _record_conflict_gate_mean(
            telemetry,
            outputs,
            start=prompt_len - 1,
            end=full_len - 1,
        )
        log_probs = torch.log_softmax(aligned, dim=-1)
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(token_log_probs.sum().detach().cpu().item())
    finally:
        model.cfg.outer_steps = old_outer_steps


def _answer_choice_causal_logprob(
    model,
    donor,
    tokenizer,
    prompt: str,
    choice: str,
    *,
    runtime: dict[str, Any],
    max_length: int,
    device: str,
    telemetry: dict[str, Any] | None = None,
    temporal_spatial_context=None,
) -> float:
    import torch

    prefixes = _causal_choice_prefixes(
        tokenizer,
        prompt,
        choice,
        max_length=max_length,
        device=device,
    )
    if not prefixes:
        return float("-inf")

    old_outer_steps = int(model.cfg.outer_steps)
    if runtime.get("core_steps_override") is not None:
        model.cfg.outer_steps = int(runtime["core_steps_override"])
    total = 0.0
    try:
        for input_ids, attention_mask, target_id in prefixes:
            extra = _donor_kwargs(
                donor,
                input_ids,
                attention_mask,
                device,
                return_logits=bool(model.cfg.donor_logits_scale != 0.0),
            )
            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    **extra,
                    disable_core=bool(runtime.get("disable_core", False)),
                    temporal_spatial_context=temporal_spatial_context,
                    disable_temporal_spatial_context=bool(
                        runtime.get("disable_temporal_spatial_context", False)
                    ),
                    disable_transition_state=bool(runtime.get("disable_transition_state", False)),
                )
            next_logits = outputs["logits"][:, -1, :].float()
            _record_conflict_gate_mean(telemetry, outputs, start=-1, end=None)
            total += float(
                torch.log_softmax(next_logits, dim=-1)[0, int(target_id)]
                .detach()
                .cpu()
                .item()
            )
    finally:
        model.cfg.outer_steps = old_outer_steps
    return total


def _forced_choice_case(
    model,
    donor,
    tokenizer,
    case: dict[str, Any],
    *,
    runtime: dict[str, Any],
    max_length: int,
    device: str,
    choice_score_normalization: str = "sum",
) -> tuple[str, list[dict[str, Any]]]:
    prompt = case.get("prompt") or case.get("question", "")
    temporal_spatial_context = _case_temporal_spatial_context(case, device=device)
    scored = []
    for choice in _choice_candidates(case):
        telemetry: dict[str, Any] = {"donor_qtrm_conflict_gate_mean_values": []}
        logprob_sum = _answer_choice_logprob(
            model,
            donor,
            tokenizer,
            prompt,
            choice,
            runtime=runtime,
            max_length=max_length,
            device=device,
            telemetry=telemetry,
            temporal_spatial_context=temporal_spatial_context,
        )
        token_count = _choice_token_count(tokenizer, choice)
        score = _normalized_choice_score(
            logprob_sum,
            token_count,
            choice_score_normalization,
        )
        scored.append(
            {
                "choice": choice,
                "logprob": score,
                "logprob_sum": logprob_sum,
                "token_count": token_count,
                "score_normalization": choice_score_normalization,
                **_finalize_choice_telemetry(telemetry),
            }
        )
    scored.sort(key=lambda item: float(item["logprob"]), reverse=True)
    if not scored:
        return "", []
    best = float(scored[0]["logprob"])
    for row in scored:
        row["tied_for_best"] = abs(float(row["logprob"]) - best) <= FORCED_CHOICE_TIE_EPS
    if sum(1 for row in scored if bool(row["tied_for_best"])) > 1:
        return FORCED_CHOICE_TIE_COMPLETION, scored
    return str(scored[0]["choice"]), scored


def _causal_forced_choice_case(
    model,
    donor,
    tokenizer,
    case: dict[str, Any],
    *,
    runtime: dict[str, Any],
    max_length: int,
    device: str,
    choice_score_normalization: str = "sum",
) -> tuple[str, list[dict[str, Any]]]:
    prompt = case.get("prompt") or case.get("question", "")
    temporal_spatial_context = _case_temporal_spatial_context(case, device=device)
    scored = []
    for choice in _choice_candidates(case):
        telemetry: dict[str, Any] = {"donor_qtrm_conflict_gate_mean_values": []}
        logprob_sum = _answer_choice_causal_logprob(
            model,
            donor,
            tokenizer,
            prompt,
            choice,
            runtime=runtime,
            max_length=max_length,
            device=device,
            telemetry=telemetry,
            temporal_spatial_context=temporal_spatial_context,
        )
        token_count = _choice_token_count(tokenizer, choice)
        score = _normalized_choice_score(
            logprob_sum,
            token_count,
            choice_score_normalization,
        )
        scored.append(
            {
                "choice": choice,
                "logprob": score,
                "logprob_sum": logprob_sum,
                "token_count": token_count,
                "score_normalization": choice_score_normalization,
                **_finalize_choice_telemetry(telemetry),
            }
        )
    scored.sort(key=lambda item: float(item["logprob"]), reverse=True)
    if not scored:
        return "", []
    best = float(scored[0]["logprob"])
    for row in scored:
        row["tied_for_best"] = abs(float(row["logprob"]) - best) <= FORCED_CHOICE_TIE_EPS
    if sum(1 for row in scored if bool(row["tied_for_best"])) > 1:
        return FORCED_CHOICE_TIE_COMPLETION, scored
    return str(scored[0]["choice"]), scored


def _generate_case(
    model,
    donor,
    tokenizer,
    prompt: str,
    *,
    runtime: dict[str, Any],
    max_length: int,
    max_new_tokens: int,
    device: str,
    no_repeat_ngram_size: int,
    suppressed_token_ids: Iterable[int],
    temporal_spatial_context=None,
) -> tuple[str, int]:
    import torch

    inputs = _prepare_inputs(tokenizer, prompt, max_length, device)
    generated = inputs["input_ids"][0].detach().cpu().tolist()
    prompt_len = len(generated)
    old_outer_steps = int(model.cfg.outer_steps)
    if runtime.get("core_steps_override") is not None:
        model.cfg.outer_steps = int(runtime["core_steps_override"])
    try:
        for _ in range(max_new_tokens):
            cur_ids = torch.tensor([generated], dtype=torch.long, device=device)
            cur_mask = torch.ones_like(cur_ids)
            extra = _donor_kwargs(
                donor,
                cur_ids,
                cur_mask,
                device,
                return_logits=bool(model.cfg.donor_logits_scale != 0.0),
            )
            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
                outputs = model(
                    cur_ids,
                    attention_mask=cur_mask,
                    **extra,
                    disable_core=bool(runtime.get("disable_core", False)),
                    temporal_spatial_context=temporal_spatial_context,
                    disable_temporal_spatial_context=bool(
                        runtime.get("disable_temporal_spatial_context", False)
                    ),
                    disable_transition_state=bool(runtime.get("disable_transition_state", False)),
                )
            logits = outputs["logits"][0, -1].float()
            banned = set(int(token_id) for token_id in suppressed_token_ids)
            banned.update(_no_repeat_ngram_banned_tokens(generated, prompt_len, no_repeat_ngram_size))
            if banned:
                valid = [token_id for token_id in banned if 0 <= token_id < logits.shape[-1]]
                if valid:
                    logits[torch.tensor(valid, device=logits.device, dtype=torch.long)] = -torch.inf
            next_id = int(logits.argmax(dim=-1).detach().cpu().item())
            if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
                break
            generated.append(next_id)
    finally:
        model.cfg.outer_steps = old_outer_steps
    return _completion_text(tokenizer, generated, prompt_len=prompt_len), len(generated) - prompt_len


def run_eval(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoTokenizer

    from qtrm_mm.config import load_config
    from qtrm_mm.qtrm_model import QTRMMultimodalModel
    from qtrm_mm.qwen_donor import QwenDonorAdapter

    cfg = load_config(args.config)
    apply_eval_model_overrides(cfg.model, args)
    if not cfg.donor.model_id:
        raise SystemExit("donor.model_id is required")
    device = _select_device(cfg.train.device, args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.donor.model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = QTRMMultimodalModel(cfg.model)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model", state), strict=False)
    model = model.to(device).eval()

    donor = QwenDonorAdapter(cfg.donor)
    max_length = args.max_length or cfg.train.seq_len
    cases = load_cases(args.cases, max_cases=args.max_cases)
    suppressed_token_ids = _visible_reasoning_token_ids(
        tokenizer,
        enabled=bool(args.suppress_visible_reasoning_tokens),
    )

    records: list[dict[str, Any]] = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f, torch.no_grad(), redirect_stdout(sys.stderr):
        for mode in resolve_modes(args):
            runtime = mode_runtime(mode)
            old_qtrm_scale = float(model.cfg.qtrm_logits_scale)
            old_donor_scale = float(model.cfg.donor_logits_scale)
            model.cfg.qtrm_logits_scale = (
                float(runtime["qtrm_logits_scale"])
                if runtime["qtrm_logits_scale"] is not None
                else float(args.qtrm_logits_scale)
                if args.qtrm_logits_scale is not None
                else old_qtrm_scale
            )
            model.cfg.donor_logits_scale = (
                float(runtime["donor_logits_scale"])
                if runtime["donor_logits_scale"] is not None
                else float(args.donor_logits_scale)
                if args.donor_logits_scale is not None
                else old_donor_scale
            )
            try:
                for case in cases:
                    prompt = case.get("prompt") or case.get("question", "")
                    choice_scores = None
                    if args.scoring == "forced_choice":
                        completion, choice_scores = _forced_choice_case(
                            model,
                            donor,
                            tokenizer,
                            case,
                            runtime=runtime,
                            max_length=max_length,
                            device=device,
                            choice_score_normalization=args.choice_score_normalization,
                        )
                        generated_tokens = 0
                    elif args.scoring == "causal_forced_choice":
                        completion, choice_scores = _causal_forced_choice_case(
                            model,
                            donor,
                            tokenizer,
                            case,
                            runtime=runtime,
                            max_length=max_length,
                            device=device,
                            choice_score_normalization=args.choice_score_normalization,
                        )
                        generated_tokens = 0
                    else:
                        temporal_spatial_context = _case_temporal_spatial_context(
                            case,
                            device=device,
                        )
                        completion, generated_tokens = _generate_case(
                            model,
                            donor,
                            tokenizer,
                            prompt,
                            runtime=runtime,
                            max_length=max_length,
                            max_new_tokens=args.max_new_tokens,
                            device=device,
                            no_repeat_ngram_size=args.no_repeat_ngram_size,
                            suppressed_token_ids=suppressed_token_ids,
                            temporal_spatial_context=temporal_spatial_context,
                        )
                    record = score_case_record(
                        case,
                        mode=mode,
                        completion=completion,
                        runtime=runtime,
                        generated_tokens=generated_tokens,
                    )
                    record["scoring"] = args.scoring
                    record["choice_score_normalization"] = args.choice_score_normalization
                    if choice_scores is not None:
                        record["choice_scores"] = choice_scores
                        record["choice_tied"] = (
                            sum(1 for row in choice_scores if bool(row.get("tied_for_best")))
                            > 1
                        )
                    records.append(record)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
            finally:
                model.cfg.qtrm_logits_scale = old_qtrm_scale
                model.cfg.donor_logits_scale = old_donor_scale
    return records


def main() -> None:
    args = build_arg_parser().parse_args()
    records = run_eval(args)
    hits = sum(1 for record in records if bool(record.get("hit")))
    print(f"wrote {len(records)} records to {args.out}")
    print(f"hits={hits}/{len(records)}")


if __name__ == "__main__":
    main()
