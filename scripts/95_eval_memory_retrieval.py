#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence

import torch

from wgram_lm.config import load_config
from wgram_lm.eval.memory_retrieval import (
    audit_records,
    build_case_prompt_and_workspace_memory,
    case_task_family,
    canonical_answer_text,
    expected_unknown_case,
    filter_results_for_case,
    expand_linked_evidence_results,
    govern_evidence_sources,
    load_cases,
    select_evidence_results,
    score_answer,
    summarize_records,
    target_retrieval_stats,
    target_retrieved,
)
from wgram_lm.eval.ssot_contract import (
    CANONICAL_ANSWER_CHANNEL,
    CANONICAL_EVIDENCE_INJECTION,
    validate_canonical_model_config as validate_canonical_model_contract,
    validate_canonical_ssot_args as validate_canonical_ssot_contract,
)
from wgram_lm.history import append_jsonl, eval_record_to_history_row, resolve_history_path
from wgram_lm.wgram_model import QTRMMultimodalModel
from wgram_lm.qwen_donor import QwenDonorAdapter


DEFAULT_MODES = [
    "donor_only_with_evidence",
    "qtrm_residual_with_evidence",
    "qtrm_workspace_off_with_evidence",
    "qtrm_core_off_with_evidence",
    "qtrm_coda_off_with_evidence",
    "qtrm_residual_head_off_with_evidence",
    "qtrm_donor_hidden_off_with_evidence",
    "qtrm_workspace_only_with_evidence",
    "qtrm_workspace_gate_off_with_evidence",
    "qtrm_workspace_memory_off_with_evidence",
    "qtrm_core_context_off_with_evidence",
    "qtrm_core_to_text_off_with_evidence",
    "qtrm_evidence_bottleneck_off_with_evidence",
    "qtrm_evidence_span_reader_off_with_evidence",
    "qtrm_answer_residual_governor_off_with_evidence",
    "qtrm_answer_decision_features_off_with_evidence",
    "qtrm_answer_decision_off_with_evidence",
    "donor_only_no_evidence",
    "qtrm_residual_no_evidence",
]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Probe whether QTRM residual generation uses provided MemoryOS evidence."
    )
    ap.add_argument("--config", default="configs/qwen35_2b_4090_donor_residual_s010_1000.yaml")
    ap.add_argument("--checkpoint", default="runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt")
    ap.add_argument("--cases", default="data/eval/memory_retrieval_probe.jsonl")
    ap.add_argument("--mode", action="append", default=None, help="Mode to run. Can be repeated.")
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0)
    ap.add_argument(
        "--suppress-visible-reasoning-tokens",
        action="store_true",
        help="Suppress common visible reasoning tokens such as <think> during generation.",
    )
    ap.add_argument(
        "--short-answer-governor",
        action="store_true",
        help="Score and report only the first short answer line while preserving raw_completion.",
    )
    ap.add_argument(
        "--answer-channel",
        default="greedy",
        choices=["greedy", "evidence_span_copy"],
        help=(
            "greedy=autoregressive token generation; evidence_span_copy=run the "
            "prompt-conditioned evidence span reader once and answer only with "
            "the selected hidden-workspace span or UNKNOWN."
        ),
    )
    ap.add_argument(
        "--evidence-span-max-tokens",
        type=int,
        default=16,
        help="Maximum copied workspace span length for --answer-channel evidence_span_copy.",
    )
    ap.add_argument(
        "--evidence-span-no-answer-threshold",
        type=float,
        default=0.5,
        help=(
            "Sigmoid threshold on evidence_span_no_answer_logits above which the "
            "span-copy answer channel returns UNKNOWN."
        ),
    )
    ap.add_argument(
        "--evidence-span-min-score",
        type=float,
        default=None,
        help=(
            "Optional minimum start+end span score for evidence_span_copy. "
            "Lower-confidence spans return UNKNOWN."
        ),
    )
    ap.add_argument(
        "--answer-revision",
        default="none",
        choices=["none", "evidence_span_boundary"],
        help=(
            "Optional deterministic REVISE branch after evidence_span_copy. "
            "evidence_span_boundary expands truncated atomic identifiers inside "
            "the already encoded evidence workspace."
        ),
    )
    ap.add_argument("--answer-revision-max-left-tokens", type=int, default=2)
    ap.add_argument("--answer-revision-max-right-tokens", type=int, default=2)
    ap.add_argument(
        "--truth-gate",
        action="store_true",
        help=(
            "For evidence_span_copy, require logical evidence heads to allow "
            "copying: support/causal high and refute/missing low."
        ),
    )
    ap.add_argument("--truth-support-threshold", type=float, default=0.5)
    ap.add_argument("--truth-causal-threshold", type=float, default=0.5)
    ap.add_argument("--truth-refute-threshold", type=float, default=0.5)
    ap.add_argument("--truth-missing-threshold", type=float, default=0.5)
    ap.add_argument(
        "--answer-decision-checkpoint",
        default="",
        help=(
            "Optional post-hoc answer-decision head checkpoint from "
            "scripts/161_train_answer_decision_head.py. If it predicts block, "
            "the final answer is changed to UNKNOWN."
        ),
    )
    ap.add_argument(
        "--answer-decision-threshold",
        type=float,
        default=None,
        help="Override the threshold stored in --answer-decision-checkpoint.",
    )
    ap.add_argument(
        "--model-answer-decision",
        action="store_true",
        help=(
            "Use QTRM output answer_decision_logits as an in-model blocker. "
            "Requires model.answer_decision_head_enabled=true in the checkpoint."
        ),
    )
    ap.add_argument(
        "--model-answer-decision-threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for --model-answer-decision.",
    )
    ap.add_argument("--memory-max-chars", type=int, default=2000)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument(
        "--evidence-mode",
        default="target",
        choices=["target", "all", "lexical", "memoryos"],
        help=(
            "target=gold evidence only, all=target+distractors in file order, "
            "lexical=rank target+distractors by query overlap, memoryos=retrieve from --memory-index."
        ),
    )
    ap.add_argument(
        "--evidence-injection",
        default=CANONICAL_EVIDENCE_INJECTION,
        choices=["ssot", "prompt", "workspace", "dual"],
        help=(
            "ssot=canonical path: compile retrieved evidence into one donor-visible "
            "chat-template token stream and use token-aligned masks; prompt=legacy "
            "visible-evidence alias; workspace=ablation path that hides evidence "
            "from the prompt and encodes it as workspace-side memory states; "
            "dual=legacy/probe path that does both."
        ),
    )
    ap.add_argument("--retrieval-top-k", type=int, default=3)
    ap.add_argument(
        "--memory-link-expansion",
        type=int,
        default=0,
        help="Append up to N case-scoped records named by already selected evidence.",
    )
    ap.add_argument("--memory-index", default=None)
    ap.add_argument("--memory-model-id", default=None)
    ap.add_argument("--memory-backend", default=None)
    ap.add_argument("--hnsw-ef-search", type=int, default=None)
    ap.add_argument("--retrieve-top-n", type=int, default=None)
    ap.add_argument("--rerank-backend", default="none", choices=["none", "lexical", "cross_encoder"])
    ap.add_argument("--reranker-model-id", default="Qwen/Qwen3-Reranker-0.6B")
    ap.add_argument("--reranker-device", default=None)
    ap.add_argument(
        "--evidence-source-governor",
        default="none",
        choices=["none", "reliability"],
        help=(
            "Optional non-label evidence source governor before prompt/workspace formatting. "
            "reliability prefers signed/current/latest sources and prunes anonymous/stale/decoy records."
        ),
    )
    ap.add_argument(
        "--evidence-source-selector-checkpoint",
        default="",
        help=(
            "Optional learned evidence source selector checkpoint from "
            "scripts/165_train_evidence_source_selector.py."
        ),
    )
    ap.add_argument(
        "--evidence-source-selector-threshold",
        type=float,
        default=None,
        help="Override the threshold stored in --evidence-source-selector-checkpoint.",
    )
    ap.add_argument(
        "--evidence-source-selector-mode",
        default="span_mask",
        choices=["metadata", "span_mask"],
        help=(
            "metadata=score sources only; span_mask=preserve full workspace context "
            "but restrict the copied answer span to selected source text tokens."
        ),
    )
    ap.add_argument(
        "--memoryos-global",
        action="store_true",
        help="Do not case-filter MemoryOS retrieval results. Useful for real global-memory tests.",
    )
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--qtrm-logits-scale", type=float, default=None)
    ap.add_argument("--donor-logits-scale", type=float, default=1.0)
    ap.add_argument(
        "--core-halt-mode",
        default="config",
        choices=["config", "enabled", "disabled"],
        help=(
            "Control recursive-core early halt during eval. config preserves the "
            "checkpoint config default, enabled forces early halt, disabled forces full depth."
        ),
    )
    ap.add_argument("--no-logit-shift", action="store_true")
    ap.add_argument(
        "--require-canonical-ssot",
        action="store_true",
        help=(
            "Fail fast unless this eval uses the canonical user-facing path: "
            f"--evidence-injection {CANONICAL_EVIDENCE_INJECTION} and "
            f"--answer-channel {CANONICAL_ANSWER_CHANNEL}. This keeps "
            "workspace/dual hidden-evidence and span-copy probes from being "
            "reported as the main QTRM answer architecture."
        ),
    )
    ap.add_argument("--jsonl-out", default="runs/eval/memory_retrieval_probe.jsonl")
    ap.add_argument(
        "--history-jsonl-out",
        default="auto",
        help=(
            "Append per-generation eval history. Use auto for "
            "runs/history/evals/YYYY-MM-DD.jsonl, or none/off to disable."
        ),
    )
    ap.add_argument(
        "--audit-jsonl-out",
        default=None,
        help="Optional JSONL path for human/LLM-judge audit items from ambiguous records.",
    )
    ap.add_argument("--print-completions", action="store_true")
    return ap


def resolve_qtrm_scale(config_scale: float, override: float | None) -> float:
    return float(config_scale if override is None else override)


def select_device(cfg_device: str, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() and cfg_device in {"auto", "cuda"} else "cpu"


def load_qtrm(config_path: str, checkpoint_path: str, device: str) -> QTRMMultimodalModel:
    cfg = load_config(config_path)
    model = QTRMMultimodalModel(cfg.model)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")
    return model.to(device).eval()


def prepare_inputs(tokenizer, text: str, max_length: int, device: str) -> dict[str, torch.Tensor]:
    original_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=True,
        )
    finally:
        tokenizer.truncation_side = original_side
    return {k: v.to(device) for k, v in enc.items()}


def apply_token_suppression(
    logits: torch.Tensor,
    suppressed_token_ids: Iterable[int] | None,
) -> torch.Tensor:
    ids = [int(token_id) for token_id in (suppressed_token_ids or []) if 0 <= int(token_id) < logits.shape[-1]]
    if not ids:
        return logits
    filtered = logits.float().clone()
    filtered[torch.tensor(ids, device=filtered.device, dtype=torch.long)] = -torch.inf
    return filtered


def no_repeat_ngram_banned_tokens(generated: Sequence[int], ngram_size: int) -> list[int]:
    n = int(ngram_size)
    if n <= 0 or len(generated) < n - 1:
        return []
    if n == 1:
        return sorted(set(int(token_id) for token_id in generated))
    prefix = tuple(int(token_id) for token_id in generated[-(n - 1) :])
    banned: set[int] = set()
    for idx in range(0, len(generated) - n + 1):
        ngram = tuple(int(token_id) for token_id in generated[idx : idx + n])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])
    return sorted(banned)


def select_next_token(
    logits: torch.Tensor,
    *,
    suppressed_token_ids: Iterable[int] | None = None,
) -> int:
    filtered = apply_token_suppression(logits.float(), suppressed_token_ids)
    return int(filtered.argmax(dim=-1).detach().cpu().item())


def visible_reasoning_token_ids(tokenizer, *, enabled: bool) -> list[int]:
    if not enabled:
        return []
    ids: list[int] = []
    for marker in ("<think>", "</think>"):
        try:
            encoded = tokenizer.encode(marker, add_special_tokens=False)
        except Exception:
            encoded = []
        ids.extend(int(token_id) for token_id in encoded)
    return sorted(set(ids))


_ANSWER_HEADER_RE = re.compile(r"^\s*(?:\*\*)?\s*answer\s*(?:\*\*)?\s*:\s*(.*)$", re.IGNORECASE)


def _clean_short_answer_line(line: str) -> str:
    text = str(line or "").strip()
    text = text.replace("**", "").strip()
    text = re.sub(r"^[-*•]\s*", "", text)
    if " (" in text:
        text = text.split(" (", 1)[0].strip()
    return text


def apply_short_answer_governor(completion: str) -> str:
    """Collapse a drifting completion to the first answer-like line.

    This is a decode-time answer-channel governor, not a correctness judge. The
    raw completion is still stored separately so the gate can catch whether the
    model needed this intervention.
    """
    lines = [line.strip() for line in str(completion or "").strip().splitlines()]
    saw_answer_header = False
    for line in lines:
        if not line:
            continue
        match = _ANSWER_HEADER_RE.match(line)
        if match:
            answer = _clean_short_answer_line(match.group(1))
            if answer:
                return f"Answer: {answer}"
            saw_answer_header = True
            continue
        if saw_answer_header:
            answer = _clean_short_answer_line(line)
            if answer:
                return f"Answer: {answer}"
        if not saw_answer_header:
            return _clean_short_answer_line(line)
    return ""


def select_evidence_span_from_logits(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    *,
    max_span_tokens: int = 16,
    token_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Select the highest-scoring legal start/end pair from one batch item."""
    if start_logits.numel() == 0 or end_logits.numel() == 0:
        return {
            "status": "reader_unavailable",
            "selected_start": None,
            "selected_end": None,
            "selected_score": None,
        }
    start = start_logits[0].detach().float().cpu()
    end = end_logits[0].detach().float().cpu()
    n = min(int(start.numel()), int(end.numel()))
    if n <= 0:
        return {
            "status": "reader_unavailable",
            "selected_start": None,
            "selected_end": None,
            "selected_score": None,
        }
    max_len = max(1, int(max_span_tokens))
    if token_mask is not None:
        mask = token_mask[0].detach().bool().cpu() if token_mask.ndim > 1 else token_mask.detach().bool().cpu()
        if int(mask.numel()) < n:
            padded = torch.zeros(n, dtype=torch.bool)
            padded[: int(mask.numel())] = mask[:n]
            mask = padded
        else:
            mask = mask[:n]
        if not bool(mask.any().item()):
            return {
                "status": "source_mask_empty",
                "selected_start": None,
                "selected_end": None,
                "selected_score": None,
            }
        start = start.clone()
        end = end.clone()
        start[~mask] = float("-inf")
        end[~mask] = float("-inf")
    best_start = 0
    best_end = 0
    best_score = float("-inf")
    for start_idx in range(n):
        end_limit = min(n, start_idx + max_len)
        for end_idx in range(start_idx, end_limit):
            score = float(start[start_idx].item() + end[end_idx].item())
            if score > best_score:
                best_score = score
                best_start = start_idx
                best_end = end_idx
    if best_score <= -1.0e3:
        return {
            "status": "reader_unavailable",
            "selected_start": None,
            "selected_end": None,
            "selected_score": best_score,
        }
    return {
        "status": "span",
        "selected_start": best_start,
        "selected_end": best_end,
        "selected_score": best_score,
    }


_SOURCE_HEADER_RE = re.compile(r"^SOURCE=(.*?) CHUNK=(.*?) SCORE=")


def evidence_source_key(rec: dict[str, Any]) -> tuple[str, str]:
    return (str(rec.get("source", "?")), str(rec.get("chunk_id", "?")))


def selected_evidence_source_keys(
    evidence_results: Iterable[tuple[float, dict[str, Any]]],
) -> set[tuple[str, str]]:
    return {
        evidence_source_key(rec)
        for _, rec in evidence_results
        if bool(rec.get("source_selector_selected", False))
    }


def _workspace_source_text_char_spans(
    workspace_text: str,
    selected_source_keys: set[tuple[str, str]],
) -> list[tuple[int, int]]:
    if not selected_source_keys:
        return []
    spans: list[tuple[int, int]] = []
    lines = str(workspace_text or "").splitlines(keepends=True)
    offset = 0
    pending_key: tuple[str, str] | None = None
    for line in lines:
        header = _SOURCE_HEADER_RE.match(line.rstrip("\r\n"))
        if header:
            pending_key = (header.group(1), header.group(2))
            offset += len(line)
            continue
        if pending_key is not None:
            text_end = offset + len(line.rstrip("\r\n"))
            if pending_key in selected_source_keys and text_end > offset:
                spans.append((offset, text_end))
            pending_key = None
        offset += len(line)
    return [(start, end) for start, end in spans if end > start]


def evidence_source_token_mask(
    tokenizer,
    workspace_text: str,
    workspace_input_ids: torch.Tensor,
    selected_source_keys: set[tuple[str, str]],
) -> torch.Tensor:
    spans = _workspace_source_text_char_spans(workspace_text, selected_source_keys)
    n = int(workspace_input_ids.shape[-1])
    mask = torch.zeros((1, n), dtype=torch.bool, device=workspace_input_ids.device)
    if not spans:
        return mask
    try:
        enc = tokenizer(
            workspace_text,
            return_offsets_mapping=True,
            truncation=False,
            padding=False,
        )
        offsets = list(enc["offset_mapping"])
    except Exception:
        return mask
    if len(offsets) > n:
        offsets = offsets[-n:]
    elif len(offsets) < n:
        offsets = [(0, 0)] * (n - len(offsets)) + offsets
    for idx, offset in enumerate(offsets[:n]):
        if not isinstance(offset, (list, tuple)) or len(offset) < 2:
            continue
        start = int(offset[0])
        end = int(offset[1])
        if end <= start:
            continue
        if any(start < span_end and end > span_start for span_start, span_end in spans):
            mask[0, idx] = True
    return mask


_ASCII_ATOMIC_TEXT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _./:-]*$")
_ASCII_TOKEN_CONTINUATION_RE = re.compile(r"^[-_/]?[A-Za-z0-9]+")


def _decode_token_text(tokenizer, token_ids: Sequence[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(list(token_ids), skip_special_tokens=True)


def _looks_like_ascii_atomic_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(_ASCII_ATOMIC_TEXT_RE.fullmatch(stripped))


def _should_expand_span_left(tokenizer, workspace_ids: Sequence[int], start: int, end: int) -> bool:
    if start <= 0:
        return False
    current_first = _decode_token_text(tokenizer, [workspace_ids[start]])
    previous = _decode_token_text(tokenizer, [workspace_ids[start - 1]])
    current_answer = _decode_token_text(tokenizer, workspace_ids[start : end + 1])
    if not _looks_like_ascii_atomic_text(current_answer):
        return False
    if current_first[:1].isspace():
        return False
    previous_stripped = previous.strip()
    current_stripped = current_first.strip()
    if not previous_stripped or not current_stripped:
        return False
    if not previous_stripped[-1:].isascii() or not current_stripped[:1].isascii():
        return False
    return previous_stripped[-1:].isalpha() and current_stripped[:1].islower()


def _should_expand_span_right(tokenizer, workspace_ids: Sequence[int], start: int, end: int) -> bool:
    if end >= len(workspace_ids) - 1:
        return False
    answer = _decode_token_text(tokenizer, workspace_ids[start : end + 1]).strip()
    next_text = _decode_token_text(tokenizer, [workspace_ids[end + 1]])
    if not _looks_like_ascii_atomic_text(answer) or not next_text:
        return False
    if next_text[:1].isspace():
        return False
    next_stripped = next_text.strip()
    if not _ASCII_TOKEN_CONTINUATION_RE.match(next_stripped):
        return False
    if next_stripped.startswith(("-", "_", "/")):
        return bool(re.search(r"[A-Za-z0-9]$", answer))
    return bool(re.search(r"[-_/:][A-Za-z0-9]*$", answer) or re.search(r"[-_/:].*[A-Za-z0-9]$", answer))


def revise_evidence_span_boundary(
    tokenizer,
    workspace_input_ids: torch.Tensor,
    *,
    selected_start: int,
    selected_end: int,
    max_left_tokens: int = 2,
    max_right_tokens: int = 2,
) -> dict[str, Any]:
    """Expand a copied evidence span when tokenization cut an atomic identifier."""
    workspace_ids = [
        int(token_id)
        for token_id in workspace_input_ids[0].detach().cpu().tolist()
    ]
    start = max(0, min(int(selected_start), len(workspace_ids) - 1))
    end = max(start, min(int(selected_end), len(workspace_ids) - 1))
    original_start = start
    original_end = end
    for _ in range(max(0, int(max_left_tokens))):
        if not _should_expand_span_left(tokenizer, workspace_ids, start, end):
            break
        start -= 1
    for _ in range(max(0, int(max_right_tokens))):
        if not _should_expand_span_right(tokenizer, workspace_ids, start, end):
            break
        end += 1
    token_ids = workspace_ids[start : end + 1]
    return {
        "status": "revised" if (start, end) != (original_start, original_end) else "unchanged",
        "original_start": original_start,
        "original_end": original_end,
        "selected_start": start,
        "selected_end": end,
        "selected_token_ids": token_ids,
        "selected_text": _decode_token_text(tokenizer, token_ids).strip(),
    }


def _first_sigmoid_prob(outputs: dict[str, torch.Tensor], key: str) -> float | None:
    value = outputs.get(key)
    if value is None or value.numel() == 0:
        return None
    return float(torch.sigmoid(value[0].detach().float()).cpu().item())


def _first_direct_prob(outputs: dict[str, torch.Tensor], key: str) -> float | None:
    value = outputs.get(key)
    if value is None or value.numel() == 0:
        return None
    return float(value[0].detach().float().cpu().item())


def _evidence_causal_prob(outputs: dict[str, torch.Tensor]) -> tuple[float | None, str | None]:
    gate_prob = _first_direct_prob(outputs, "evidence_bottleneck_gate")
    if gate_prob is not None:
        return gate_prob, "evidence_bottleneck_gate"
    gate_logit_prob = _first_sigmoid_prob(outputs, "evidence_bottleneck_gate_logits")
    if gate_logit_prob is not None:
        return gate_logit_prob, "evidence_bottleneck_gate_logits"
    return _first_sigmoid_prob(outputs, "evidence_causal_gate_logits"), "evidence_causal_gate_logits"


def evidence_truth_gate_from_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    support_threshold: float = 0.5,
    causal_threshold: float = 0.5,
    refute_threshold: float = 0.5,
    missing_threshold: float = 0.5,
) -> dict[str, Any]:
    """Decide whether logical evidence heads allow copying an answer span."""
    support_prob = _first_sigmoid_prob(outputs, "evidence_support_logits")
    refute_prob = _first_sigmoid_prob(outputs, "evidence_refute_logits")
    missing_prob = _first_sigmoid_prob(outputs, "evidence_missing_logits")
    causal_prob, causal_source = _evidence_causal_prob(outputs)

    block_reasons: list[str] = []
    if support_prob is None or refute_prob is None or missing_prob is None or causal_prob is None:
        block_reasons.append("truth_heads_unavailable")
    else:
        if support_prob < float(support_threshold):
            block_reasons.append("support_low")
        if causal_prob < float(causal_threshold):
            block_reasons.append("causal_low")
        if refute_prob >= float(refute_threshold):
            block_reasons.append("refute_high")
        if missing_prob >= float(missing_threshold):
            block_reasons.append("missing_high")

    return {
        "enabled": True,
        "allow": not block_reasons,
        "status": "allow" if not block_reasons else "blocked",
        "block_reasons": block_reasons,
        "support_prob": support_prob,
        "causal_prob": causal_prob,
        "causal_source": causal_source,
        "refute_prob": refute_prob,
        "missing_prob": missing_prob,
        "support_threshold": float(support_threshold),
        "causal_threshold": float(causal_threshold),
        "refute_threshold": float(refute_threshold),
        "missing_threshold": float(missing_threshold),
    }


def evidence_span_copy_from_outputs(
    outputs: dict[str, torch.Tensor],
    tokenizer,
    workspace_input_ids: torch.Tensor | None,
    *,
    max_span_tokens: int = 16,
    no_answer_threshold: float = 0.5,
    min_span_score: float | None = None,
    truth_gate: dict[str, Any] | None = None,
    unknown_text: str = "UNKNOWN",
    answer_revision: str = "none",
    answer_revision_max_left_tokens: int = 2,
    answer_revision_max_right_tokens: int = 2,
    source_token_mask: torch.Tensor | None = None,
) -> tuple[str, dict[str, Any]]:
    start_logits = outputs.get("evidence_span_start_logits")
    end_logits = outputs.get("evidence_span_end_logits")
    no_answer_logits = outputs.get("evidence_span_no_answer_logits")
    source_token_mask_count = (
        int(source_token_mask.detach().bool().sum().item())
        if source_token_mask is not None
        else None
    )
    source_mask_has_tokens = source_token_mask_count is not None and source_token_mask_count > 0
    if (
        start_logits is None
        or end_logits is None
        or workspace_input_ids is None
        or start_logits.numel() == 0
        or end_logits.numel() == 0
    ):
        return (
            f"Answer: {unknown_text}",
            {
                "status": "reader_unavailable",
                "selected_start": None,
                "selected_end": None,
                "selected_score": None,
                "no_answer_prob": None,
                "selected_token_ids": [],
                "truth_gate": truth_gate,
                "source_token_mask_active": source_token_mask is not None,
                "source_token_mask_count": source_token_mask_count,
                "no_answer_deferred_by_source_mask": False,
            },
        )

    no_answer_prob = None
    no_answer_deferred_by_source_mask = False
    if no_answer_logits is not None and no_answer_logits.numel() > 0:
        no_answer_prob = float(torch.sigmoid(no_answer_logits[0].detach().float()).cpu().item())
        no_answer_deferred_by_source_mask = (
            no_answer_prob >= float(no_answer_threshold) and source_mask_has_tokens
        )
        if no_answer_prob >= float(no_answer_threshold) and not no_answer_deferred_by_source_mask:
            return (
                f"Answer: {unknown_text}",
                {
                    "status": "no_answer",
                    "selected_start": None,
                    "selected_end": None,
                    "selected_score": None,
                    "no_answer_prob": no_answer_prob,
                    "selected_token_ids": [],
                    "truth_gate": truth_gate,
                    "source_token_mask_active": source_token_mask is not None,
                    "source_token_mask_count": source_token_mask_count,
                    "no_answer_deferred_by_source_mask": False,
                },
            )

    span = select_evidence_span_from_logits(
        start_logits,
        end_logits,
        max_span_tokens=max_span_tokens,
        token_mask=source_token_mask,
    )
    if span["status"] != "span":
        span["no_answer_prob"] = no_answer_prob
        span["selected_token_ids"] = []
        span["truth_gate"] = truth_gate
        span["source_token_mask_active"] = source_token_mask is not None
        span["source_token_mask_count"] = source_token_mask_count
        span["no_answer_deferred_by_source_mask"] = no_answer_deferred_by_source_mask
        return f"Answer: {unknown_text}", span
    if min_span_score is not None and float(span["selected_score"]) < float(min_span_score):
        span["status"] = "low_span_score"
        span["no_answer_prob"] = no_answer_prob
        span["selected_token_ids"] = []
        span["truth_gate"] = truth_gate
        span["source_token_mask_active"] = source_token_mask is not None
        span["source_token_mask_count"] = source_token_mask_count
        span["no_answer_deferred_by_source_mask"] = no_answer_deferred_by_source_mask
        return f"Answer: {unknown_text}", span
    if truth_gate is not None and not bool(truth_gate.get("allow", False)):
        span["status"] = "truth_gate_blocked"
        span["no_answer_prob"] = no_answer_prob
        span["selected_token_ids"] = []
        span["truth_gate"] = truth_gate
        span["source_token_mask_active"] = source_token_mask is not None
        span["source_token_mask_count"] = source_token_mask_count
        span["no_answer_deferred_by_source_mask"] = no_answer_deferred_by_source_mask
        return f"Answer: {unknown_text}", span

    start = int(span["selected_start"])
    end = int(span["selected_end"])
    revision = {
        "status": "disabled",
        "strategy": str(answer_revision),
        "original_start": start,
        "original_end": end,
        "selected_start": start,
        "selected_end": end,
    }
    if answer_revision == "evidence_span_boundary":
        revision = revise_evidence_span_boundary(
            tokenizer,
            workspace_input_ids,
            selected_start=start,
            selected_end=end,
            max_left_tokens=answer_revision_max_left_tokens,
            max_right_tokens=answer_revision_max_right_tokens,
        )
        revision["strategy"] = answer_revision
        start = int(revision["selected_start"])
        end = int(revision["selected_end"])
        span["selected_start"] = start
        span["selected_end"] = end
    token_ids = [
        int(token_id)
        for token_id in workspace_input_ids[0, start : end + 1].detach().cpu().tolist()
    ]
    answer = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    if not answer:
        span["status"] = "empty_span"
        span["no_answer_prob"] = no_answer_prob
        span["selected_token_ids"] = token_ids
        span["truth_gate"] = truth_gate
        span["revision"] = revision
        span["source_token_mask_active"] = source_token_mask is not None
        span["source_token_mask_count"] = source_token_mask_count
        span["no_answer_deferred_by_source_mask"] = no_answer_deferred_by_source_mask
        return f"Answer: {unknown_text}", span
    span["no_answer_prob"] = no_answer_prob
    span["selected_token_ids"] = token_ids
    span["truth_gate"] = truth_gate
    span["revision"] = revision
    span["source_token_mask_active"] = source_token_mask is not None
    span["source_token_mask_count"] = source_token_mask_count
    span["no_answer_deferred_by_source_mask"] = no_answer_deferred_by_source_mask
    return f"Answer: {answer}", span


@torch.no_grad()
def donor_kwargs(
    donor: QwenDonorAdapter,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
) -> dict[str, torch.Tensor]:
    encoded = donor.encode_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_logits=True,
    )
    out: dict[str, torch.Tensor] = {
        "text_states": encoded["text_states"].to(device),
        "donor_logits": encoded["logits"].to(device),
    }
    if encoded.get("visual_features") is not None:
        out["visual_features"] = encoded["visual_features"].to(device)
    return out


@torch.no_grad()
def donor_workspace_memory_kwargs(
    donor: QwenDonorAdapter,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
) -> dict[str, torch.Tensor]:
    encoded = donor.encode_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_logits=False,
    )
    out: dict[str, torch.Tensor] = {
        "workspace_text_states": encoded["text_states"].to(device),
    }
    encoded_mask = encoded.get("attention_mask")
    if encoded_mask is not None:
        out["workspace_attention_mask"] = encoded_mask.to(device)
    return out


def mode_settings(mode: str, *, qtrm_scale: float, donor_scale: float) -> tuple[bool, float, float]:
    if mode.endswith("_with_evidence"):
        include_evidence = True
    elif mode.endswith("_no_evidence"):
        include_evidence = False
    else:
        raise ValueError(f"mode must end with _with_evidence or _no_evidence: {mode}")

    if mode.startswith("donor_only_"):
        return include_evidence, 0.0, donor_scale
    if (
        mode.startswith("qtrm_residual_")
        or mode.startswith("qtrm_workspace_off_")
        or mode.startswith("qtrm_core_off_")
        or mode.startswith("qtrm_coda_off_")
        or mode.startswith("qtrm_residual_head_off_")
        or mode.startswith("qtrm_donor_hidden_off_")
        or mode.startswith("qtrm_workspace_only_")
        or mode.startswith("qtrm_workspace_gate_off_")
        or mode.startswith("qtrm_workspace_memory_off_")
        or mode.startswith("qtrm_core_context_off_")
        or mode.startswith("qtrm_core_to_text_off_")
        or mode.startswith("qtrm_evidence_bottleneck_off_")
        or mode.startswith("qtrm_evidence_span_reader_off_")
        or mode.startswith("qtrm_answer_residual_governor_off_")
        or mode.startswith("qtrm_answer_decision_features_off_")
        or mode.startswith("qtrm_answer_decision_off_")
    ):
        return include_evidence, qtrm_scale, donor_scale
    raise ValueError(f"unknown mode: {mode}")


def truth_gate_enabled_for_mode(mode: str, *, requested: bool) -> bool:
    if not bool(requested):
        return False
    return not mode.startswith("qtrm_evidence_bottleneck_off_")


def mode_forward_kwargs(mode: str, *, core_halt_mode: str = "config") -> dict[str, bool]:
    kwargs = {
        "disable_workspace": mode.startswith("qtrm_workspace_off_"),
        "disable_core": mode.startswith("qtrm_core_off_"),
    }
    if (
        mode.startswith("qtrm_coda_off_")
        or mode.startswith("qtrm_residual_head_off_")
        or mode.startswith("qtrm_donor_hidden_off_")
        or mode.startswith("qtrm_workspace_only_")
        or mode.startswith("qtrm_workspace_gate_off_")
        or mode.startswith("qtrm_workspace_memory_off_")
        or mode.startswith("qtrm_core_context_off_")
        or mode.startswith("qtrm_core_to_text_off_")
        or mode.startswith("qtrm_evidence_bottleneck_off_")
        or mode.startswith("qtrm_evidence_span_reader_off_")
        or mode.startswith("qtrm_answer_residual_governor_off_")
        or mode.startswith("qtrm_answer_decision_features_off_")
        or mode.startswith("qtrm_answer_decision_off_")
    ):
        kwargs.update(
            {
                "disable_coda": mode.startswith("qtrm_coda_off_"),
                "disable_qtrm_residual": mode.startswith("qtrm_residual_head_off_"),
                "disable_donor_context": mode.startswith("qtrm_donor_hidden_off_"),
                "workspace_only_context": mode.startswith("qtrm_workspace_only_"),
            }
        )
        if mode.startswith("qtrm_core_context_off_"):
            kwargs["disable_core_context"] = True
        if mode.startswith("qtrm_core_to_text_off_"):
            kwargs["disable_core_to_text"] = True
        if mode.startswith("qtrm_workspace_gate_off_"):
            kwargs["disable_workspace_memory_gate"] = True
        if mode.startswith("qtrm_workspace_memory_off_"):
            kwargs["disable_workspace_memory_context"] = True
        if mode.startswith("qtrm_evidence_bottleneck_off_"):
            kwargs["disable_evidence_bottleneck"] = True
        if mode.startswith("qtrm_evidence_span_reader_off_"):
            kwargs["disable_evidence_span_reader"] = True
        if mode.startswith("qtrm_answer_residual_governor_off_"):
            kwargs["disable_answer_residual_governor"] = True
        if mode.startswith("qtrm_answer_decision_features_off_"):
            kwargs["disable_answer_decision_features"] = True
        if mode.startswith("qtrm_answer_decision_off_"):
            kwargs["disable_answer_decision_head"] = True
    if core_halt_mode == "enabled":
        kwargs["enable_core_halt"] = True
    elif core_halt_mode == "disabled":
        kwargs["enable_core_halt"] = False
    elif core_halt_mode != "config":
        raise ValueError(f"unknown core_halt_mode: {core_halt_mode}")
    return kwargs


def core_halt_telemetry(outputs: dict[str, torch.Tensor], *, core_halt_mode: str) -> dict[str, Any]:
    q_halt = outputs.get("core_q_halt_logits")
    q_continue = outputs.get("core_q_continue_logits")
    core_steps = outputs.get("core_steps")
    core_halted = outputs.get("core_halted")

    record: dict[str, Any] = {
        "mode": core_halt_mode,
        "core_steps": None,
        "core_halted": None,
        "q_halt_steps": 0,
        "q_halt_last_mean": None,
        "q_continue_steps": 0,
        "q_continue_last_mean": None,
    }
    if core_steps is not None:
        record["core_steps"] = core_steps.detach().cpu().tolist()
    if core_halted is not None:
        record["core_halted"] = core_halted.detach().cpu().tolist()
    if q_halt is not None and q_halt.numel() > 0:
        record["q_halt_steps"] = int(q_halt.shape[1]) if q_halt.ndim >= 2 else int(q_halt.numel())
        q_halt_last = q_halt[:, -1] if q_halt.ndim >= 2 else q_halt[-1:]
        record["q_halt_last_mean"] = float(q_halt_last.float().mean().detach().cpu().item())
    if q_continue is not None and q_continue.numel() > 0:
        record["q_continue_steps"] = int(q_continue.shape[1]) if q_continue.ndim >= 2 else int(q_continue.numel())
        q_continue_last = q_continue[:, -1] if q_continue.ndim >= 2 else q_continue[-1:]
        record["q_continue_last_mean"] = float(q_continue_last.float().mean().detach().cpu().item())
    return record


def _gate_tensor_summary(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None or tensor.numel() == 0:
        return {
            "steps": 0,
            "mean": None,
            "last_mean": None,
            "values": [],
        }
    values = tensor.detach().float().cpu()
    steps = int(values.shape[1]) if values.ndim >= 2 else int(values.numel())
    last = values[:, -1] if values.ndim >= 2 else values[-1:]
    return {
        "steps": steps,
        "mean": float(values.mean().item()),
        "last_mean": float(last.mean().item()),
        "values": values.tolist(),
    }


def latent_gate_telemetry(outputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    workspace = _gate_tensor_summary(outputs.get("workspace_update_gate_mean"))
    core_context = _gate_tensor_summary(outputs.get("core_context_gate_mean"))
    return {
        "workspace_update_gate_steps": workspace["steps"],
        "workspace_update_gate_mean": workspace["mean"],
        "workspace_update_gate_last_mean": workspace["last_mean"],
        "workspace_update_gate_values": workspace["values"],
        "core_context_gate_steps": core_context["steps"],
        "core_context_gate_mean": core_context["mean"],
        "core_context_gate_last_mean": core_context["last_mean"],
        "core_context_gate_values": core_context["values"],
    }


def json_safe_value(value: Any) -> Any:
    if torch.is_tensor(value):
        tensor = value.detach()
        summary: dict[str, Any] = {
            "tensor_shape": list(tensor.shape),
            "tensor_dtype": str(tensor.dtype),
            "tensor_device": str(tensor.device),
        }
        if tensor.numel() <= 8:
            summary["tensor_values"] = tensor.cpu().tolist()
        return summary
    if isinstance(value, dict):
        return {str(k): json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(v) for v in value]
    return value


@torch.no_grad()
def greedy_completion(
    model: QTRMMultimodalModel,
    donor: QwenDonorAdapter,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    device: str,
    max_new_tokens: int,
    forward_kwargs: dict[str, Any] | None = None,
    suppressed_token_ids: Iterable[int] | None = None,
    no_repeat_ngram_size: int = 0,
) -> tuple[str, str, list[int]]:
    generated = input_ids[0].detach().cpu().tolist()
    prompt_len = len(generated)
    forward_kwargs = forward_kwargs or {}

    for _ in range(max_new_tokens):
        cur_ids = torch.tensor([generated], dtype=torch.long, device=device)
        cur_mask = torch.ones_like(cur_ids)
        extra = donor_kwargs(donor, cur_ids, cur_mask, device)
        with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
            outputs = model(cur_ids, attention_mask=cur_mask, **extra, **forward_kwargs)
        step_suppressed_ids = list(suppressed_token_ids or [])
        step_suppressed_ids.extend(
            no_repeat_ngram_banned_tokens(generated[prompt_len:], no_repeat_ngram_size)
        )
        next_id = select_next_token(
            outputs["logits"][0, -1].float(),
            suppressed_token_ids=step_suppressed_ids,
        )
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break
        generated.append(next_id)

    completion_ids = generated[prompt_len:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    full_text = tokenizer.decode(generated, skip_special_tokens=True)
    return completion, full_text, completion_ids


@torch.no_grad()
def first_step_logit_shift(
    model: QTRMMultimodalModel,
    donor: QwenDonorAdapter,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    device: str,
    qtrm_scale: float,
    donor_scale: float,
    top_k: int = 5,
    forward_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    original_qtrm_scale = float(model.cfg.qtrm_logits_scale)
    original_donor_scale = float(model.cfg.donor_logits_scale)
    extra = donor_kwargs(donor, input_ids, attention_mask, device)
    forward_kwargs = forward_kwargs or {}

    try:
        model.cfg.donor_logits_scale = donor_scale
        model.cfg.qtrm_logits_scale = 0.0
        with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
            donor_only = model(
                input_ids,
                attention_mask=attention_mask,
                **extra,
                **forward_kwargs,
            )["logits"][0, -1].float()

        model.cfg.qtrm_logits_scale = qtrm_scale
        with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
            residual = model(
                input_ids,
                attention_mask=attention_mask,
                **extra,
                **forward_kwargs,
            )["logits"][0, -1].float()
    finally:
        model.cfg.qtrm_logits_scale = original_qtrm_scale
        model.cfg.donor_logits_scale = original_donor_scale

    delta = residual - donor_only
    k = min(top_k, delta.numel())
    top_abs = torch.topk(delta.abs(), k=k)
    donor_top = int(donor_only.argmax(dim=-1).detach().cpu().item())
    residual_top = int(residual.argmax(dim=-1).detach().cpu().item())
    return {
        "argmax_changed": donor_top != residual_top,
        "donor_top_id": donor_top,
        "donor_top_token": tokenizer.decode([donor_top], skip_special_tokens=False),
        "residual_top_id": residual_top,
        "residual_top_token": tokenizer.decode([residual_top], skip_special_tokens=False),
        "max_abs_delta": float(delta.abs().max().detach().cpu().item()),
        "mean_abs_delta": float(delta.abs().mean().detach().cpu().item()),
        "l2_delta": float(torch.linalg.vector_norm(delta).detach().cpu().item()),
        "top_abs_delta": [
            {
                "token_id": int(idx.detach().cpu().item()),
                "token": tokenizer.decode([int(idx.detach().cpu().item())], skip_special_tokens=False),
                "delta": float(delta[int(idx)].detach().cpu().item()),
            }
            for idx in top_abs.indices
        ],
    }


@torch.no_grad()
def prompt_forward_telemetry(
    model: QTRMMultimodalModel,
    donor: QwenDonorAdapter,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    device: str,
    forward_kwargs: dict[str, Any] | None = None,
    core_halt_mode: str = "config",
) -> dict[str, Any]:
    extra = donor_kwargs(donor, input_ids, attention_mask, device)
    forward_kwargs = forward_kwargs or {}
    with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            **extra,
            **forward_kwargs,
        )
    return {
        "core_halt": core_halt_telemetry(outputs, core_halt_mode=core_halt_mode),
        "latent_gates": latent_gate_telemetry(outputs),
    }


@torch.no_grad()
def prompt_core_halt_telemetry(
    model: QTRMMultimodalModel,
    donor: QwenDonorAdapter,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    device: str,
    forward_kwargs: dict[str, Any] | None = None,
    core_halt_mode: str = "config",
) -> dict[str, Any]:
    return prompt_forward_telemetry(
        model,
        donor,
        input_ids,
        attention_mask,
        device=device,
        forward_kwargs=forward_kwargs,
        core_halt_mode=core_halt_mode,
    )["core_halt"]


@torch.no_grad()
def evidence_span_copy_completion(
    model: QTRMMultimodalModel,
    donor: QwenDonorAdapter,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    workspace_input_ids: torch.Tensor | None,
    *,
    device: str,
    forward_kwargs: dict[str, Any] | None = None,
    source_token_mask: torch.Tensor | None = None,
    max_span_tokens: int = 16,
    no_answer_threshold: float = 0.5,
    min_span_score: float | None = None,
    truth_gate_enabled: bool = False,
    truth_support_threshold: float = 0.5,
    truth_causal_threshold: float = 0.5,
    truth_refute_threshold: float = 0.5,
    truth_missing_threshold: float = 0.5,
    answer_revision: str = "none",
    answer_revision_max_left_tokens: int = 2,
    answer_revision_max_right_tokens: int = 2,
) -> tuple[str, str, list[int], dict[str, Any]]:
    forward_kwargs = forward_kwargs or {}
    if workspace_input_ids is None:
        return (
            "Answer: UNKNOWN",
            tokenizer.decode(input_ids[0].detach().cpu().tolist(), skip_special_tokens=True)
            + "Answer: UNKNOWN",
            [],
            {
                "status": "no_workspace_memory",
                "selected_start": None,
                "selected_end": None,
                "selected_score": None,
                "no_answer_prob": None,
                "selected_token_ids": [],
                "truth_gate": None,
            },
        )
    extra = donor_kwargs(donor, input_ids, attention_mask, device)
    with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            **extra,
            **forward_kwargs,
        )
    truth_gate = (
        evidence_truth_gate_from_outputs(
            outputs,
            support_threshold=truth_support_threshold,
            causal_threshold=truth_causal_threshold,
            refute_threshold=truth_refute_threshold,
            missing_threshold=truth_missing_threshold,
        )
        if truth_gate_enabled
        else None
    )
    completion, meta = evidence_span_copy_from_outputs(
        outputs,
        tokenizer,
        workspace_input_ids,
        max_span_tokens=max_span_tokens,
        no_answer_threshold=no_answer_threshold,
        min_span_score=min_span_score,
        truth_gate=truth_gate,
        answer_revision=answer_revision,
        answer_revision_max_left_tokens=answer_revision_max_left_tokens,
        answer_revision_max_right_tokens=answer_revision_max_right_tokens,
        source_token_mask=source_token_mask,
    )
    full_text = (
        tokenizer.decode(input_ids[0].detach().cpu().tolist(), skip_special_tokens=True)
        + completion
    )
    return completion, full_text, list(meta.get("selected_token_ids") or []), meta


def _load_answer_decision_module():
    path = Path("scripts/161_train_answer_decision_head.py")
    spec = importlib.util.spec_from_file_location("answer_decision_head_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_evidence_source_selector_module():
    path = Path("scripts/165_train_evidence_source_selector.py")
    spec = importlib.util.spec_from_file_location("evidence_source_selector_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_evidence_source_selector_checkpoint(
    checkpoint_path: str | Path,
    *,
    threshold_override: float | None = None,
) -> dict[str, Any]:
    module = _load_evidence_source_selector_module()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_names = list(state.get("feature_names") or [])
    args = dict(state.get("args") or {})
    hidden_dim = int(args.get("hidden_dim", 32))
    dropout = float(args.get("dropout", 0.0))
    model = module.EvidenceSourceSelector(
        len(feature_names),
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    model.load_state_dict(state["model"])
    model.eval()
    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else float(state.get("selected_threshold", 0.5))
    )
    return {
        "module": module,
        "model": model,
        "threshold": threshold,
        "feature_names": feature_names,
        "checkpoint": str(checkpoint_path),
    }


def apply_evidence_source_selector(
    *,
    selector: dict[str, Any],
    case: dict[str, Any],
    evidence_results: list[tuple[float, dict[str, Any]]],
) -> list[tuple[float, dict[str, Any]]]:
    module = selector["module"]
    model = selector["model"]
    threshold = float(selector["threshold"])
    expected_dim = len(selector.get("feature_names") or [])
    total = len(evidence_results)
    enriched_results: list[tuple[float, dict[str, Any]]] = []
    feature_rows: list[list[float]] = []
    for rank, (score, rec) in enumerate(evidence_results):
        features = module.extract_features(
            case,
            rec,
            retrieval_score=float(score),
            rank=rank,
            total_records=total,
        )
        if expected_dim and len(features) != expected_dim:
            raise ValueError(
                f"source-selector feature mismatch: got {len(features)} expected {expected_dim}"
            )
        feature_rows.append(features)
    if feature_rows:
        with torch.no_grad():
            x = torch.tensor(feature_rows, dtype=torch.float32)
            probs = torch.sigmoid(model(x)).detach().cpu().tolist()
    else:
        probs = []
    for (score, rec), prob in zip(evidence_results, probs):
        enriched = dict(rec)
        source_probability = float(prob)
        enriched["source_selector_checkpoint"] = str(selector.get("checkpoint", ""))
        enriched["source_selector_probability"] = source_probability
        enriched["source_selector_threshold"] = threshold
        enriched["source_selector_selected"] = source_probability >= threshold
        enriched_results.append((score, enriched))
    return enriched_results


def load_answer_decision_checkpoint(
    checkpoint_path: str | Path,
    *,
    threshold_override: float | None = None,
) -> dict[str, Any]:
    module = _load_answer_decision_module()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_names = list(state.get("feature_names") or [])
    args = dict(state.get("args") or {})
    hidden_dim = int(args.get("hidden_dim", 32))
    dropout = float(args.get("dropout", 0.0))
    model = module.AnswerDecisionHead(
        len(feature_names),
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    model.load_state_dict(state["model"])
    model.eval()
    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else float(state.get("selected_threshold", 0.5))
    )
    return {
        "module": module,
        "model": model,
        "threshold": threshold,
        "feature_names": feature_names,
        "include_task_family": bool(args.get("include_task_family_features", False)),
        "checkpoint": str(checkpoint_path),
    }


def apply_answer_decision(
    *,
    decision: dict[str, Any],
    case: dict[str, Any],
    completion: str,
    answer_channel_meta: dict[str, Any],
    completion_ids: Sequence[int],
    input_ids: torch.Tensor,
    logit_shift: dict[str, Any] | None,
    prompt_telemetry: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    module = decision["module"]
    model = decision["model"]
    threshold = float(decision["threshold"])
    decision_record = {
        "id": case.get("id", ""),
        "question": case.get("question", ""),
        "task_family": case_task_family(case),
        "completion": completion,
        "answer_channel_meta": answer_channel_meta,
        "completion_token_count": len(list(completion_ids or [])),
        "prompt_token_count": int(input_ids.shape[-1]),
        "first_step_logit_shift": logit_shift or {},
        "latent_gates": prompt_telemetry.get("latent_gates") or {},
    }
    features = module.extract_features(
        decision_record,
        include_task_family=bool(decision.get("include_task_family", False)),
    )
    expected_dim = len(decision.get("feature_names") or [])
    if expected_dim and len(features) != expected_dim:
        raise ValueError(
            f"answer-decision feature mismatch: got {len(features)} expected {expected_dim}"
        )
    with torch.no_grad():
        x = torch.tensor([features], dtype=torch.float32)
        block_prob = float(torch.sigmoid(model(x))[0].item())
    blocked = block_prob >= threshold and canonical_answer_text(completion) != "UNKNOWN"
    meta = dict(answer_channel_meta)
    meta["answer_decision"] = {
        "checkpoint": str(decision.get("checkpoint", "")),
        "block_probability": block_prob,
        "threshold": threshold,
        "blocked": bool(blocked),
        "feature_names": list(decision.get("feature_names") or []),
    }
    if blocked:
        return "Answer: UNKNOWN", meta
    return completion, meta


def apply_model_answer_decision(
    *,
    completion: str,
    answer_channel_meta: dict[str, Any],
    threshold: float,
) -> tuple[str, dict[str, Any]]:
    decision = dict(answer_channel_meta.get("model_answer_decision") or {})
    block_prob = decision.get("block_probability")
    meta = dict(answer_channel_meta)
    blocked = (
        block_prob is not None
        and float(block_prob) >= float(threshold)
        and canonical_answer_text(completion) != "UNKNOWN"
    )
    meta["answer_decision"] = {
        "source": "model",
        "block_probability": None if block_prob is None else float(block_prob),
        "threshold": float(threshold),
        "blocked": bool(blocked),
    }
    if "block_logit" in decision:
        meta["answer_decision"]["block_logit"] = float(decision["block_logit"])
    if blocked:
        return "Answer: UNKNOWN", meta
    return completion, meta


def build_answer_decision_prompt(prompt: str, completion: str) -> str:
    return (
        f"{str(prompt).rstrip()}\n\n"
        "Candidate answer:\n"
        f"{str(completion).strip()}\n\n"
        "Decide whether the candidate must be blocked to UNKNOWN."
    )


def build_model_answer_decision_features(
    *,
    case: dict[str, Any],
    completion: str,
    answer_channel_meta: dict[str, Any],
    completion_ids: Sequence[int],
    input_ids: torch.Tensor,
    logit_shift: dict[str, Any] | None,
    prompt_telemetry: dict[str, Any],
) -> tuple[list[float], list[str]]:
    module = _load_answer_decision_module()
    decision_record = {
        "id": case.get("id", ""),
        "question": case.get("question", ""),
        "task_family": case_task_family(case),
        "completion": completion,
        "answer_channel_meta": answer_channel_meta,
        "completion_token_count": len(list(completion_ids or [])),
        "prompt_token_count": int(input_ids.shape[-1]),
        "first_step_logit_shift": logit_shift or {},
        "latent_gates": (prompt_telemetry or {}).get("latent_gates") or {},
    }
    return (
        module.extract_features(decision_record, include_task_family=False),
        module.feature_names(include_task_family=False),
    )


@torch.no_grad()
def model_answer_decision_metadata(
    model: QTRMMultimodalModel,
    donor: QwenDonorAdapter,
    tokenizer,
    *,
    case: dict[str, Any] | None = None,
    prompt: str,
    completion: str,
    answer_channel_meta: dict[str, Any] | None = None,
    completion_ids: Sequence[int] | None = None,
    input_ids: torch.Tensor | None = None,
    logit_shift: dict[str, Any] | None = None,
    prompt_telemetry: dict[str, Any] | None = None,
    max_length: int,
    device: str,
    forward_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decision_prompt = build_answer_decision_prompt(prompt, completion)
    decision_inputs = prepare_inputs(tokenizer, decision_prompt, max_length, device)
    decision_ids = decision_inputs["input_ids"]
    decision_mask = decision_inputs.get("attention_mask", torch.ones_like(decision_ids))
    extra = donor_kwargs(donor, decision_ids, decision_mask, device)
    feature_names: list[str] = []
    feature_dim = max(0, int(getattr(model.cfg, "answer_decision_feature_dim", 0)))
    if feature_dim > 0:
        if (
            case is None
            or answer_channel_meta is None
            or completion_ids is None
            or input_ids is None
            or prompt_telemetry is None
        ):
            return {
                "available": False,
                "reason": "missing_answer_decision_telemetry_features",
            }
        features, feature_names = build_model_answer_decision_features(
            case=case,
            completion=completion,
            answer_channel_meta=answer_channel_meta,
            completion_ids=completion_ids,
            input_ids=input_ids,
            logit_shift=logit_shift,
            prompt_telemetry=prompt_telemetry,
        )
        if len(features) != feature_dim:
            return {
                "available": False,
                "reason": "answer_decision_feature_dim_mismatch",
                "feature_count": len(features),
                "expected_feature_dim": feature_dim,
                "feature_names": feature_names,
            }
        extra["answer_decision_features"] = torch.tensor(
            [features],
            device=device,
            dtype=torch.float32,
        )
    with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
        outputs = model(
            decision_ids,
            attention_mask=decision_mask,
            **extra,
            **(forward_kwargs or {}),
        )
    logits = outputs.get("answer_decision_logits")
    if logits is None or logits.numel() == 0:
        return {"available": False}
    logit = float(logits.reshape(-1)[0].float().detach().cpu().item())
    metadata = {
        "available": True,
        "block_logit": logit,
        "block_probability": float(torch.sigmoid(torch.tensor(logit)).item()),
        "feature_source": "answer_channel_telemetry" if feature_dim > 0 else "hidden_state",
        "feature_count": feature_dim,
        "feature_names": feature_names,
    }
    hidden_logits = outputs.get("answer_decision_hidden_logits")
    if hidden_logits is not None and hidden_logits.numel() > 0:
        metadata["hidden_logit"] = float(
            hidden_logits.reshape(-1)[0].float().detach().cpu().item()
        )
    feature_logits = outputs.get("answer_decision_feature_logits")
    if feature_logits is not None and feature_logits.numel() > 0:
        metadata["feature_logit"] = float(
            feature_logits.reshape(-1)[0].float().detach().cpu().item()
        )
    return metadata


def replace_completion_suffix(
    full_text: str,
    *,
    old_completion: str,
    new_completion: str,
) -> str:
    if old_completion == new_completion:
        return full_text
    if old_completion and full_text.endswith(old_completion):
        return full_text[: -len(old_completion)] + new_completion
    if full_text.endswith("\n"):
        return full_text + new_completion
    return full_text + "\n" + new_completion


def evaluate_case(
    *,
    case: dict[str, Any],
    mode: str,
    model: QTRMMultimodalModel,
    donor: QwenDonorAdapter,
    tokenizer,
    device: str,
    max_length: int,
    max_new_tokens: int,
    memory_max_chars: int,
    evidence_mode: str,
    retrieval_top_k: int,
    memory_link_expansion: int,
    memory_index: str | None,
    memory_model_id: str | None,
    memory_backend: str | None,
    hnsw_ef_search: int | None,
    retrieve_top_n: int | None,
    rerank_backend: str,
    reranker_model_id: str,
    reranker_device: str | None,
    evidence_source_governor: str,
    memoryos_case_filter: bool,
    evidence_injection: str,
    base_qtrm_scale: float,
    donor_logits_scale: float,
    measure_logit_shift: bool,
    core_halt_mode: str,
    suppressed_token_ids: Iterable[int] | None,
    no_repeat_ngram_size: int,
    short_answer_governor: bool,
    answer_channel: str,
    evidence_span_max_tokens: int,
    evidence_span_no_answer_threshold: float,
    evidence_span_min_score: float | None,
    answer_revision: str,
    answer_revision_max_left_tokens: int,
    answer_revision_max_right_tokens: int,
    truth_gate: bool,
    truth_support_threshold: float,
    truth_causal_threshold: float,
    truth_refute_threshold: float,
    truth_missing_threshold: float,
    answer_decision: dict[str, Any] | None = None,
    model_answer_decision: bool = False,
    model_answer_decision_threshold: float = 0.5,
    evidence_source_selector: dict[str, Any] | None = None,
    evidence_source_selector_mode: str = "span_mask",
) -> dict[str, Any]:
    include_evidence, qtrm_scale, donor_scale = mode_settings(
        mode,
        qtrm_scale=base_qtrm_scale,
        donor_scale=donor_logits_scale,
    )
    forward_kwargs = mode_forward_kwargs(mode, core_halt_mode=core_halt_mode)
    model.cfg.qtrm_logits_scale = qtrm_scale
    model.cfg.donor_logits_scale = donor_scale

    evidence_results = []
    if include_evidence:
        if evidence_mode == "memoryos":
            if not memory_index:
                raise ValueError("--memory-index is required when --evidence-mode memoryos")
            from wgram_lm.memoryos.retrieve import retrieve

            raw_results = retrieve(
                memory_index,
                str(case.get("question", "")),
                top_k=retrieve_top_n or max(retrieval_top_k * 8, retrieval_top_k),
                model_id=memory_model_id,
                backend=memory_backend,
                hnsw_ef_search=hnsw_ef_search,
                rerank_backend=rerank_backend,
                reranker_model_id=reranker_model_id,
                rerank_top_k=(retrieve_top_n or max(retrieval_top_k * 8, retrieval_top_k))
                if memoryos_case_filter
                else retrieval_top_k,
                reranker_device=reranker_device,
            )
            if memoryos_case_filter:
                case_candidates = filter_results_for_case(
                    raw_results,
                    case_id=str(case.get("id", "")),
                    top_k=retrieve_top_n or max(retrieval_top_k * 8, retrieval_top_k),
                )
                evidence_results = case_candidates[:retrieval_top_k]
                evidence_results = expand_linked_evidence_results(
                    evidence_results,
                    case_candidates,
                    max_extra=memory_link_expansion,
                )
            else:
                evidence_results = raw_results[:retrieval_top_k]
                evidence_results = expand_linked_evidence_results(
                    evidence_results,
                    raw_results,
                    max_extra=memory_link_expansion,
                )
        else:
            evidence_results = select_evidence_results(
                case,
                evidence_mode=evidence_mode,
                top_k=retrieval_top_k,
            )
        evidence_results = govern_evidence_sources(
            case,
            evidence_results,
            governor=evidence_source_governor,
        )
        if evidence_source_selector is not None:
            evidence_results = apply_evidence_source_selector(
                selector=evidence_source_selector,
                case=case,
                evidence_results=evidence_results,
            )
    prompt, workspace_memory_text = build_case_prompt_and_workspace_memory(
        case,
        include_evidence=include_evidence,
        evidence_results=evidence_results,
        max_evidence_chars=memory_max_chars,
        evidence_injection=evidence_injection,
    )
    inputs = prepare_inputs(tokenizer, prompt, max_length, device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    span_reader_input_ids_for_answer: torch.Tensor | None = None
    source_token_mask_for_answer: torch.Tensor | None = None
    span_reader_context = "none"
    if workspace_memory_text:
        workspace_inputs = prepare_inputs(tokenizer, workspace_memory_text, max_length, device)
        span_reader_input_ids_for_answer = workspace_inputs["input_ids"]
        workspace_attention_mask = workspace_inputs.get(
            "attention_mask",
            torch.ones_like(workspace_inputs["input_ids"]),
        )
        span_reader_context = "workspace"
        forward_kwargs.update(
            donor_workspace_memory_kwargs(
                donor,
                workspace_inputs["input_ids"],
                workspace_attention_mask,
                device,
            )
        )
        if (
            evidence_source_selector is not None
            and evidence_source_selector_mode == "span_mask"
        ):
            source_token_mask_for_answer = evidence_source_token_mask(
                tokenizer,
                workspace_memory_text,
                span_reader_input_ids_for_answer,
                selected_evidence_source_keys(evidence_results),
            )
    elif include_evidence and evidence_injection == "ssot":
        span_reader_input_ids_for_answer = input_ids
        span_reader_context = "ssot_prompt"
        forward_kwargs["evidence_span_reader_context"] = "input"
        if (
            evidence_source_selector is not None
            and evidence_source_selector_mode == "span_mask"
        ):
            source_token_mask_for_answer = evidence_source_token_mask(
                tokenizer,
                prompt,
                span_reader_input_ids_for_answer,
                selected_evidence_source_keys(evidence_results),
            )
    logit_shift = first_step_logit_shift(
        model,
        donor,
        tokenizer,
        input_ids,
        attention_mask,
        device=device,
        qtrm_scale=base_qtrm_scale,
        donor_scale=donor_logits_scale,
        forward_kwargs=forward_kwargs,
    ) if measure_logit_shift else None
    prompt_telemetry = prompt_forward_telemetry(
        model,
        donor,
        input_ids,
        attention_mask,
        device=device,
        forward_kwargs=forward_kwargs,
        core_halt_mode=core_halt_mode,
    )
    answer_channel_meta: dict[str, Any] = {"status": "greedy"}
    if answer_channel == "evidence_span_copy":
        if mode.startswith("donor_only_"):
            completion = "Answer: UNKNOWN"
            full_text = tokenizer.decode(input_ids[0].detach().cpu().tolist(), skip_special_tokens=True) + completion
            completion_ids = []
            answer_channel_meta = {
                "status": "unavailable_for_donor_only",
                "selected_start": None,
                "selected_end": None,
                "selected_score": None,
                "no_answer_prob": None,
                "selected_token_ids": [],
                "truth_gate": None,
            }
        else:
            completion, full_text, completion_ids, answer_channel_meta = evidence_span_copy_completion(
                model,
                donor,
                tokenizer,
                input_ids,
                attention_mask,
                span_reader_input_ids_for_answer,
                device=device,
                forward_kwargs=forward_kwargs,
                source_token_mask=source_token_mask_for_answer,
                max_span_tokens=evidence_span_max_tokens,
                no_answer_threshold=evidence_span_no_answer_threshold,
                min_span_score=evidence_span_min_score,
                truth_gate_enabled=truth_gate_enabled_for_mode(mode, requested=truth_gate),
                truth_support_threshold=truth_support_threshold,
                truth_causal_threshold=truth_causal_threshold,
                truth_refute_threshold=truth_refute_threshold,
                truth_missing_threshold=truth_missing_threshold,
                answer_revision=answer_revision,
                answer_revision_max_left_tokens=answer_revision_max_left_tokens,
                answer_revision_max_right_tokens=answer_revision_max_right_tokens,
            )
    elif answer_channel == "greedy":
        completion, full_text, completion_ids = greedy_completion(
            model,
            donor,
            tokenizer,
            input_ids,
            attention_mask,
            device=device,
            max_new_tokens=max_new_tokens,
            forward_kwargs=forward_kwargs,
            suppressed_token_ids=suppressed_token_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    else:
        raise ValueError(f"unknown answer_channel: {answer_channel}")
    raw_completion = completion
    if answer_decision is not None:
        decision_input_completion = completion
        completion, answer_channel_meta = apply_answer_decision(
            decision=answer_decision,
            case=case,
            completion=completion,
            answer_channel_meta=answer_channel_meta,
            completion_ids=completion_ids,
            input_ids=input_ids,
            logit_shift=logit_shift,
            prompt_telemetry=prompt_telemetry,
        )
        full_text = replace_completion_suffix(
            full_text,
            old_completion=decision_input_completion,
            new_completion=completion,
        )
    if model_answer_decision:
        answer_channel_meta = dict(answer_channel_meta)
        answer_channel_meta["model_answer_decision"] = model_answer_decision_metadata(
            model,
            donor,
            tokenizer,
            case=case,
            prompt=prompt,
            completion=completion,
            answer_channel_meta=answer_channel_meta,
            completion_ids=completion_ids,
            input_ids=input_ids,
            logit_shift=logit_shift,
            prompt_telemetry=prompt_telemetry,
            max_length=max_length,
            device=device,
            forward_kwargs=forward_kwargs,
        )
        decision_input_completion = completion
        completion, answer_channel_meta = apply_model_answer_decision(
            completion=completion,
            answer_channel_meta=answer_channel_meta,
            threshold=model_answer_decision_threshold,
        )
        full_text = replace_completion_suffix(
            full_text,
            old_completion=decision_input_completion,
            new_completion=completion,
        )
    if short_answer_governor:
        completion = apply_short_answer_governor(completion)
    answer_score = score_answer(
        completion,
        case["answer_aliases"],
        expected_unknown=expected_unknown_case(case),
    )
    hit = bool(answer_score["hit"])
    retrieval_stats = (
        target_retrieval_stats(case, evidence_results)
        if include_evidence
        else {
            "target_count": 0,
            "retrieved_target_count": 0,
            "retrieved_target": False,
            "all_targets_retrieved": False,
            "target_recall": 0.0,
        }
    )
    workspace_attention_for_record = forward_kwargs.get("workspace_attention_mask")
    active_workspace_memory_token_count = 0
    if (
        workspace_attention_for_record is not None
        and not bool(forward_kwargs.get("disable_workspace"))
        and not bool(forward_kwargs.get("disable_workspace_memory_context"))
    ):
        active_workspace_memory_token_count = int(workspace_attention_for_record.sum().item())
    return {
        "id": case["id"],
        "category": case.get("category", "uncategorized"),
        "question": case.get("question", ""),
        "task_family": case_task_family(case),
        "expected_unknown": expected_unknown_case(case),
        "mode": mode,
        "hit": hit,
        "exact_match": answer_score["exact_match"],
        "normalized_exact": answer_score["normalized_exact"],
        "normalized_contains": answer_score["normalized_contains"],
        "unknown_correct": answer_score["unknown_correct"],
        "match_type": answer_score["match_type"],
        "matched_aliases": answer_score["matched_aliases"],
        "canonical_answer": answer_score["canonical_answer"],
        "needs_human_audit": answer_score["needs_human_audit"],
        "audit_reasons": answer_score["audit_reasons"],
        "judge_status": answer_score["judge_status"],
        "answer_aliases": case["answer_aliases"],
        "completion": completion,
        "raw_completion": raw_completion,
        "short_answer_governor": bool(short_answer_governor),
        "answer_channel": answer_channel,
        "answer_channel_meta": answer_channel_meta,
        "completion_token_count": len(completion_ids),
        "prompt_token_count": int(input_ids.shape[1]),
        "include_evidence": include_evidence,
        "evidence_mode": evidence_mode if include_evidence else "none",
        "evidence_injection": evidence_injection if include_evidence else "none",
        "span_reader_context": span_reader_context,
        "workspace_memory_token_count": active_workspace_memory_token_count,
        "retrieved_target": retrieval_stats["retrieved_target"],
        "target_count": retrieval_stats["target_count"],
        "retrieved_target_count": retrieval_stats["retrieved_target_count"],
        "all_targets_retrieved": retrieval_stats["all_targets_retrieved"],
        "target_recall": retrieval_stats["target_recall"],
        "retrieved_roles": [rec.get("evidence_role", "unknown") for _, rec in evidence_results],
        "retrieved_sources": [rec.get("source", "?") for _, rec in evidence_results],
        "retrieved_rerank_backend": [rec.get("rerank_backend", "none") for _, rec in evidence_results],
        "retrieved_rerank_scores": [rec.get("rerank_score") for _, rec in evidence_results],
        "retrieved_retrieval_scores": [rec.get("retrieval_score", score) for score, rec in evidence_results],
        "evidence_source_governor": evidence_source_governor,
        "retrieved_source_governor_scores": [rec.get("source_governor_score") for _, rec in evidence_results],
        "evidence_source_selector": (
            str(evidence_source_selector.get("checkpoint", ""))
            if evidence_source_selector is not None
            else ""
        ),
        "evidence_source_selector_mode": (
            evidence_source_selector_mode if evidence_source_selector is not None else "none"
        ),
        "retrieved_source_selector_probabilities": [
            rec.get("source_selector_probability") for _, rec in evidence_results
        ],
        "retrieved_source_selector_selected": [
            rec.get("source_selector_selected") for _, rec in evidence_results
        ],
        "qtrm_logits_scale": qtrm_scale,
        "donor_logits_scale": donor_scale,
        "forward_ablation": forward_kwargs,
        "suppressed_token_ids": list(suppressed_token_ids or []),
        "no_repeat_ngram_size": int(no_repeat_ngram_size),
        "core_halt": prompt_telemetry["core_halt"],
        "latent_gates": prompt_telemetry["latent_gates"],
        "first_step_logit_shift": logit_shift,
        "full_text": full_text,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_canonical_ssot_contract(args)
    cfg = load_config(args.config)
    if bool(args.require_canonical_ssot):
        validate_canonical_model_contract(cfg)
    if not cfg.donor.model_id:
        raise SystemExit("donor.model_id is required")
    device = select_device(cfg.train.device, args.device)
    max_length = args.max_length or cfg.train.seq_len
    modes = args.mode or DEFAULT_MODES
    cases = load_cases(args.cases)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.donor.model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    suppressed_token_ids = visible_reasoning_token_ids(
        tokenizer,
        enabled=args.suppress_visible_reasoning_tokens,
    )

    print("=" * 72)
    print("QTRM Memory Retrieval Probe")
    print(f"config={args.config}")
    print(f"checkpoint={args.checkpoint}")
    print(f"cases={args.cases} count={len(cases)}")
    print(
        f"device={device} max_length={max_length} max_new_tokens={args.max_new_tokens} "
        f"evidence_mode={args.evidence_mode} retrieval_top_k={args.retrieval_top_k} "
        f"retrieve_top_n={args.retrieve_top_n} rerank_backend={args.rerank_backend} "
        f"evidence_source_governor={args.evidence_source_governor} "
        f"evidence_source_selector={bool(args.evidence_source_selector_checkpoint)} "
        f"evidence_source_selector_mode={args.evidence_source_selector_mode} "
        f"evidence_injection={args.evidence_injection} "
        f"answer_channel={args.answer_channel} "
        f"answer_revision={args.answer_revision} "
        f"truth_gate={args.truth_gate} "
        f"answer_decision={bool(args.answer_decision_checkpoint)} "
        f"model_answer_decision={args.model_answer_decision} "
        f"no_repeat_ngram_size={args.no_repeat_ngram_size} "
        f"short_answer_governor={args.short_answer_governor} "
        f"suppressed_token_ids={suppressed_token_ids}"
    )
    print("=" * 72)

    model = load_qtrm(args.config, args.checkpoint, device)
    donor = QwenDonorAdapter(cfg.donor)
    base_qtrm_scale = resolve_qtrm_scale(cfg.model.qtrm_logits_scale, args.qtrm_logits_scale)
    answer_decision = (
        load_answer_decision_checkpoint(
            args.answer_decision_checkpoint,
            threshold_override=args.answer_decision_threshold,
        )
        if args.answer_decision_checkpoint
        else None
    )
    evidence_source_selector = (
        load_evidence_source_selector_checkpoint(
            args.evidence_source_selector_checkpoint,
            threshold_override=args.evidence_source_selector_threshold,
        )
        if args.evidence_source_selector_checkpoint
        else None
    )

    records = []
    for case in cases:
        for mode in modes:
            record = evaluate_case(
                case=case,
                mode=mode,
                model=model,
                donor=donor,
                tokenizer=tokenizer,
                device=device,
                max_length=max_length,
                max_new_tokens=args.max_new_tokens,
                memory_max_chars=args.memory_max_chars,
                evidence_mode=args.evidence_mode,
                retrieval_top_k=args.retrieval_top_k,
                memory_link_expansion=args.memory_link_expansion,
                memory_index=args.memory_index,
                memory_model_id=args.memory_model_id,
                memory_backend=args.memory_backend,
                hnsw_ef_search=args.hnsw_ef_search,
                retrieve_top_n=args.retrieve_top_n,
                rerank_backend=args.rerank_backend,
                reranker_model_id=args.reranker_model_id,
                reranker_device=args.reranker_device,
                evidence_source_governor=args.evidence_source_governor,
                memoryos_case_filter=not args.memoryos_global,
                evidence_injection=args.evidence_injection,
                base_qtrm_scale=base_qtrm_scale,
                donor_logits_scale=args.donor_logits_scale,
                measure_logit_shift=not args.no_logit_shift,
                core_halt_mode=args.core_halt_mode,
                suppressed_token_ids=suppressed_token_ids,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                short_answer_governor=args.short_answer_governor,
                answer_channel=args.answer_channel,
                evidence_span_max_tokens=args.evidence_span_max_tokens,
                evidence_span_no_answer_threshold=args.evidence_span_no_answer_threshold,
                evidence_span_min_score=args.evidence_span_min_score,
                answer_revision=args.answer_revision,
                answer_revision_max_left_tokens=args.answer_revision_max_left_tokens,
                answer_revision_max_right_tokens=args.answer_revision_max_right_tokens,
                truth_gate=args.truth_gate,
                truth_support_threshold=args.truth_support_threshold,
                truth_causal_threshold=args.truth_causal_threshold,
                truth_refute_threshold=args.truth_refute_threshold,
                truth_missing_threshold=args.truth_missing_threshold,
                answer_decision=answer_decision,
                model_answer_decision=args.model_answer_decision,
                model_answer_decision_threshold=args.model_answer_decision_threshold,
                evidence_source_selector=evidence_source_selector,
                evidence_source_selector_mode=args.evidence_source_selector_mode,
            )
            records.append(record)
            status = "hit" if record["hit"] else "miss"
            retrieval_status = "retrieved" if record["retrieved_target"] else "no-target"
            shift = record.get("first_step_logit_shift") or {}
            shift_text = f" delta={shift.get('max_abs_delta', 0.0):.3f}" if shift else ""
            core_halt = record.get("core_halt") or {}
            core_steps = core_halt.get("core_steps")
            halt_text = f" core_steps={core_steps}" if core_steps is not None else ""
            print(
                f"{status:4s} {record['mode']:28s} {record['id']:22s} "
                f"{retrieval_status:10s}{shift_text}{halt_text} -> {record['completion']!r}"
            )

    summary = summarize_records(records)
    print("\nSummary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.jsonl_out:
        out_path = Path(args.jsonl_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for record in records:
                if not args.print_completions:
                    record = {k: v for k, v in record.items() if k != "full_text"}
                f.write(json.dumps(json_safe_value(record), ensure_ascii=False) + "\n")
            f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
        print(f"\nwrote {out_path}")
    history_path = resolve_history_path(args.history_jsonl_out, kind="evals")
    if history_path is not None:
        for record in records:
            append_jsonl(
                history_path,
                eval_record_to_history_row(
                    record,
                    checkpoint=args.checkpoint,
                    config=args.config,
                    source="memory_retrieval_eval",
                ),
            )
        print(f"appended {len(records)} history rows to {history_path}")
    if args.audit_jsonl_out:
        audit_items = audit_records(records)
        audit_path = Path(args.audit_jsonl_out)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("w", encoding="utf-8") as f:
            for item in audit_items:
                f.write(json.dumps(json_safe_value(item), ensure_ascii=False) + "\n")
        print(f"wrote {audit_path} ({len(audit_items)} audit items)")


if __name__ == "__main__":
    main()
