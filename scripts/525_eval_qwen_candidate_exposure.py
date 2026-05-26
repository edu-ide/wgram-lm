#!/usr/bin/env python3
"""Evaluate Qwen-generated candidate exposure plus the Stage59 choice verifier.

Typed heuristic candidates are kept only as a deprecated diagnostic scaffold.
They must not be used as a promoted Stage59 path because the final answer path
must learn its working table instead of receiving hand-built candidates.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import itertools
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import torch

from qtrm_mm.eval.general_answer_interface import (
    answer_aliases,
    answer_kind,
    extract_answer_candidate_text,
    normalize_answer_text,
    normalized_alias_set,
    summarize_records,
)


def _load_script(name: str, filename: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


stage523 = _load_script("qtrm_stage523_for_525", "523_train_state_text_speaker.py")
stage524 = _load_script("qtrm_stage524_for_525", "524_train_state_choice_verifier.py")


def configure_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def encode_candidate_strings(
    candidates: list[list[str]],
    *,
    allowed_chars: list[str],
    max_choices: int,
    max_choice_chars: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    char_index = {char: index for index, char in enumerate(allowed_chars)}
    rows: list[list[list[int]]] = []
    masks: list[list[bool]] = []
    for row_candidates in candidates:
        row_ids: list[list[int]] = []
        row_mask: list[bool] = []
        for candidate in row_candidates[: int(max_choices)]:
            ids = [char_index.get(char, 0) for char in str(candidate)[: int(max_choice_chars)]]
            row_ids.append(ids + [0] * (int(max_choice_chars) - len(ids)))
            row_mask.append(True)
        while len(row_ids) < int(max_choices):
            row_ids.append([0] * int(max_choice_chars))
            row_mask.append(False)
        rows.append(row_ids)
        masks.append(row_mask)
    return (
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.bool, device=device),
    )


def dedupe_candidates(candidates: list[str], *, max_candidates: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = extract_answer_candidate_text(candidate)
        normalized = normalize_answer_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(text)
        if len(out) >= int(max_candidates):
            break
    return out


def candidate_prompt_for_row(row: dict[str, Any], *, mode: str) -> str:
    if mode == "row_prompt":
        return stage523.row_prompt(row)
    question = str(row.get("question") or stage523.row_prompt(row)).strip()
    if question.startswith("Answer with only") and "Question:" in question:
        question = question.split("Question:", 1)[1].split("\nAnswer:", 1)[0].strip()
    if mode == "question_answer_only":
        return (
            "Output only one concise final answer. Do not explain.\n"
            f"Question: {question}\n"
            "Answer:"
        )
    if mode == "candidate_proposer":
        return (
            "Propose one short candidate answer. Use only the answer text, no reasoning.\n"
            f"Question: {question}\n"
            "Candidate answer:"
        )
    raise ValueError(f"unsupported candidate prompt mode: {mode}")


def _safe_eval_int_expression(expr: str) -> int | None:
    if not re.fullmatch(r"[0-9\s\+\-\*\/\(\)]+", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.USub,
        ast.UAdd,
        ast.Constant,
    )
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        return None
    try:
        value = eval(compile(tree, "<candidate_expr>", "eval"), {"__builtins__": {}}, {})
    except Exception:
        return None
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    return None


def typed_heuristic_candidates(row: dict[str, Any], *, max_candidates: int) -> list[str]:
    question = str(row.get("question") or stage523.row_prompt(row))
    candidates: list[str] = []

    def add(value: Any) -> None:
        text = str(value).strip()
        if text:
            candidates.append(text)

    # High-priority structured candidates first. Later generic copies are noisy.
    if re.search(r"\bTRUE\b|\bFALSE\b", question, flags=re.IGNORECASE):
        add("TRUE")
        add("FALSE")
    if "EMPTY" in question.upper():
        add("EMPTY")
    for mapped in re.findall(r"maps to\s+([A-Za-z][A-Za-z0-9_-]*)", question, flags=re.IGNORECASE):
        add(mapped)
    # Arithmetic expression candidates.
    expr_match = re.search(r"Compute\s+(.+?)(?:\.|$)", question, flags=re.IGNORECASE)
    if expr_match:
        expr = expr_match.group(1)
        value = _safe_eval_int_expression(expr)
        if value is not None:
            add(value)
            for delta in (-2, -1, 1, 2):
                add(value + delta)
        for part in re.findall(r"-?\d+", expr):
            add(part)

    # List-transform aggregate candidates before single-number distractors.
    list_match = re.search(r"\[([^\]]+)\]", question)
    if list_match:
        list_numbers = [int(item) for item in re.findall(r"-?\d+", list_match.group(1))]
        evens = [number for number in list_numbers if number % 2 == 0]
        if evens:
            add(",".join(str(number) for number in evens))
            add(",".join(str(number * 2) for number in evens))
            add(",".join(str(number) for number in reversed(evens)))
            add(",".join(str(number * 2) for number in reversed(evens)))
        else:
            add("EMPTY")

    # Generic copied symbolic values.
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*\b", question):
        if token.lower() not in {
            "answer",
            "compute",
            "from",
            "the",
            "list",
            "keep",
            "only",
            "even",
            "numbers",
            "double",
            "each",
            "kept",
            "number",
            "and",
            "return",
            "comma",
            "separated",
            "values",
            "with",
            "no",
            "spaces",
            "if",
            "none",
            "maps",
            "map",
            "to",
            "after",
            "two",
            "mappings",
            "what",
            "does",
            "let",
            "evaluate",
            "or",
            "not",
            "true",
            "false",
        }:
            add(token)

    numbers = [int(item) for item in re.findall(r"-?\d+", question)]
    for number in numbers:
        add(number)
        add(number * 2)
    for a, b in itertools.combinations(numbers[:8], 2):
        add(a + b)
        add(a - b)
        add(b - a)
        if abs(a) < 10000 and abs(b) < 10000:
            add(a * b)

    return dedupe_candidates(candidates, max_candidates=max_candidates)


@torch.no_grad()
def generate_candidates_for_row(
    *,
    qwen_model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[str]:
    encoded = tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=192,
        add_special_tokens=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    generated = qwen_model.generate(
        **encoded,
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(float(temperature) > 0.0),
        temperature=max(float(temperature), 1e-5),
        top_p=float(top_p),
        num_return_sequences=int(num_candidates),
        pad_token_id=pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_len = int(encoded["input_ids"].size(1))
    candidates: list[str] = []
    for sequence in generated:
        new_tokens = sequence[prompt_len:]
        candidates.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return candidates


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    rows = stage523.load_jsonl(args.eval_jsonl, limit=int(args.eval_limit))
    qtrm_model, tokenizer, load_stats = stage523.build_qtrm(args, device)
    payload = torch.load(str(args.verifier_checkpoint), map_location=device)
    allowed_chars = list(payload["allowed_chars"])
    verifier_args = payload.get("args") or {}
    max_choice_chars = int(verifier_args.get("max_choice_chars", args.max_choice_chars))
    verifier = stage524.ChoiceVerifier(
        d_state=int(qtrm_model.d_state),
        vocab_size=len(allowed_chars),
        max_choice_chars=max_choice_chars,
    ).to(device)
    verifier.load_state_dict(payload["verifier"], strict=True)
    verifier.eval()
    qtrm_model.eval()
    qtrm_model.qwen.eval()

    records: list[dict[str, Any]] = []
    generated_cache: dict[str, list[str]] = {}
    for start in range(0, len(rows), int(args.batch_size)):
        batch = rows[start : start + int(args.batch_size)]
        batch_candidates: list[list[str]] = []
        for row in batch:
            if args.candidate_source == "typed_heuristic":
                candidates = typed_heuristic_candidates(row, max_candidates=int(args.max_candidates))
            else:
                prompt = candidate_prompt_for_row(row, mode=str(args.candidate_prompt_mode))
                raw_candidates = generate_candidates_for_row(
                    qwen_model=qtrm_model.qwen,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    num_candidates=int(args.num_candidates),
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    device=device,
                )
                candidates = dedupe_candidates(raw_candidates, max_candidates=int(args.max_candidates))
            if not candidates:
                candidates = [""]
            generated_cache[stage523.row_id(row)] = candidates
            batch_candidates.append(candidates)

        context = stage523.thought_context_for_batch(
            qtrm_model,
            tokenizer,
            batch,
            max_length=args.max_length,
            n_steps=args.n_steps,
            device=device,
        )
        choice_ids, choice_mask = encode_candidate_strings(
            batch_candidates,
            allowed_chars=allowed_chars,
            max_choices=int(args.max_candidates),
            max_choice_chars=max_choice_chars,
            device=device,
        )
        logits = verifier(context["readout"], choice_ids, choice_mask)
        pred_indices = logits.argmax(dim=-1).detach().cpu().tolist()
        for row, candidates, pred_index in zip(batch, batch_candidates, pred_indices):
            aliases = answer_aliases(row)
            alias_set = set(normalized_alias_set(aliases))
            normalized_candidates = [normalize_answer_text(candidate) for candidate in candidates]
            oracle_index = next(
                (index for index, candidate in enumerate(normalized_candidates) if candidate in alias_set),
                None,
            )
            selected = candidates[int(pred_index)] if int(pred_index) < len(candidates) else ""
            normalized_selected = normalize_answer_text(selected)
            first = candidates[0] if candidates else ""
            records.append(
                {
                    "id": stage523.row_id(row),
                    "task_family": row.get("task_family") or row.get("family") or row.get("category") or "unknown",
                    "answer_kind": answer_kind(aliases[0] if aliases else ""),
                    "aliases": list(aliases),
                    "candidates": candidates,
                    "normalized_candidates": normalized_candidates,
                    "oracle_exact": oracle_index is not None,
                    "oracle_index": oracle_index,
                    "selected": selected,
                    "normalized_selected": normalized_selected,
                    "selected_index": int(pred_index),
                    "exact": normalized_selected in alias_set,
                    "first_selected": first,
                    "first_exact": normalize_answer_text(first) in alias_set,
                    "selection_mode": "qwen_generated_candidates_plus_qtrm_verifier",
                }
            )

    summary = summarize_records(records)
    oracle_hits = sum(1 for row in records if bool(row.get("oracle_exact")))
    first_hits = sum(1 for row in records if bool(row.get("first_exact")))
    summary.update(
        {
            "stage": "Stage59 Qwen candidate exposure plus QTRM verifier",
            "oracle_coverage": float(oracle_hits / max(1, len(records))),
            "first_candidate_accuracy": float(first_hits / max(1, len(records))),
            "eval_jsonl": str(args.eval_jsonl),
            "verifier_checkpoint": str(args.verifier_checkpoint),
            "load_stats": load_stats,
            "plain_language_read": (
                "This asks whether Qwen can put the right answer somewhere on the table, "
                "so the QTRM verifier can choose it. If oracle_coverage is low, the next "
                "bottleneck is candidate exposure, not selection."
            ),
        }
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/mnt/sdc1/tripleyoung/qtrm_eval/20260522_092733_LOCAL_STAGE54B_oracle_guard_mixedall_seed42/best_stochastic_oracle.pt")
    parser.add_argument("--verifier-checkpoint", default="local_eval/stage59_local_choice_verifier_shuffled_frozen_saved_t512_e128_ep16_s1608/best_choice_verifier.pt")
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--eval-jsonl", default="scratch/stage59/shuffled_choices_eval.jsonl")
    parser.add_argument("--out-dir", default="local_eval/stage59_qwen_candidate_exposure")
    parser.add_argument("--eval-limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--max-choice-chars", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--candidate-prompt-mode", choices=("row_prompt", "question_answer_only", "candidate_proposer"), default="row_prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--candidate-source", choices=("qwen_generate", "typed_heuristic"), default="qwen_generate")
    parser.add_argument(
        "--allow-diagnostic-scaffold",
        action="store_true",
        help="Required to run deprecated typed_heuristic candidates; final-only experiments must not use it.",
    )
    parser.add_argument("--seed", type=int, default=1525)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--core-impl", default="state_transition")
    parser.add_argument("--core-update", default="mlp")
    parser.add_argument("--answer-path", choices=("state_head", "lm_head"), default="lm_head")
    parser.add_argument("--workspace-pooling", default="sequence")
    parser.add_argument("--recurrent-readout-pooling", default="sharp_attention")
    parser.add_argument("--recurrent-readout-temperature", type=float, default=0.25)
    parser.add_argument("--n-steps", type=int, default=14)
    parser.add_argument("--state-update-schedule", default="nested")
    parser.add_argument("--stochastic-high-level-guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stochastic-high-level-scale", type=float, default=1.0)
    parser.add_argument("--stochastic-high-level-min-std", type=float, default=1e-4)
    parser.add_argument("--stochastic-high-level-max-std", type=float, default=1.0)
    parser.add_argument("--stochastic-high-level-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stochastic-posterior-guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stochastic-transition-mode", choices=("delta", "true_gram"), default="true_gram")
    args = parser.parse_args()

    if args.candidate_source == "typed_heuristic" and not args.allow_diagnostic_scaffold:
        raise SystemExit(
            "typed_heuristic is a deprecated diagnostic scaffold. "
            "Use --allow-diagnostic-scaffold only for audit/reproduction, not final-path experiments."
        )

    configure_seed(args.seed)
    summary = evaluate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
