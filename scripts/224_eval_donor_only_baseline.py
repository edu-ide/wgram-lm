#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_rows(path: str | Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("prompt"):
                raise ValueError(f"{path}:{line_no}: missing prompt")
            if not (row.get("chosen") or row.get("answer")):
                raise ValueError(f"{path}:{line_no}: missing chosen/answer")
            rows.append(row)
            if max_rows is not None and len(rows) >= int(max_rows):
                break
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def normalize_answer(text: Any) -> str:
    value = str(text or "").strip()
    if "\n" in value:
        value = next((part.strip() for part in value.splitlines() if part.strip()), "")
    return value.strip()


def target_answer_from_row(row: Mapping[str, Any]) -> str:
    return normalize_answer(row.get("chosen") or row.get("answer") or "")


def choices_for_row(row: Mapping[str, Any]) -> list[str]:
    raw_choices = row.get("choices")
    if isinstance(raw_choices, list) and raw_choices:
        candidates = [normalize_answer(choice) for choice in raw_choices]
    else:
        candidates = [
            normalize_answer(row.get("chosen") or row.get("answer") or ""),
            normalize_answer(row.get("rejected") or ""),
        ]
    target = target_answer_from_row(row)
    if target:
        candidates.append(target)
    seen: set[str] = set()
    choices: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            choices.append(candidate)
    if not choices:
        raise ValueError("row has no usable choices")
    return choices


def _ratio(hits: int, total: int) -> str:
    return f"{int(hits)}/{int(total)}"


def score_record(
    row: Mapping[str, Any],
    *,
    predicted_answer: str,
    mode: str,
    choice_scores: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target = target_answer_from_row(row)
    completion = normalize_answer(predicted_answer)
    return {
        "id": row.get("id"),
        "source_id": row.get("source_id"),
        "task_family": row.get("task_family") or row.get("category"),
        "mode": str(mode),
        "target_answer": target,
        "completion": completion,
        "hit": completion == target,
        "choice_scores": list(choice_scores),
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, dict[str, int]] = {}
    by_mode_family: dict[str, dict[str, dict[str, int]]] = {}
    mismatches: list[dict[str, Any]] = []
    for record in records:
        mode = str(record.get("mode") or "")
        family = str(record.get("task_family") or "")
        hit = int(bool(record.get("hit")))
        mode_bucket = by_mode.setdefault(mode, {"hits": 0, "total": 0})
        mode_bucket["hits"] += hit
        mode_bucket["total"] += 1
        family_bucket = by_mode_family.setdefault(mode, {}).setdefault(
            family, {"hits": 0, "total": 0}
        )
        family_bucket["hits"] += hit
        family_bucket["total"] += 1
        if not hit and len(mismatches) < 16:
            mismatches.append(dict(record))
    formatted_modes = {
        mode: {
            "exact": _ratio(values["hits"], values["total"]),
            "accuracy": values["hits"] / max(1, values["total"]),
        }
        for mode, values in by_mode.items()
    }
    formatted_families = {
        mode: {
            family: {
                "exact": _ratio(values["hits"], values["total"]),
                "accuracy": values["hits"] / max(1, values["total"]),
            }
            for family, values in families.items()
        }
        for mode, families in by_mode_family.items()
    }
    return {
        "cases": len(records),
        "by_mode": formatted_modes,
        "by_mode_family": formatted_families,
        "mismatch_examples": mismatches,
    }


def _answer_suffix(tokenizer, answer: str) -> str:
    text = str(answer)
    return text if text.startswith((" ", "\n", "\t")) else f" {text}"


def donor_choice_logprob(
    *,
    model,
    tokenizer,
    prompt: str,
    choice: str,
    device,
    max_length: int | None,
) -> float:
    import torch
    import torch.nn.functional as F

    suffix = _answer_suffix(tokenizer, choice)
    target_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if not target_ids:
        return float("-inf")
    full_text = f"{prompt}{suffix}"
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    input_ids = enc["input_ids"]
    if input_ids.shape[1] <= len(target_ids):
        return float("-inf")
    out = model(
        input_ids=input_ids,
        attention_mask=enc.get("attention_mask"),
        use_cache=False,
    )
    logits = out.logits.float()[:, :-1, :]
    labels = input_ids[:, 1:]
    token_logps = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    target_len = min(len(target_ids), token_logps.shape[1])
    return float(token_logps[:, -target_len:].mean().detach().cpu().item())


def donor_forced_choice_answer(
    *,
    model,
    tokenizer,
    prompt: str,
    choices: Sequence[str],
    device,
    max_length: int | None,
) -> tuple[str, list[dict[str, Any]]]:
    scores = [
        {
            "choice": str(choice),
            "logprob": donor_choice_logprob(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                choice=str(choice),
                device=device,
                max_length=max_length,
            ),
        }
        for choice in choices
    ]
    best = max(scores, key=lambda item: float(item["logprob"]))
    return str(best["choice"]), scores


def donor_greedy_answer(
    *,
    model,
    tokenizer,
    prompt: str,
    device,
    max_length: int | None,
    max_new_tokens: int,
) -> str:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    input_len = int(enc["input_ids"].shape[1])
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    out = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=int(max_new_tokens),
        pad_token_id=pad_token_id,
    )
    generated = out[0, input_len:]
    return normalize_answer(tokenizer.decode(generated, skip_special_tokens=True))


def evaluate_rows(
    *,
    rows: Sequence[dict[str, Any]],
    config_path: str,
    modes: Sequence[str],
    max_length: int | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from qtrm_mm.config import load_config
    from qtrm_mm.qwen_donor import QwenDonorAdapter

    cfg = load_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.donor.model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    donor = QwenDonorAdapter(cfg.donor)
    model = donor.model
    if model is None:
        raise RuntimeError("donor model not loaded")
    device = next(model.parameters()).device
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            prompt = str(row["prompt"])
            if "forced_choice" in modes:
                predicted, choice_scores = donor_forced_choice_answer(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    choices=choices_for_row(row),
                    device=device,
                    max_length=max_length,
                )
                records.append(
                    score_record(
                        row,
                        predicted_answer=predicted,
                        mode="forced_choice",
                        choice_scores=choice_scores,
                    )
                )
            if "greedy" in modes:
                predicted = donor_greedy_answer(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                )
                records.append(
                    score_record(
                        row,
                        predicted_answer=predicted,
                        mode="greedy",
                        choice_scores=[],
                    )
                )
    return {
        "config": config_path,
        "donor_model_id": cfg.donor.model_id,
        "modes": list(modes),
        "summary": summarize_records(records),
        "records": records,
    }


def parse_modes(value: str) -> list[str]:
    modes = [part.strip() for part in str(value).split(",") if part.strip()]
    allowed = {"forced_choice", "greedy"}
    unknown = [mode for mode in modes if mode not in allowed]
    if unknown:
        raise ValueError(f"unknown modes: {unknown}")
    if not modes:
        raise ValueError("at least one mode is required")
    return modes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate frozen Qwen donor-only baselines.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--modes", default="forced_choice,greedy")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = load_rows(args.data_jsonl, max_rows=args.max_rows)
    report = evaluate_rows(
        rows=rows,
        config_path=args.config,
        modes=parse_modes(args.modes),
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
