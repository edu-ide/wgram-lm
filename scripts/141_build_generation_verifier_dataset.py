#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence


STOP_SENSITIVE_PATTERNS = (
    r"\bonce only\b",
    r"\bdirect answer\b",
    r"\bone word\b",
    r"\bexactly one\b",
    r"\bthen stop\b",
    r"\band stop\b",
    r"\bonly\.",
    r"정답만",
    r"한 문장",
)

VISIBLE_REASONING_PATTERNS = (
    r"<\s*/?\s*think\s*>",
    r"\blet me\b",
    r"\bi need to\b",
    r"\bthe user (is|asked|wants|needs)\b",
)

ANSWER_DRIFT_PATTERNS = (
    r"(^|\n)\s*a\.\s+.+(\n|\s+)b\.\s+",
    r"\n\s*(what|why|how|when|where|who|which)\b.+\?",
    r"\bpls answer\b",
    r"\bplease answer\b",
    r"\bquestion\s*\d*\s*:",
    r"\banswer in a minimum\b",
    r"\bdo not (mention|reveal|add)\b",
    r"\bif the user'?s question\b",
    r"\byou should reply\b",
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Build QTRM-generated output verifier training rows from eval JSONL."
    )
    ap.add_argument("--eval-jsonl", required=True, help="JSONL from scripts/92_eval_qtrm_logits.py --json")
    ap.add_argument("--out", required=True, help="Output verifier JSONL path")
    ap.add_argument("--prompt-meta-jsonl", default=None, help="Optional prompt suite JSONL with prompt_id/category")
    ap.add_argument("--repeat-threshold", type=float, default=0.15)
    ap.add_argument("--severe-repeat-threshold", type=float, default=0.25)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    return ap


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_prompt_meta(path: str | Path | None) -> dict[int, dict]:
    if path is None:
        return {}
    meta = {}
    for row in load_jsonl(path):
        prompt_id = int(row.get("prompt_id", row.get("sample", len(meta))))
        meta[prompt_id] = row
    return meta


def is_stop_sensitive_prompt(prompt: str, *, category: str | None = None) -> bool:
    if category == "repeat_stress":
        return True
    text = str(prompt or "").lower()
    return any(re.search(pattern, text) for pattern in STOP_SENSITIVE_PATTERNS)


def has_visible_reasoning(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(re.search(pattern, lowered) for pattern in VISIBLE_REASONING_PATTERNS)


def has_answer_drift(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(re.search(pattern, lowered, flags=re.DOTALL) for pattern in ANSWER_DRIFT_PATTERNS)


def completion_text(*, prompt: str, generated_text: str) -> str:
    prompt = str(prompt or "")
    generated_text = str(generated_text or "")
    if prompt and generated_text.startswith(prompt):
        return generated_text[len(prompt) :]
    return generated_text


def generation_verifier_targets(
    row: dict,
    *,
    category: str | None = None,
    max_new_tokens: int = 64,
    repeat_threshold: float = 0.15,
    severe_repeat_threshold: float = 0.25,
) -> dict:
    repetition = row.get("greedy_repetition") or {}
    repeated_2gram_rate = float(repetition.get("repeated_2gram_rate") or 0.0)
    repeated_3gram_rate = float(repetition.get("repeated_3gram_rate") or 0.0)
    completion_tokens = int(repetition.get("completion_tokens") or 0)
    prompt = str(row.get("text") or "")
    generated_text = str(row.get("greedy_text") or "")
    completion = completion_text(prompt=prompt, generated_text=generated_text)
    repeat_failure = repeated_2gram_rate >= float(repeat_threshold)
    severe_repeat = repeated_2gram_rate >= float(severe_repeat_threshold)
    stop_failure = (
        completion_tokens >= int(max_new_tokens)
        and is_stop_sensitive_prompt(prompt, category=category)
    )
    format_failure = has_visible_reasoning(completion)
    answer_drift_failure = has_answer_drift(completion)
    quality_pass = (
        not repeat_failure
        and not stop_failure
        and not format_failure
        and not answer_drift_failure
    )
    sample_weight = 1.0
    if repeat_failure:
        sample_weight += 0.5
    if severe_repeat:
        sample_weight += 0.5
    if stop_failure:
        sample_weight += 0.5
    if format_failure:
        sample_weight += 0.5
    if answer_drift_failure:
        sample_weight += 0.5
    return {
        "generation_verifier_repeat_target": 1.0 if repeat_failure else 0.0,
        "generation_verifier_stop_target": 1.0 if stop_failure else 0.0,
        "generation_verifier_quality_target": 1.0 if quality_pass else 0.0,
        "generation_verifier_sample_weight": sample_weight,
        "repeated_2gram_rate": repeated_2gram_rate,
        "repeated_3gram_rate": repeated_3gram_rate,
        "completion_tokens": completion_tokens,
        "severe_repeat": bool(severe_repeat),
        "format_failure": bool(format_failure),
        "answer_drift_failure": bool(answer_drift_failure),
    }


def build_rows(
    eval_rows: Sequence[dict],
    *,
    prompt_meta: dict[int, dict] | None = None,
    max_new_tokens: int = 64,
    repeat_threshold: float = 0.15,
    severe_repeat_threshold: float = 0.25,
) -> list[dict]:
    prompt_meta = prompt_meta or {}
    out = []
    for idx, row in enumerate(eval_rows):
        sample = int(row.get("sample", idx))
        meta = prompt_meta.get(sample, {})
        category = meta.get("category")
        targets = generation_verifier_targets(
            row,
            category=category,
            max_new_tokens=max_new_tokens,
            repeat_threshold=repeat_threshold,
            severe_repeat_threshold=severe_repeat_threshold,
        )
        prompt = str(row.get("text") or meta.get("text") or "")
        generated_text = str(row.get("greedy_text") or prompt)
        out.append(
            {
                "text": generated_text,
                "prompt": prompt,
                "source_sample": sample,
                "candidate_id": row.get("candidate_id"),
                "category": category,
                "distill_source": "qtrm_generated_verifier",
                **targets,
            }
        )
    return out


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    eval_rows = load_jsonl(args.eval_jsonl)
    rows = build_rows(
        eval_rows,
        prompt_meta=load_prompt_meta(args.prompt_meta_jsonl),
        max_new_tokens=args.max_new_tokens,
        repeat_threshold=args.repeat_threshold,
        severe_repeat_threshold=args.severe_repeat_threshold,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {
        "rows": len(rows),
        "repeat_failures": sum(row["generation_verifier_repeat_target"] == 1.0 for row in rows),
        "stop_failures": sum(row["generation_verifier_stop_target"] == 1.0 for row in rows),
        "format_failures": sum(bool(row.get("format_failure")) for row in rows),
        "answer_drift_failures": sum(bool(row.get("answer_drift_failure")) for row in rows),
        "quality_pass": sum(row["generation_verifier_quality_target"] == 1.0 for row in rows),
    }
    print(json.dumps({"out": str(out), **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
