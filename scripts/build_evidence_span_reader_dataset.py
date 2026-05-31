#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from wgram_lm.data.jsonl_dataset import split_memory_prompt_for_workspace


def clean_answer_text(text: str) -> str:
    answer = str(text or "").strip()
    if answer.casefold().startswith("answer:"):
        answer = answer.split(":", 1)[1].strip()
    return answer.strip()


def is_unknown_answer(text: str, *, expected_unknown: bool = False) -> bool:
    answer = clean_answer_text(text)
    return bool(expected_unknown or answer.casefold() == "unknown")


def _candidate_aliases(row: dict[str, Any], answer_text: str) -> list[str]:
    aliases = [str(alias).strip() for alias in (row.get("answer_aliases") or [])]
    aliases.append(answer_text)
    aliases = [alias for alias in aliases if alias and alias.casefold() != "unknown"]
    return sorted(set(aliases), key=len, reverse=True)


def find_answer_span(
    workspace_evidence: str,
    aliases: Iterable[str],
) -> dict[str, Any] | None:
    evidence = str(workspace_evidence or "")
    folded = evidence.casefold()
    for alias in aliases:
        wanted = str(alias or "").strip()
        if not wanted:
            continue
        start = folded.find(wanted.casefold())
        if start < 0:
            continue
        end = start + len(wanted)
        return {
            "start_char": start,
            "end_char": end,
            "text": evidence[start:end],
            "alias": wanted,
        }
    return None


def build_span_reader_row(row: dict[str, Any]) -> dict[str, Any] | None:
    prompt = str(row.get("prompt") or "")
    visible_prompt, workspace_evidence = split_memory_prompt_for_workspace(prompt)
    if not workspace_evidence:
        return None

    raw_answer = row.get("answer") or row.get("chosen") or ""
    answer_text = clean_answer_text(str(raw_answer))
    no_answer = is_unknown_answer(
        answer_text,
        expected_unknown=bool(row.get("expected_unknown")),
    )
    span = None if no_answer else find_answer_span(
        workspace_evidence,
        _candidate_aliases(row, answer_text),
    )

    out = {
        "type": "evidence_span_reader",
        "case_id": row.get("case_id") or row.get("id"),
        "category": row.get("category"),
        "task_family": row.get("task_family"),
        "prompt": visible_prompt,
        "visible_prompt": visible_prompt,
        "workspace_text": workspace_evidence,
        "workspace_evidence": workspace_evidence,
        "answer": f"Answer: {answer_text}",
        "answer_text": answer_text,
        "no_answer": bool(no_answer),
        "answer_span": span,
        "span_status": "no_answer" if no_answer else ("found" if span else "missing"),
        "source_training_scope": row.get("training_scope"),
    }
    return out


def build_span_reader_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    built: list[dict[str, Any]] = []
    for row in rows:
        out = build_span_reader_row(row)
        if out is not None:
            built.append(out)
    return built


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def write_span_reader_rows(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_rows: int = 0,
) -> int:
    rows = build_span_reader_rows(iter_jsonl(input_path))
    if max_rows > 0:
        rows = rows[:max_rows]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build prompt-conditioned evidence-span reader rows from MemoryOS "
            "hidden workspace evidence prompts."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    count = write_span_reader_rows(
        args.input_jsonl,
        args.output_jsonl,
        max_rows=args.max_rows,
    )
    print(f"wrote {count} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
