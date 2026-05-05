#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build workspace-counterfactual preference rows by pairing each "
            "MemoryOS evidence block with another row's evidence block."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def split_memory_prompt_for_workspace(prompt: str) -> tuple[str, str]:
    text = str(prompt or "")
    marker = "\n\nUser prompt:\n"
    if not text.startswith("MemoryOS evidence") or marker not in text:
        return text, ""
    before_user_prompt, visible_prompt = text.split(marker, 1)
    evidence_text = before_user_prompt.split("\n\nUse the evidence above", 1)[0].strip()
    if not evidence_text:
        return text, ""
    return visible_prompt.strip(), evidence_text


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            _, evidence = split_memory_prompt_for_workspace(str(row.get("prompt") or ""))
            if evidence:
                row = dict(row)
                row["_workspace_evidence_text"] = evidence
                rows.append(row)
    return rows


def choose_counterfactual(
    rows: list[dict[str, Any]],
    index: int,
) -> dict[str, Any] | None:
    row = rows[index]
    chosen = row.get("chosen")
    answer_text = normalize_answer_text(str(chosen or ""))
    case_id = row.get("case_id") or row.get("id")
    for offset in range(1, len(rows)):
        candidate = rows[(index + offset) % len(rows)]
        candidate_case_id = candidate.get("case_id") or candidate.get("id")
        if candidate_case_id == case_id:
            continue
        if candidate.get("chosen") == chosen:
            continue
        if answer_text and answer_text in str(candidate.get("_workspace_evidence_text") or ""):
            continue
        return candidate
    return None


def normalize_answer_text(chosen: str) -> str:
    text = chosen.strip()
    if text.lower().startswith("answer:"):
        text = text.split(":", 1)[1].strip()
    return text


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_jsonl)
    if len(rows) < 2:
        raise SystemExit("need at least two MemoryOS evidence rows")

    limit = len(rows) if args.max_rows <= 0 else min(int(args.max_rows), len(rows))
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows[:limit]):
            counterfactual = choose_counterfactual(rows, index)
            if counterfactual is None:
                continue
            out = {
                key: value
                for key, value in row.items()
                if key != "_workspace_evidence_text"
            }
            out["counterfactual_workspace_text"] = counterfactual["_workspace_evidence_text"]
            out["counterfactual_case_id"] = (
                counterfactual.get("case_id") or counterfactual.get("id")
            )
            out["training_scope"] = "workspace_counterfactual"
            out["workspace_counterfactual_policy"] = "next_different_case_different_answer"
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {written} rows to {out_path}")


if __name__ == "__main__":
    main()
