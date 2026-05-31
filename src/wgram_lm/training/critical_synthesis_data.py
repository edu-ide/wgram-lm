from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from wgram_lm.eval.critical_synthesis import (
    build_critical_synthesis_prompt,
    build_critical_synthesis_target,
    load_critical_synthesis_cases,
)


def build_critical_synthesis_trace_rows(
    cases: Iterable[dict[str, Any]],
    *,
    max_evidence_chars: int = 4000,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.append(
            {
                "type": "critical_synthesis_trace",
                "case_id": case.get("id"),
                "domain": case.get("domain", "general"),
                "prompt": build_critical_synthesis_prompt(
                    case,
                    max_evidence_chars=max_evidence_chars,
                ),
                "answer": build_critical_synthesis_target(case),
                "source_evidence": [rec.get("source", "") for rec in case.get("evidence") or []],
            }
        )
    return rows


def write_critical_synthesis_trace_jsonl(
    cases_path: str | Path,
    out_path: str | Path,
    *,
    max_evidence_chars: int = 4000,
) -> int:
    rows = build_critical_synthesis_trace_rows(
        load_critical_synthesis_cases(cases_path),
        max_evidence_chars=max_evidence_chars,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)
