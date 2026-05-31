from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

SYNTHESIS_STAGES = (
    "critique",
    "preserve",
    "risks",
    "reframe",
    "positive_conclusion",
)


def _lines(title: str, values: Iterable[Any]) -> list[str]:
    items = [str(value).strip() for value in values if str(value).strip()]
    if not items:
        return [f"{title}: none specified"]
    if len(items) == 1:
        return [f"{title}: {items[0]}"]
    return [f"{title}:"] + [f"- {item}" for item in items]


def _format_evidence(case: dict[str, Any], *, max_chars: int) -> str:
    lines = ["Critical synthesis evidence"]
    for idx, rec in enumerate(case.get("evidence") or []):
        source = rec.get("source", f"evidence-{idx}")
        kind = rec.get("source_type", "local_note")
        text = str(rec.get("text", "")).replace("\n", " ").strip()
        block = f"SOURCE={source} TYPE={kind}\n{text}"
        current = "\n".join(lines)
        tentative = f"{current}\n{block}"
        if len(tentative) <= max_chars:
            lines.append(block)
            continue
        remaining = max_chars - len(current) - len(source) - len(kind) - 16
        if remaining > 0:
            lines.append(f"SOURCE={source} TYPE={kind}\n{text[:remaining]}")
        break
    return "\n".join(lines)[:max_chars]


def build_critical_synthesis_prompt(case: dict[str, Any], *, max_evidence_chars: int = 4000) -> str:
    evidence = _format_evidence(case, max_chars=max_evidence_chars)
    question = str(case.get("question", "")).strip()
    domain = str(case.get("domain", "general")).strip()
    return (
        f"{evidence}\n\n"
        "You are a critical synthesis reasoner.\n"
        "Do not stop at suspicion. Do not blindly accept the old tradition or the new frame.\n"
        "Critique what is controlling, fear-based, contradictory, or weakly grounded.\n"
        "Preserve what is liberating, compassionate, coherent, or practically useful.\n"
        "Mark unverifiable metaphysical claims as symbolic or hypothesis-level claims.\n"
        "End with a constructive positive conclusion.\n"
        "Return this shape:\n"
        "Critique: <problems or tensions>\n"
        "Preserve: <values worth keeping>\n"
        "Risks: <risks of both old authority and new overclaim>\n"
        "Reframe: <balanced new view>\n"
        "Positive conclusion: <practical constructive conclusion>\n\n"
        f"Domain: {domain}\n"
        f"Question: {question}"
    ).strip()


def build_critical_synthesis_target(case: dict[str, Any]) -> str:
    parts: list[str] = []
    parts.extend(_lines("Critique", case.get("critique_points") or []))
    parts.extend(_lines("Preserve", case.get("preserve_values") or []))
    parts.extend(_lines("Risks", case.get("risk_notes") or []))
    reframe = str(case.get("reframe") or "").strip()
    if reframe:
        parts.append(f"Reframe: {reframe}")
    conclusion = str(case.get("positive_conclusion") or "").strip()
    parts.append(f"Positive conclusion: {conclusion}")
    return "\n".join(parts)


def load_critical_synthesis_cases(path: str | Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if not case.get("id"):
                case["id"] = f"critical-synthesis-{line_no}"
            for field in ("question", "critique_points", "preserve_values", "risk_notes", "positive_conclusion"):
                if not case.get(field):
                    raise ValueError(f"{path}:{line_no}: missing {field}")
            cases.append(case)
    return cases
