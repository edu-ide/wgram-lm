from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .teacher_schema import Qwen36TeacherRecord, TeacherEvidenceDoc


MISSING_ANSWER_MARKERS = {
    "unknown",
    "needs_search",
    "need_search",
    "insufficient_evidence",
    "not_enough_information",
    "cannot_determine",
}
REFUTE_ANSWER_MARKERS = {"refuted", "unsupported", "false", "contradicted"}
SUPPORT_ANSWER_MARKERS = {"supported", "true", "entailed", "faithful"}


def teacher_record_to_training_row(
    record: Qwen36TeacherRecord,
    *,
    source_name: str = "",
    include_evidence: bool = True,
    max_evidence_chars: int = 4000,
    preference_weight: float = 1.0,
) -> dict[str, Any]:
    prompt = record.prompt
    evidence_context = ""
    has_memory = bool(record.memory_docs)
    has_grounded_evidence = bool(record.target_doc_ids or record.evidence_spans)
    unsupported_with_memory = has_memory and not has_grounded_evidence
    if include_evidence and has_memory:
        evidence_context = format_memory_docs(
            record.memory_docs,
            max_chars=max_evidence_chars,
        )
        prompt = build_workspace_prompt(record.prompt, evidence_context)

    answer = record.answer
    rejected = record.rejected_answer
    unsupported_answer = ""
    if unsupported_with_memory:
        unsupported_answer = answer
        answer = "NEEDS_SEARCH"
        rejected = rejected or unsupported_answer

    row: dict[str, Any] = {
        "prompt": prompt,
        "teacher_model": record.teacher_model,
    }
    if source_name:
        row["distill_source"] = source_name

    if rejected:
        row["chosen"] = answer
        row["rejected"] = rejected
        row["preference_weight"] = float(preference_weight)
    else:
        row["answer"] = answer

    if unsupported_answer:
        row["unsupported_answer"] = unsupported_answer
    if record.trace_summary:
        row["trace_summary"] = record.trace_summary
    if record.evidence_ids:
        row["evidence_ids"] = list(record.evidence_ids)
    if record.evidence_spans:
        row["evidence_spans"] = list(record.evidence_spans)
    if record.target_doc_ids:
        row["target_doc_ids"] = list(record.target_doc_ids)
    if record.memory_docs:
        row["memory_docs"] = [doc.to_dict() for doc in record.memory_docs]
    if record.topk_logprobs:
        row["topk_logprobs"] = [item.to_dict() for item in record.topk_logprobs]

    if evidence_context:
        row.update(
            _logical_targets(
                answer,
                has_grounded_evidence=has_grounded_evidence,
                unsupported_with_memory=unsupported_with_memory,
            )
        )
    return row


def format_memory_docs(docs: list[TeacherEvidenceDoc], *, max_chars: int = 4000) -> str:
    lines = ["MemoryOS evidence"]
    for idx, doc in enumerate(docs, start=1):
        source = doc.source or f"doc_{doc.doc_id or idx}"
        text = str(doc.text).replace("\n", " ").strip()
        header = f"SOURCE={source} CHUNK={int(doc.doc_id)} SCORE=1.0000"
        block = f"{header}\n{text}"
        current = "\n".join(lines)
        tentative = f"{current}\n{block}"
        if len(tentative) <= max_chars:
            lines.append(block)
            continue
        remaining = max_chars - len(current) - len(header) - 2
        if remaining > 0:
            lines.append(f"{header}\n{text[:remaining]}")
        break
    return "\n".join(lines)[:max_chars]


def build_workspace_prompt(prompt: str, memory_context: str) -> str:
    if not memory_context:
        return prompt
    return (
        f"{memory_context}\n\n"
        "Use the evidence above when it is relevant. If it is not relevant, answer from the prompt.\n\n"
        f"User prompt:\n{prompt}"
    )


def build_training_mix(
    *,
    inputs: list[Path | str],
    out_path: Path | str,
    max_rows_per_source: int = 0,
    include_evidence: bool = True,
    max_evidence_chars: int = 4000,
    preference_weight: float = 1.0,
) -> dict[str, Any]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    source_counts: Counter[str] = Counter()
    read = 0
    skipped = 0
    written = 0
    with out.open("w", encoding="utf-8") as f:
        for path_like in inputs:
            path = Path(path_like)
            source_name = _source_name(path)
            for payload in _iter_jsonl(path):
                read += 1
                if max_rows_per_source > 0 and source_counts[source_name] >= max_rows_per_source:
                    continue
                try:
                    record = Qwen36TeacherRecord.from_dict(payload)
                    row = teacher_record_to_training_row(
                        record,
                        source_name=source_name,
                        include_evidence=include_evidence,
                        max_evidence_chars=max_evidence_chars,
                        preference_weight=preference_weight,
                    )
                except Exception:
                    skipped += 1
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                source_counts[source_name] += 1
                written += 1
    return {
        "read": read,
        "skipped": skipped,
        "written": written,
        "sources": dict(source_counts),
        "out": str(out),
    }


def _logical_targets(
    answer: str,
    *,
    has_grounded_evidence: bool,
    unsupported_with_memory: bool,
) -> dict[str, float]:
    normalized = _normalize_answer_marker(answer)
    missing = unsupported_with_memory or normalized in MISSING_ANSWER_MARKERS
    refuted = normalized in REFUTE_ANSWER_MARKERS
    supported = normalized in SUPPORT_ANSWER_MARKERS or (
        has_grounded_evidence and not missing and not refuted
    )
    return {
        "logical_support_target": 1.0 if supported else 0.0,
        "logical_refute_target": 1.0 if refuted else 0.0,
        "logical_missing_target": 1.0 if missing else 0.0,
        "causal_evidence_target": 1.0 if has_grounded_evidence and not missing else 0.0,
    }


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _source_name(path: Path) -> str:
    name = path.name
    for suffix in (".jsonl", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    for suffix in ("_s100", "_s1000", "_s10000"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _normalize_answer_marker(answer: str) -> str:
    return "_".join(str(answer).strip().lower().split())
