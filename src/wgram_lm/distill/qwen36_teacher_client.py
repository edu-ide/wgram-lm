from __future__ import annotations

import json
import re
from typing import Any

from .teacher_schema import (
    DEFAULT_TEACHER_MODEL,
    Qwen36TeacherRecord,
    TeacherEvidenceDoc,
)


TEACHER_SYSTEM_PROMPT = """You are generating QTRM/MSA distillation data.
Return only valid JSON with these fields when available:
answer, evidence_ids, evidence_spans, rejected_answer, trace_summary,
target_doc_ids.

Rules:
- answer must be concise and factually grounded.
- evidence_ids should name the documents used, e.g. "doc-2".
- target_doc_ids should contain integer doc ids that should be routed by MSA.
- rejected_answer should be a plausible but wrong answer when a distractor
  exists.
- trace_summary should be short and describe the evidence logic, not a long
  hidden chain of thought.
"""


def build_teacher_messages(
    *,
    prompt: str,
    memory_docs: list[TeacherEvidenceDoc] | None = None,
) -> list[dict[str, str]]:
    content = [f"Prompt:\n{prompt.strip()}"]
    docs = memory_docs or []
    if docs:
        content.append("Memory docs:")
        for doc in docs:
            source = f" source={doc.source}" if doc.source else ""
            content.append(f"doc_id={doc.doc_id}{source}\n{doc.text.strip()}")
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(content)},
    ]


def parse_teacher_json_response(
    response_text: str,
    *,
    prompt: str,
    memory_docs: list[TeacherEvidenceDoc] | None = None,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
) -> Qwen36TeacherRecord:
    data = _loads_json_object(response_text)
    data["prompt"] = prompt
    data.setdefault("teacher_model", teacher_model)
    if memory_docs is not None:
        data["memory_docs"] = [doc.to_dict() for doc in memory_docs]
    return Qwen36TeacherRecord.from_dict(data)


def _loads_json_object(response_text: str) -> dict[str, Any]:
    stripped = response_text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            raise ValueError("teacher response did not contain a JSON object") from exc
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("teacher response JSON must be an object")
    return data
