from __future__ import annotations

import json
import re

from .teacher_schema import Qwen36TeacherRecord


def build_online_teacher_prompt(prompt: str) -> str:
    return (
        "Return only the final answer. Do not explain. Do not include reasoning.\n\n"
        f"{str(prompt).strip()}"
    )


def clean_teacher_answer(text: str, *, max_chars: int = 256) -> str:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json|text)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    json_answer = _json_answer(stripped)
    if json_answer:
        return json_answer[:max_chars]
    stripped = re.sub(r"^(?:final\s+)?answer\s*:\s*", "", stripped, flags=re.IGNORECASE)
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        raise ValueError("teacher generated an empty answer")
    answer = lines[0]
    answer = re.split(r"\s+(?:Explanation|Reasoning|Trace)\s*:", answer, maxsplit=1)[0].strip()
    answer = answer.strip("`'\" ")
    if not answer:
        raise ValueError("teacher generated an empty answer")
    return answer[:max_chars]


def teacher_answer_record(
    *,
    prompt: str,
    answer: str,
    teacher_model: str,
) -> Qwen36TeacherRecord:
    return Qwen36TeacherRecord(
        prompt=str(prompt),
        answer=clean_teacher_answer(answer),
        teacher_model=str(teacher_model),
        trace_summary="online_teacher_hard_answer",
    )


def _json_answer(text: str) -> str:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            return ""
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return ""
    if not isinstance(data, dict):
        return ""
    answer = str(data.get("answer", "")).strip()
    return answer.strip("`'\" ")
