from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_TEACHER_MODEL = "Qwen/Qwen3.6-27B"


@dataclass(frozen=True)
class TeacherEvidenceDoc:
    doc_id: int
    text: str
    source: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeacherEvidenceDoc":
        return cls(
            doc_id=int(data["doc_id"]),
            text=str(data["text"]),
            source=str(data.get("source", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"doc_id": self.doc_id, "text": self.text}
        if self.source:
            payload["source"] = self.source
        return payload


@dataclass(frozen=True)
class TeacherTopKLogprob:
    position: int
    token_id: int
    logprob: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeacherTopKLogprob":
        return cls(
            position=int(data["position"]),
            token_id=int(data["token_id"]),
            logprob=float(data["logprob"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "token_id": self.token_id,
            "logprob": self.logprob,
        }


@dataclass(frozen=True)
class Qwen36TeacherRecord:
    prompt: str
    answer: str
    teacher_model: str = DEFAULT_TEACHER_MODEL
    evidence_ids: list[str] = field(default_factory=list)
    evidence_spans: list[str] = field(default_factory=list)
    rejected_answer: str = ""
    trace_summary: str = ""
    memory_docs: list[TeacherEvidenceDoc] = field(default_factory=list)
    target_doc_ids: list[int] = field(default_factory=list)
    topk_logprobs: list[TeacherTopKLogprob] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen36TeacherRecord":
        validate_teacher_record(data)
        return cls(
            prompt=str(data["prompt"]),
            answer=str(data["answer"]),
            teacher_model=str(data.get("teacher_model", DEFAULT_TEACHER_MODEL)),
            evidence_ids=[str(item) for item in data.get("evidence_ids", [])],
            evidence_spans=[str(item) for item in data.get("evidence_spans", [])],
            rejected_answer=str(data.get("rejected_answer", "")),
            trace_summary=str(data.get("trace_summary", "")),
            memory_docs=[
                TeacherEvidenceDoc.from_dict(item)
                for item in data.get("memory_docs", [])
            ],
            target_doc_ids=[int(item) for item in data.get("target_doc_ids", [])],
            topk_logprobs=[
                TeacherTopKLogprob.from_dict(item)
                for item in data.get("topk_logprobs", [])
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": self.prompt,
            "answer": self.answer,
            "teacher_model": self.teacher_model,
        }
        if self.evidence_ids:
            payload["evidence_ids"] = list(self.evidence_ids)
        if self.evidence_spans:
            payload["evidence_spans"] = list(self.evidence_spans)
        if self.rejected_answer:
            payload["rejected_answer"] = self.rejected_answer
        if self.trace_summary:
            payload["trace_summary"] = self.trace_summary
        if self.memory_docs:
            payload["memory_docs"] = [doc.to_dict() for doc in self.memory_docs]
        if self.target_doc_ids:
            payload["target_doc_ids"] = list(self.target_doc_ids)
        if self.topk_logprobs:
            payload["topk_logprobs"] = [
                item.to_dict() for item in self.topk_logprobs
            ]
        validate_teacher_record(payload)
        return payload


def validate_teacher_record(data: dict[str, Any]) -> None:
    prompt = str(data.get("prompt", "")).strip()
    answer = str(data.get("answer", "")).strip()
    if not prompt:
        raise ValueError("teacher record requires a non-empty prompt")
    if not answer:
        raise ValueError("teacher record requires a non-empty answer")

    _validate_str_list(data, "evidence_ids")
    _validate_str_list(data, "evidence_spans")

    memory_docs = data.get("memory_docs", [])
    if not isinstance(memory_docs, list):
        raise ValueError("memory_docs must be a list")
    doc_ids = set()
    for idx, doc in enumerate(memory_docs):
        if not isinstance(doc, dict):
            raise ValueError(f"memory_docs[{idx}] must be an object")
        if "doc_id" not in doc:
            raise ValueError(f"memory_docs[{idx}] requires doc_id")
        if not str(doc.get("text", "")).strip():
            raise ValueError(f"memory_docs[{idx}] requires non-empty text")
        doc_ids.add(int(doc["doc_id"]))

    target_doc_ids = data.get("target_doc_ids", [])
    if not isinstance(target_doc_ids, list):
        raise ValueError("target_doc_ids must be a list")
    missing_targets = [int(doc_id) for doc_id in target_doc_ids if int(doc_id) not in doc_ids]
    if missing_targets:
        raise ValueError(
            "target_doc_ids must refer to memory_docs; missing "
            f"{missing_targets}"
        )

    topk_logprobs = data.get("topk_logprobs", [])
    if not isinstance(topk_logprobs, list):
        raise ValueError("topk_logprobs must be a list")
    for idx, item in enumerate(topk_logprobs):
        if not isinstance(item, dict):
            raise ValueError(f"topk_logprobs[{idx}] must be an object")
        for key in ("position", "token_id", "logprob"):
            if key not in item:
                raise ValueError(f"topk_logprobs[{idx}] requires {key}")


def _validate_str_list(data: dict[str, Any], key: str) -> None:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{key}[{idx}] must be a string")
