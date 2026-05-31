from __future__ import annotations

import pytest

from wgram_lm.distill.qwen36_teacher_client import (
    build_teacher_messages,
    parse_teacher_json_response,
)
from wgram_lm.distill.teacher_schema import TeacherEvidenceDoc


def test_build_teacher_messages_requests_structured_distillation_record() -> None:
    messages = build_teacher_messages(
        prompt="What is the capital of France?",
        memory_docs=[TeacherEvidenceDoc(doc_id=2, text="Paris is the capital of France.")],
    )

    assert messages[0]["role"] == "system"
    assert "valid JSON" in messages[0]["content"]
    assert "target_doc_ids" in messages[0]["content"]
    assert "doc_id=2" in messages[1]["content"]


def test_parse_teacher_json_response_merges_base_prompt_and_docs() -> None:
    response = """
    {
      "answer": "Paris",
      "evidence_ids": ["doc-2"],
      "evidence_spans": ["Paris is the capital of France."],
      "rejected_answer": "Lyon",
      "trace_summary": "The cited document directly states the capital.",
      "target_doc_ids": [2]
    }
    """

    record = parse_teacher_json_response(
        response,
        prompt="What is the capital of France?",
        memory_docs=[TeacherEvidenceDoc(doc_id=2, text="Paris is the capital of France.")],
    )

    assert record.prompt == "What is the capital of France?"
    assert record.answer == "Paris"
    assert record.rejected_answer == "Lyon"
    assert record.target_doc_ids == [2]
    assert record.memory_docs[0].doc_id == 2


def test_parse_teacher_json_response_rejects_non_json() -> None:
    with pytest.raises(ValueError, match="JSON"):
        parse_teacher_json_response("Paris", prompt="question")
