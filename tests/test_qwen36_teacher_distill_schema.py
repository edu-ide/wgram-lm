from __future__ import annotations

import json

import pytest

from wgram_lm.distill.teacher_schema import (
    Qwen36TeacherRecord,
    TeacherEvidenceDoc,
    TeacherTopKLogprob,
    validate_teacher_record,
)


def test_teacher_record_round_trips_qtrm_and_msa_fields() -> None:
    record = Qwen36TeacherRecord(
        prompt="Which city is the capital of France?",
        answer="Paris",
        teacher_model="Qwen/Qwen3.6-27B",
        evidence_ids=["doc-2"],
        evidence_spans=["doc-2 says Paris is the capital of France."],
        rejected_answer="Lyon",
        trace_summary="Use the capital-city evidence and reject the distractor.",
        memory_docs=[
            TeacherEvidenceDoc(doc_id=1, text="Lyon is a city in France."),
            TeacherEvidenceDoc(doc_id=2, text="Paris is the capital of France."),
        ],
        target_doc_ids=[2],
        topk_logprobs=[
            TeacherTopKLogprob(position=0, token_id=123, logprob=-0.1),
            TeacherTopKLogprob(position=0, token_id=456, logprob=-2.4),
        ],
    )

    payload = record.to_dict()
    restored = Qwen36TeacherRecord.from_dict(json.loads(json.dumps(payload)))

    assert restored.prompt == record.prompt
    assert restored.answer == "Paris"
    assert restored.evidence_ids == ["doc-2"]
    assert restored.memory_docs[1].doc_id == 2
    assert restored.target_doc_ids == [2]
    assert restored.topk_logprobs[0].token_id == 123


def test_validate_teacher_record_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="prompt"):
        validate_teacher_record({"prompt": "", "answer": "Paris"})


def test_validate_teacher_record_requires_msa_targets_to_exist_in_memory_docs() -> None:
    with pytest.raises(ValueError, match="target_doc_ids"):
        validate_teacher_record(
            {
                "prompt": "question",
                "answer": "answer",
                "memory_docs": [{"doc_id": 1, "text": "only doc"}],
                "target_doc_ids": [2],
            }
        )
