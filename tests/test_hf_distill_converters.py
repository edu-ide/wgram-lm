from __future__ import annotations

import pytest

from wgram_lm.distill.hf_dataset_convert import convert_hf_row


def test_yana_reasoning_dpo_converter_extracts_chosen_rejected() -> None:
    row = {
        "prompt": "<question>What is 2+2?</question>",
        "chosen": "<Problem>What is 2+2?</Problem><Thinking>Add two and two.</Thinking><Answer>\\boxed{4}</Answer>",
        "rejected": "<Problem>What is 2+2?</Problem><Thinking></Thinking><Answer>\\boxed{5}</Answer>",
        "strategy": "tamper_answer",
    }

    record = convert_hf_row(row, adapter="yana_reasoning_dpo")

    assert record.prompt == "What is 2+2?"
    assert record.answer == "4"
    assert record.rejected_answer == "5"
    assert "Add two and two" in record.trace_summary
    assert record.teacher_model == "hf:Yana/ft-llm-2026-reasoning-dpo"


def test_noesis_text_sft_converter_splits_user_assistant_text() -> None:
    row = {
        "text": "User: 한국어는 어떻게 발전했나요?\nAssistant: <think>언어 변화와 차용어를 고려한다.</think>한국어는 여러 시대의 변화 속에서 발전했습니다.",
        "domain": "reasoning",
        "src": "am_qwen36",
    }

    record = convert_hf_row(row, adapter="noesis_text_sft")

    assert record.prompt == "한국어는 어떻게 발전했나요?"
    assert record.answer == "한국어는 여러 시대의 변화 속에서 발전했습니다."
    assert "언어 변화" in record.trace_summary
    assert record.teacher_model.startswith("hf:AMAImedia/NOESIS-50K")


def test_noesis_text_sft_converter_rejects_prompt_only_rows() -> None:
    row = {"text": "User: Build a document OCR pipeline."}

    with pytest.raises(ValueError, match="assistant"):
        convert_hf_row(row, adapter="noesis_text_sft")


def test_noesis_text_sft_converter_rejects_unclosed_think_rows() -> None:
    row = {"text": "User: solve\nAssistant: <think>partial hidden reasoning"}

    with pytest.raises(ValueError, match="unclosed"):
        convert_hf_row(row, adapter="noesis_text_sft")


def test_ragognize_converter_builds_memory_docs_and_targets() -> None:
    row = {
        "user_prompt": "Which city is the capital of France?",
        "answerable": True,
        "documents": [
            {"title": "France", "text": "Paris is the capital of France."},
            {"title": "Lyon", "text": "Lyon is a city in France."},
        ],
        "responses": {
            "model-a": {
                "details": {
                    "annotations": {
                        "result": {
                            "all_valid": True,
                            "hallucinations": [],
                        }
                    }
                },
                "text": "Paris is the capital of France.",
            }
        },
    }

    record = convert_hf_row(row, adapter="ragognize")

    assert record.prompt == "Which city is the capital of France?"
    assert record.answer == "Paris is the capital of France."
    assert [doc.doc_id for doc in record.memory_docs] == [1, 2]
    assert record.target_doc_ids == [1]
    assert record.evidence_spans == ["Paris is the capital of France."]


def test_halluclaim_converter_supports_common_document_claim_label_schema() -> None:
    row = {
        "doc": "Paris is the capital of France.",
        "claim": "Paris is the capital of France.",
        "type": "SUPPORTED",
    }

    record = convert_hf_row(row, adapter="halluclaim_76k")

    assert record.prompt.startswith("Determine whether the claim is supported")
    assert record.answer == "SUPPORTED"
    assert record.memory_docs[0].doc_id == 1
    assert record.target_doc_ids == [1]
    assert record.evidence_spans == ["Paris is the capital of France."]


def test_yana_converter_drops_identical_rejected_answer() -> None:
    row = {
        "prompt": "<question>Tell me a joke.</question>",
        "chosen": "<Thinking>Use a pun.</Thinking><Answer>Blueberry joke</Answer>",
        "rejected": "<Thinking>Use a pun.</Thinking><Answer>Blueberry joke</Answer>",
    }

    record = convert_hf_row(row, adapter="yana_reasoning_dpo")

    assert record.answer == "Blueberry joke"
    assert record.rejected_answer == ""


def test_unknown_adapter_is_rejected() -> None:
    with pytest.raises(ValueError, match="adapter"):
        convert_hf_row({}, adapter="missing")
