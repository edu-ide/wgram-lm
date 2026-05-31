import json
from pathlib import Path

from wgram_lm.distill.teacher_schema import Qwen36TeacherRecord, TeacherEvidenceDoc


def test_teacher_record_with_evidence_becomes_workspace_preference_row() -> None:
    from wgram_lm.distill.training_mix import teacher_record_to_training_row

    record = Qwen36TeacherRecord(
        prompt="What is the access code?",
        answer="VX-913",
        rejected_answer="UNKNOWN",
        teacher_model="unit-teacher",
        evidence_spans=["The access code is VX-913."],
        memory_docs=[
            TeacherEvidenceDoc(doc_id=1, source="wrong.md", text="The access code is QL-404."),
            TeacherEvidenceDoc(doc_id=2, source="right.md", text="The access code is VX-913."),
        ],
        target_doc_ids=[2],
    )

    row = teacher_record_to_training_row(record, source_name="ragognize_evidence")

    assert row["prompt"].startswith("MemoryOS evidence")
    assert "SOURCE=right.md CHUNK=2 SCORE=1.0000" in row["prompt"]
    assert "User prompt:\nWhat is the access code?" in row["prompt"]
    assert row["chosen"] == "VX-913"
    assert row["rejected"] == "UNKNOWN"
    assert row["preference_weight"] == 1.0
    assert row["logical_support_target"] == 1.0
    assert row["logical_refute_target"] == 0.0
    assert row["causal_evidence_target"] == 1.0
    assert row["target_doc_ids"] == [2]
    assert row["teacher_model"] == "unit-teacher"
    assert row["distill_source"] == "ragognize_evidence"


def test_refuted_teacher_record_sets_refute_target() -> None:
    from wgram_lm.distill.training_mix import teacher_record_to_training_row

    record = Qwen36TeacherRecord(
        prompt="Judge the claim.",
        answer="REFUTED",
        rejected_answer="SUPPORTED",
        evidence_spans=["The claim contradicts the document."],
        memory_docs=[
            TeacherEvidenceDoc(
                doc_id=1,
                source="claim.md",
                text="The claim contradicts the document.",
            )
        ],
        target_doc_ids=[1],
    )

    row = teacher_record_to_training_row(record, source_name="halluclaim_76k")

    assert row["chosen"] == "REFUTED"
    assert row["rejected"] == "SUPPORTED"
    assert row["logical_support_target"] == 0.0
    assert row["logical_refute_target"] == 1.0
    assert row["logical_missing_target"] == 0.0
    assert row["causal_evidence_target"] == 1.0


def test_unsupported_evidence_record_becomes_needs_search_preference() -> None:
    from wgram_lm.distill.training_mix import teacher_record_to_training_row

    record = Qwen36TeacherRecord(
        prompt="Who won the hidden tournament?",
        answer="A confident but unsupported answer.",
        memory_docs=[
            TeacherEvidenceDoc(doc_id=1, source="retrieved.md", text="This document does not say who won.")
        ],
    )

    row = teacher_record_to_training_row(record, source_name="ragognize_evidence")

    assert row["chosen"] == "NEEDS_SEARCH"
    assert row["rejected"] == "A confident but unsupported answer."
    assert row["unsupported_answer"] == "A confident but unsupported answer."
    assert row["logical_support_target"] == 0.0
    assert row["logical_refute_target"] == 0.0
    assert row["logical_missing_target"] == 1.0
    assert row["causal_evidence_target"] == 0.0


def test_build_training_mix_caps_each_source(tmp_path: Path) -> None:
    from wgram_lm.distill.training_mix import build_training_mix

    first = tmp_path / "yana.jsonl"
    first.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "p1", "answer": "a1", "rejected_answer": "r1"}),
                json.dumps({"prompt": "p2", "answer": "a2", "rejected_answer": "r2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    second = tmp_path / "noesis.jsonl"
    second.write_text(
        json.dumps({"prompt": "p3", "answer": "a3", "trace_summary": "reasoned"})
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "mix.jsonl"

    stats = build_training_mix(
        inputs=[first, second],
        out_path=out,
        max_rows_per_source=1,
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert stats["written"] == 2
    assert stats["sources"] == {"yana": 1, "noesis": 1}
    assert [row["distill_source"] for row in rows] == ["yana", "noesis"]
    assert rows[0]["chosen"] == "a1"
    assert rows[0]["rejected"] == "r1"
    assert rows[1]["answer"] == "a3"
    assert rows[1]["trace_summary"] == "reasoned"


def test_hf_first_wave_warmup_keeps_general_qtrm_residual_active() -> None:
    from wgram_lm.config import load_config

    cfg = load_config("configs/qwen35_2b_4090_hf_first_wave_warmup_s400.yaml")

    assert cfg.model.evidence_bottleneck_enabled is True
    assert cfg.model.evidence_bottleneck_applies_to_residual is False


def test_openmythos_style_warmup_v2_keeps_core_on_the_text_causal_path() -> None:
    from wgram_lm.config import load_config

    cfg = load_config("configs/qwen35_2b_4090_hf_first_wave_warmup_v2_s400.yaml")

    assert cfg.model.core_to_text_enabled is True
    assert cfg.model.core_to_text_gate_init_bias < 0.0
    assert cfg.model.coda_attn_every == 1
    assert cfg.model.evidence_bottleneck_applies_to_residual is False
    assert cfg.train.out_dir.endswith("_v2_s400")
