import json
from pathlib import Path


def test_resolve_history_path_uses_daily_generation_file(tmp_path):
    from wgram_lm.history import resolve_history_path

    path = resolve_history_path(
        "auto",
        root=tmp_path,
        kind="generations",
        date_text="2026-05-01",
    )

    assert path == tmp_path / "history" / "generations" / "2026-05-01.jsonl"


def test_append_generation_history_writes_jsonl_row(tmp_path):
    from wgram_lm.history import append_generation_history

    out = tmp_path / "history.jsonl"
    row = append_generation_history(
        out,
        source="infer_with_donor",
        checkpoint="runs/demo/last.pt",
        config="configs/demo.yaml",
        prompt="질문",
        output="질문 답변",
        mode="qtrm_residual",
        completion="답변",
        metadata={"max_new_tokens": 16},
        timestamp="2026-05-01T12:00:00+09:00",
    )

    written = json.loads(out.read_text(encoding="utf-8").strip())
    assert written == row
    assert written["source"] == "infer_with_donor"
    assert written["checkpoint"] == "runs/demo/last.pt"
    assert written["prompt"] == "질문"
    assert written["output"] == "질문 답변"
    assert written["completion"] == "답변"
    assert written["metadata"]["max_new_tokens"] == 16


def test_eval_record_to_history_row_preserves_failure_fields():
    from wgram_lm.history import eval_record_to_history_row

    row = eval_record_to_history_row(
        {
            "id": "case-1",
            "mode": "qtrm_residual_with_evidence",
            "question": "코드는?",
            "completion": "Answer: UNKNOWN",
            "raw_completion": "Answer: UNKNOWN",
            "hit": False,
            "answer_channel": "evidence_span_copy",
            "answer_channel_meta": {"status": "no_answer"},
            "retrieved_sources": ["doc.md"],
            "retrieved_roles": ["target"],
            "completion_token_count": 2,
        },
        checkpoint="runs/demo/last.pt",
        config="configs/demo.yaml",
        source="memory_eval",
        timestamp="2026-05-01T12:00:00+09:00",
    )

    assert row["source"] == "memory_eval"
    assert row["case_id"] == "case-1"
    assert row["prompt"] == "코드는?"
    assert row["completion"] == "Answer: UNKNOWN"
    assert row["hit"] is False
    assert row["failure_type"] == "miss"
    assert row["answer_channel_meta"] == {"status": "no_answer"}
    assert row["retrieved_evidence"][0]["source"] == "doc.md"
