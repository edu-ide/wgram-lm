from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script():
    path = Path("scripts/155_build_controller_trace_replay.py")
    spec = importlib.util.spec_from_file_location("controller_trace_replay_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_controller_trace_replay_builder_emits_retrieve_verify_answer_rows(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source.jsonl"
    out = tmp_path / "trace.jsonl"
    prompt = (
        "MemoryOS evidence\n"
        "SOURCE=archive.md CHUNK=0 SCORE=1.0000\n"
        "The access code is VX-913.\n\n"
        "Use the evidence above when it is relevant. If it is not relevant, answer from the prompt.\n\n"
        "User prompt:\n"
        "Answer using only the evidence.\n"
        "Question: What is the access code?"
    )
    source.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "prompt": prompt,
                "answer": "Answer: VX-913",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    count = module.write_controller_trace_replay(source, out)

    rows = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert count == 3
    assert [row["action_target"] for row in rows] == [
        "RETRIEVE_MEMORY",
        "VERIFY_EVIDENCE",
        "ANSWER",
    ]
    assert rows[0]["state_summary"] != rows[1]["state_summary"]
    assert rows[1]["previous_observation"]
    assert rows[2]["previous_observation"]
    assert all(row["type"] == "trace_replay" for row in rows)
    assert all(row["action_sample_weight"] == 1.0 for row in rows)
    assert "VX-913" in rows[0]["workspace_context"]
    assert "VX-913" not in rows[0]["chat_prompt"]


def test_controller_trace_replay_builder_accepts_eval_memory_case_rows(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source.jsonl"
    out = tmp_path / "trace.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "heldout-1",
                "instruction": "Use signed records over anonymous notes.",
                "question": "What is the Garnet override phrase?",
                "answer_aliases": ["UNKNOWN"],
                "evidence": [
                    {
                        "source": "signed.md",
                        "chunk_id": 0,
                        "text": "Signed notice: the Garnet override phrase is redacted.",
                    }
                ],
                "distractors": [
                    {
                        "source": "anonymous.md",
                        "chunk_id": 1,
                        "text": "Anonymous note: the phrase is stone-arch.",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    count = module.write_controller_trace_replay(source, out)

    rows = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert count == 3
    assert rows[0]["task_id"] == "heldout-1"
    assert rows[0]["action_target"] == "RETRIEVE_MEMORY"
    assert rows[1]["action_target"] == "VERIFY_EVIDENCE"
    assert rows[2]["action_target"] == "ANSWER"
    assert "Use signed records" in rows[0]["chat_prompt"]
    assert "signed.md" in rows[0]["workspace_context"]
    assert "anonymous.md" in rows[0]["workspace_context"]
    assert rows[2]["observation"] == "UNKNOWN"


def test_controller_trace_replay_builder_can_emit_signal_conditioned_rows(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source.jsonl"
    out = tmp_path / "trace.jsonl"
    prompt = (
        "MemoryOS evidence\n"
        "SOURCE=archive.md CHUNK=0 SCORE=1.0000\n"
        "The access code is VX-913.\n\n"
        "Use the evidence above when it is relevant. If it is not relevant, answer from the prompt.\n\n"
        "User prompt:\n"
        "Question: What is the access code?"
    )
    source.write_text(
        json.dumps(
            {
                "case_id": "case-signal",
                "prompt": prompt,
                "answer": "VX-913",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    count = module.write_controller_trace_replay(
        source,
        out,
        signal_conditioned=True,
    )

    rows = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert count == 3
    assert [row["controller_signal"] for row in rows] == [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    assert all(row["hide_trace_step_from_input"] for row in rows)
    assert len({row["state_summary"] for row in rows}) == 1
    assert rows[1]["previous_observation"] == ""
