from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_hf_distill_convert_script_converts_local_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "sample.jsonl"
    out = tmp_path / "out.jsonl"
    source.write_text(
        json.dumps(
            {
                "prompt": "<question>What is 2+2?</question>",
                "chosen": "<Thinking>Add two and two.</Thinking><Answer>\\boxed{4}</Answer>",
                "rejected": "<Answer>\\boxed{5}</Answer>",
            }
        )
        + "\n"
    )

    result = subprocess.run(
        [
            "python3",
            "scripts/131_convert_hf_distill_dataset.py",
            "--adapter",
            "yana_reasoning_dpo",
            "--local-jsonl",
            str(source),
            "--out",
            str(out),
            "--max-rows",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    row = json.loads(out.read_text().strip())
    assert row["prompt"] == "What is 2+2?"
    assert row["answer"] == "4"
    assert row["rejected_answer"] == "5"
    assert "converted=1" in result.stdout
