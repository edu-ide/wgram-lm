from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script():
    path = Path("scripts/139_build_qwen_scope_repeat_gate_prompts.py")
    spec = importlib.util.spec_from_file_location("qwen_scope_repeat_gate_prompts", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_repeat_gate_prompt_suite_has_balanced_categories() -> None:
    module = _load_script()

    prompts = module.build_prompt_suite()

    assert len(prompts) >= 50
    categories = {row["category"] for row in prompts}
    assert {"normal_qa", "math_reasoning", "evidence_check", "korean_qa", "repeat_stress"} <= categories
    assert all(row["text"].strip() for row in prompts)


def test_repeat_gate_prompt_script_writes_jsonl(tmp_path: Path) -> None:
    module = _load_script()
    out_path = tmp_path / "prompts.jsonl"

    module.main(["--out", str(out_path), "--limit", "7"])

    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 7
    assert rows[0]["prompt_id"] == 0
    assert rows[0]["category"]
    assert rows[0]["text"]
