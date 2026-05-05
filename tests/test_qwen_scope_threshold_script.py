from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script():
    path = Path("scripts/140_summarize_qwen_scope_repeat_scores.py")
    spec = importlib.util.spec_from_file_location("qwen_scope_threshold_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_qwen_scope_threshold_script_writes_summary(tmp_path: Path) -> None:
    module = _load_script()
    input_path = tmp_path / "scores.json"
    out_path = tmp_path / "summary.json"
    payload = {
        "scores": [
            {"prompt_index": 0, "total_value_sum": 0.4, "repeat_label": "normal"},
            {"prompt_index": 1, "total_value_sum": 2.5, "repeat_label": "repeat"},
            {"prompt_index": 2, "total_value_sum": 3.0, "repeat_label": "repeat"},
        ]
    }
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    module.main(["--input", str(input_path), "--out", str(out_path)])

    summary = json.loads(out_path.read_text(encoding="utf-8"))
    assert summary["input"] == str(input_path)
    assert summary["threshold_summary"]["best_threshold"]["threshold"] == 2.5
    assert summary["threshold_summary"]["best_threshold"]["f1"] == 1.0
