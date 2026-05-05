from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path("scripts/156_eval_controller_trace_policy.py")
    spec = importlib.util.spec_from_file_location("controller_trace_policy_eval_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_summarize_action_predictions_reports_confusion_by_action_name() -> None:
    module = _load_script()

    summary = module.summarize_action_predictions(
        preds=[1, 3, 3, 3],
        targets=[1, 2, 3, 3],
    )

    assert summary["samples"] == 4
    assert summary["accuracy"] == 0.75
    assert summary["per_target"]["RETRIEVE_MEMORY"]["accuracy"] == 1.0
    assert summary["per_target"]["VERIFY_EVIDENCE"]["accuracy"] == 0.0
    assert summary["confusion"]["VERIFY_EVIDENCE"]["ANSWER"] == 1


def test_jsonl_line_count_counts_nonempty_lines(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")

    assert module._jsonl_line_count(path) == 2
