from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script():
    path = Path("scripts/138_score_qwen_scope_repeat_candidates.py")
    spec = importlib.util.spec_from_file_location("qwen_scope_score_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_qwen_scope_score_script_parses_candidate_specs() -> None:
    module = _load_script()

    candidates = module.parse_candidate_specs(["12:847", "23:29838,31860"])

    assert candidates == {12: {847}, 23: {29838, 31860}}


def test_qwen_scope_score_script_writes_scores_with_generation_metrics(tmp_path: Path) -> None:
    module = _load_script()
    probe_records = [
        {
            "prompt_index": 0,
            "prompt": "normal",
            "layer": 12,
            "feature_ids": [847, 1],
            "feature_values": [0.2, 0.1],
        },
        {
            "prompt_index": 1,
            "prompt": "repeat",
            "layer": 23,
            "feature_ids": [29838, 31860],
            "feature_values": [1.5, 1.0],
        },
    ]
    metrics_records = [
        {
            "sample": 0,
            "greedy_repetition": {
                "repeated_2gram_rate": 0.02,
                "repeated_3gram_rate": 0.0,
            },
        },
        {
            "sample": 1,
            "greedy_repetition": {
                "repeated_2gram_rate": 0.7,
                "repeated_3gram_rate": 0.6,
            },
        },
    ]
    probe_path = tmp_path / "probe.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    out_path = tmp_path / "scores.json"
    probe_path.write_text(
        "\n".join(json.dumps(row) for row in probe_records) + "\n",
        encoding="utf-8",
    )
    metrics_path.write_text(
        "\n".join(json.dumps(row) for row in metrics_records) + "\n",
        encoding="utf-8",
    )

    module.main(
        [
            "--input",
            str(probe_path),
            "--metrics-jsonl",
            str(metrics_path),
            "--candidate",
            "12:847",
            "--candidate",
            "23:29838,31860",
            "--out",
            str(out_path),
        ]
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["candidate_features_by_layer"] == {"12": [847], "23": [29838, 31860]}
    assert payload["scores"][0]["repeated_2gram_rate"] == 0.02
    assert payload["scores"][1]["total_value_sum"] == 2.5
    assert payload["scores"][1]["repeat_label"] == "repeat"
    assert payload["ranking_by_total_value_sum"][0]["prompt_index"] == 1
