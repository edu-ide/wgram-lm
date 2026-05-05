from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script():
    path = Path("scripts/137_compare_qwen_scope_groups.py")
    spec = importlib.util.spec_from_file_location("qwen_scope_compare_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_qwen_scope_compare_script_parses_group_indices() -> None:
    module = _load_script()

    args = module.build_arg_parser().parse_args(
        [
            "--input",
            "in.jsonl",
            "--out",
            "out.json",
            "--normal",
            "0,1,2",
            "--repeat",
            "3,4",
        ]
    )

    assert module.parse_indices(args.normal) == {0, 1, 2}
    assert module.parse_indices(args.repeat) == {3, 4}


def test_qwen_scope_compare_script_writes_summary(tmp_path: Path) -> None:
    module = _load_script()
    records = [
        {
            "prompt_index": 0,
            "layer": 0,
            "token": ".",
            "feature_ids": [1, 2],
            "feature_values": [1.0, 0.5],
        },
        {
            "prompt_index": 1,
            "layer": 0,
            "token": " Freeze",
            "feature_ids": [9, 2],
            "feature_values": [2.0, 0.7],
        },
    ]
    input_path = tmp_path / "probe.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(row) for row in records) + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "summary.json"

    module.main(
        [
            "--input",
            str(input_path),
            "--out",
            str(out_path),
            "--normal",
            "0",
            "--repeat",
            "1",
        ]
    )

    summary = json.loads(out_path.read_text(encoding="utf-8"))
    assert summary["input"] == str(input_path)
    assert summary["normal_prompt_indices"] == [0]
    assert summary["repeat_prompt_indices"] == [1]
    assert summary["layers"]["0"]["repeat_enriched_features"][0]["feature_id"] == 9
