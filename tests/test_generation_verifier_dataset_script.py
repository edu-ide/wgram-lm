from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script():
    path = Path("scripts/141_build_generation_verifier_dataset.py")
    spec = importlib.util.spec_from_file_location("generation_verifier_dataset_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generation_verifier_targets_label_repeat_and_stop_failures() -> None:
    module = _load_script()
    row = {
        "sample": 4,
        "text": "Give a direct answer only: 5 + 7 = ?",
        "greedy_text": "Give a direct answer only: 5 + 7 = ?\n12. Give a direct answer only:",
        "greedy_repetition": {
            "completion_tokens": 64,
            "repeated_2gram_rate": 0.30,
            "repeated_3gram_rate": 0.12,
        },
    }

    targets = module.generation_verifier_targets(
        row,
        category="repeat_stress",
        max_new_tokens=64,
        repeat_threshold=0.15,
        severe_repeat_threshold=0.25,
    )

    assert targets["generation_verifier_repeat_target"] == 1.0
    assert targets["generation_verifier_stop_target"] == 1.0
    assert targets["generation_verifier_quality_target"] == 0.0
    assert targets["generation_verifier_sample_weight"] > 1.0
    assert targets["format_failure"] is False


def test_generation_verifier_targets_treat_visible_reasoning_as_quality_failure() -> None:
    module = _load_script()
    row = {
        "sample": 0,
        "text": "Explain quantum entanglement.",
        "greedy_text": "Explain quantum entanglement.\n\n<think>\nLet me reason this out.",
        "greedy_repetition": {
            "completion_tokens": 12,
            "repeated_2gram_rate": 0.0,
            "repeated_3gram_rate": 0.0,
        },
    }

    targets = module.generation_verifier_targets(row)

    assert targets["format_failure"] is True
    assert targets["generation_verifier_repeat_target"] == 0.0
    assert targets["generation_verifier_stop_target"] == 0.0
    assert targets["generation_verifier_quality_target"] == 0.0


def test_generation_verifier_targets_ignore_prompt_contract_when_checking_format() -> None:
    module = _load_script()
    prompt = "Explain.\n\n/no_think\nAnswer directly. Do not reveal hidden reasoning."
    row = {
        "sample": 0,
        "text": prompt,
        "greedy_text": prompt + "\nA clean answer.",
        "greedy_repetition": {
            "completion_tokens": 4,
            "repeated_2gram_rate": 0.0,
            "repeated_3gram_rate": 0.0,
        },
    }

    targets = module.generation_verifier_targets(row)

    assert targets["format_failure"] is False
    assert targets["generation_verifier_quality_target"] == 1.0


def test_generation_verifier_targets_treat_answer_drift_as_quality_failure() -> None:
    module = _load_script()
    row = {
        "sample": 2,
        "text": "Why do seasons happen on Earth?",
        "greedy_text": "Why do seasons happen on Earth?\nA. The sun moves closer.\nB. The moon moves.",
        "greedy_repetition": {
            "completion_tokens": 18,
            "repeated_2gram_rate": 0.0,
            "repeated_3gram_rate": 0.0,
        },
    }

    targets = module.generation_verifier_targets(row)

    assert targets["answer_drift_failure"] is True
    assert targets["generation_verifier_quality_target"] == 0.0


def test_generation_verifier_dataset_script_writes_rows(tmp_path: Path) -> None:
    module = _load_script()
    eval_path = tmp_path / "eval.jsonl"
    meta_path = tmp_path / "prompts.jsonl"
    out_path = tmp_path / "verifier.jsonl"
    eval_rows = [
        {
            "sample": 0,
            "text": "What is photosynthesis?",
            "greedy_text": "What is photosynthesis?\nPlants use light to make food.",
            "greedy_repetition": {
                "completion_tokens": 12,
                "repeated_2gram_rate": 0.0,
                "repeated_3gram_rate": 0.0,
            },
        },
        {
            "sample": 1,
            "text": "Answer with one word: Is ice frozen water?",
            "greedy_text": "Answer with one word: Is ice frozen water?\nYes. Answer with one word:",
            "greedy_repetition": {
                "completion_tokens": 64,
                "repeated_2gram_rate": 0.3,
                "repeated_3gram_rate": 0.1,
            },
        },
    ]
    meta_rows = [
        {"prompt_id": 0, "category": "normal_qa", "text": "What is photosynthesis?"},
        {"prompt_id": 1, "category": "repeat_stress", "text": "Answer with one word: Is ice frozen water?"},
    ]
    eval_path.write_text(
        "\n".join(json.dumps(row) for row in eval_rows) + "\n",
        encoding="utf-8",
    )
    meta_path.write_text(
        "\n".join(json.dumps(row) for row in meta_rows) + "\n",
        encoding="utf-8",
    )

    module.main(
        [
            "--eval-jsonl",
            str(eval_path),
            "--prompt-meta-jsonl",
            str(meta_path),
            "--out",
            str(out_path),
        ]
    )

    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert rows[0]["generation_verifier_quality_target"] == 1.0
    assert rows[1]["generation_verifier_repeat_target"] == 1.0
    assert rows[1]["generation_verifier_stop_target"] == 1.0
    assert rows[1]["generation_verifier_quality_target"] == 0.0
    assert rows[1]["category"] == "repeat_stress"
    assert rows[1]["format_failure"] is False
