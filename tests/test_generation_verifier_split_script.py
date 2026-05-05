import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path("scripts/144_split_generation_verifier_dataset.py")
    spec = importlib.util.spec_from_file_location("generation_verifier_split_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def make_rows() -> list[dict]:
    signatures = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    rows = []
    for sig_idx, (repeat, stop, quality) in enumerate(signatures):
        for offset in range(4):
            sample = sig_idx * 10 + offset
            rows.append(
                {
                    "text": f"sample {sample}",
                    "source_sample": sample,
                    "category": "demo",
                    "generation_verifier_repeat_target": repeat,
                    "generation_verifier_stop_target": stop,
                    "generation_verifier_quality_target": quality,
                }
            )
    return rows


def test_split_rows_is_deterministic_and_preserves_all_rows() -> None:
    module = load_module()
    rows = make_rows()

    first = module.split_rows(rows, calibration_ratio=0.25, holdout_ratio=0.25, seed=13)
    second = module.split_rows(rows, calibration_ratio=0.25, holdout_ratio=0.25, seed=13)

    assert first == second
    assert {key: len(value) for key, value in first.items()} == {
        "train": 8,
        "calibration": 4,
        "holdout": 4,
    }
    all_samples = [
        row["source_sample"]
        for split_rows in first.values()
        for row in split_rows
    ]
    assert sorted(all_samples) == sorted(row["source_sample"] for row in rows)
    assert len(all_samples) == len(set(all_samples))


def test_split_summary_counts_targets_per_split() -> None:
    module = load_module()
    splits = module.split_rows(make_rows(), calibration_ratio=0.25, holdout_ratio=0.25, seed=5)

    summary = module.summarize_splits(splits)

    assert summary["total_rows"] == 16
    assert summary["splits"]["train"]["rows"] == 8
    assert summary["splits"]["calibration"]["repeat_failures"] == 2
    assert summary["splits"]["holdout"]["stop_failures"] == 2
    assert summary["target_totals"]["quality_pass"] == 4


def test_split_script_writes_three_jsonl_files_and_summary(tmp_path: Path) -> None:
    module = load_module()
    data_path = tmp_path / "verifier.jsonl"
    out_dir = tmp_path / "splits"
    with data_path.open("w", encoding="utf-8") as f:
        for row in make_rows():
            f.write(json.dumps(row) + "\n")

    module.main(
        [
            "--data-jsonl",
            str(data_path),
            "--out-dir",
            str(out_dir),
            "--prefix",
            "demo",
            "--calibration-ratio",
            "0.25",
            "--holdout-ratio",
            "0.25",
            "--seed",
            "7",
        ]
    )

    train_rows = [
        json.loads(line)
        for line in (out_dir / "demo_train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    calibration_rows = [
        json.loads(line)
        for line in (out_dir / "demo_calibration.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    holdout_rows = [
        json.loads(line)
        for line in (out_dir / "demo_holdout.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads((out_dir / "demo_split_summary.json").read_text(encoding="utf-8"))

    assert len(train_rows) == 8
    assert len(calibration_rows) == 4
    assert len(holdout_rows) == 4
    assert summary["prefix"] == "demo"
