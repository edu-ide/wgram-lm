import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path("scripts/145_calibrate_generation_verifier_eval.py")
    spec = importlib.util.spec_from_file_location("generation_verifier_calibration_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def eval_summary(records: list[dict]) -> dict:
    return {"records": records}


def test_calibrate_thresholds_uses_calibration_best_thresholds() -> None:
    module = load_module()
    calibration = eval_summary(
        [
            {"repeat_prob": 0.8, "repeat_target": 1, "stop_prob": 0.7, "stop_target": 1, "quality_prob": 0.9, "quality_target": 1},
            {"repeat_prob": 0.7, "repeat_target": 0, "stop_prob": 0.6, "stop_target": 0, "quality_prob": 0.8, "quality_target": 0},
            {"repeat_prob": 0.4, "repeat_target": 1, "stop_prob": 0.3, "stop_target": 1, "quality_prob": 0.4, "quality_target": 1},
            {"repeat_prob": 0.2, "repeat_target": 0, "stop_prob": 0.2, "stop_target": 0, "quality_prob": 0.2, "quality_target": 0},
        ]
    )

    thresholds = module.calibrate_thresholds(calibration)

    assert thresholds["repeat"]["threshold"] == 0.4
    assert thresholds["stop"]["threshold"] == 0.3
    assert thresholds["quality"]["threshold"] == 0.4


def test_evaluate_holdout_with_calibrated_thresholds() -> None:
    module = load_module()
    thresholds = {
        "repeat": {"threshold": 0.4},
        "stop": {"threshold": 0.3},
        "quality": {"threshold": 0.4},
    }
    holdout = eval_summary(
        [
            {"repeat_prob": 0.9, "repeat_target": 1, "stop_prob": 0.2, "stop_target": 0, "quality_prob": 0.1, "quality_target": 0},
            {"repeat_prob": 0.1, "repeat_target": 0, "stop_prob": 0.8, "stop_target": 1, "quality_prob": 0.8, "quality_target": 1},
        ]
    )

    metrics = module.evaluate_with_thresholds(holdout, thresholds)

    assert metrics["repeat"]["f1"] == 1.0
    assert metrics["stop"]["f1"] == 1.0
    assert metrics["quality"]["f1"] == 1.0


def test_calibration_script_writes_report(tmp_path: Path) -> None:
    module = load_module()
    calibration_path = tmp_path / "calibration.json"
    holdout_path = tmp_path / "holdout.json"
    out_path = tmp_path / "report.json"
    calibration_path.write_text(
        json.dumps(
            eval_summary(
                [
                    {"repeat_prob": 0.8, "repeat_target": 1, "stop_prob": 0.8, "stop_target": 1, "quality_prob": 0.9, "quality_target": 1},
                    {"repeat_prob": 0.2, "repeat_target": 0, "stop_prob": 0.2, "stop_target": 0, "quality_prob": 0.1, "quality_target": 0},
                ]
            )
        ),
        encoding="utf-8",
    )
    holdout_path.write_text(
        json.dumps(
            eval_summary(
                [
                    {"repeat_prob": 0.9, "repeat_target": 1, "stop_prob": 0.9, "stop_target": 1, "quality_prob": 0.8, "quality_target": 1},
                    {"repeat_prob": 0.1, "repeat_target": 0, "stop_prob": 0.1, "stop_target": 0, "quality_prob": 0.2, "quality_target": 0},
                ]
            )
        ),
        encoding="utf-8",
    )

    module.main(
        [
            "--calibration-eval-json",
            str(calibration_path),
            "--holdout-eval-json",
            str(holdout_path),
            "--out",
            str(out_path),
        ]
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["calibration"]["n"] == 2
    assert report["holdout"]["repeat"]["f1"] == 1.0
