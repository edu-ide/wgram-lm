from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_script():
    path = Path("scripts/160_calibrate_answer_decision_gate.py")
    spec = importlib.util.spec_from_file_location("answer_decision_gate", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(
    *,
    record_id: str,
    completion: str,
    aliases: list[str],
    expected_unknown: bool,
    support: float,
    causal: float,
    refute: float,
    missing: float,
) -> dict:
    return {
        "id": record_id,
        "mode": "qtrm_residual_with_evidence",
        "question": f"Question {record_id}",
        "completion": completion,
        "answer_aliases": aliases,
        "expected_unknown": expected_unknown,
        "answer_channel_meta": {
            "truth_gate": {
                "support_prob": support,
                "causal_prob": causal,
                "refute_prob": refute,
                "missing_prob": missing,
            }
        },
    }


def test_gate_blocks_high_missing_unknown_candidate() -> None:
    module = _load_script()
    row = _record(
        record_id="r1",
        completion="Answer: fake-code",
        aliases=["UNKNOWN"],
        expected_unknown=True,
        support=0.7,
        causal=0.7,
        refute=0.1,
        missing=0.6,
    )
    thresholds = module.Thresholds(
        support_min=0.5,
        causal_min=0.5,
        refute_max=0.5,
        missing_max=0.5,
    )

    assert module.gate_allows(row, thresholds) is False
    metrics = module.evaluate_records([row], thresholds=thresholds)
    assert metrics["hits"] == 1
    assert metrics["blocked"] == 1


def test_find_best_thresholds_improves_calibration_records() -> None:
    module = _load_script()
    records = [
        _record(
            record_id="unknown",
            completion="Answer: fake-code",
            aliases=["UNKNOWN"],
            expected_unknown=True,
            support=0.7,
            causal=0.7,
            refute=0.1,
            missing=0.6,
        ),
        _record(
            record_id="positive",
            completion="Answer: VX-9",
            aliases=["VX-9"],
            expected_unknown=False,
            support=0.8,
            causal=0.8,
            refute=0.1,
            missing=0.2,
        ),
    ]

    thresholds, metrics = module.find_best_thresholds(records)

    assert thresholds.missing_max <= 0.6
    assert metrics["accuracy"] == 1.0
    assert metrics["false_positive"] == 0


def test_main_writes_report(tmp_path: Path) -> None:
    module = _load_script()
    records = [
        _record(
            record_id="a",
            completion="Answer: fake-code",
            aliases=["UNKNOWN"],
            expected_unknown=True,
            support=0.7,
            causal=0.7,
            refute=0.1,
            missing=0.6,
        ),
        _record(
            record_id="b",
            completion="Answer: VX-9",
            aliases=["VX-9"],
            expected_unknown=False,
            support=0.8,
            causal=0.8,
            refute=0.1,
            missing=0.2,
        ),
        _record(
            record_id="c",
            completion="Answer: fake-code",
            aliases=["UNKNOWN"],
            expected_unknown=True,
            support=0.7,
            causal=0.7,
            refute=0.1,
            missing=0.6,
        ),
        _record(
            record_id="d",
            completion="Answer: VX-8",
            aliases=["VX-8"],
            expected_unknown=False,
            support=0.8,
            causal=0.8,
            refute=0.1,
            missing=0.2,
        ),
    ]
    records_path = tmp_path / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"

    module.main(
        [
            "--records-jsonl",
            str(records_path),
            "--out-json",
            str(out_json),
            "--markdown-out",
            str(out_md),
            "--calibration-fraction",
            "0.5",
        ]
    )

    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["count"] == 4
    assert "best_thresholds" in report
    assert out_md.exists()
