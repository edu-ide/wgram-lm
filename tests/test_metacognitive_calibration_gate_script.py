import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def load_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "202_build_metacognitive_calibration_gate.py"
    spec = importlib.util.spec_from_file_location("metacognitive_calibration_gate_script", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MetacognitiveCalibrationGateScriptTests(unittest.TestCase):
    def test_choice_confidence_uses_softmax_over_choice_logprobs(self) -> None:
        module = load_script()
        record = {
            "hit": True,
            "completion": "A",
            "answer_aliases": ["A"],
            "choice_scores": [
                {"choice": "A", "logprob": 2.0},
                {"choice": "B", "logprob": 0.0},
            ],
        }

        scored = module.record_calibration(record)

        self.assertTrue(scored["available"])
        self.assertEqual(scored["predicted_choice"], "A")
        self.assertGreater(scored["confidence"], 0.85)
        self.assertEqual(scored["correct"], 1.0)
        self.assertLess(scored["brier"], 0.03)

    def test_choice_calibration_preserves_conflict_gate_telemetry(self) -> None:
        module = load_script()
        record = {
            "hit": True,
            "completion": "A",
            "answer_aliases": ["A"],
            "choice_scores": [
                {"choice": "A", "logprob": 2.0, "donor_qtrm_conflict_gate_mean": 0.25},
                {"choice": "B", "logprob": 0.0, "donor_qtrm_conflict_gate_mean": 0.75},
            ],
        }

        scored = module.record_calibration(record)
        summary = module.calibration_summary([{**record, "calibration": scored}], n_bins=5)

        self.assertEqual(scored["predicted_conflict_gate_mean"], 0.25)
        self.assertEqual(scored["mean_choice_conflict_gate_mean"], 0.5)
        self.assertEqual(summary["mean_predicted_conflict_gate"], 0.25)
        self.assertEqual(summary["mean_choice_conflict_gate"], 0.5)

    def test_calibration_summary_computes_ece_and_brier(self) -> None:
        module = load_script()
        rows = [
            {"mode": "m", "hit": True, "calibration": {"available": True, "confidence": 0.9, "correct": 1.0, "brier": 0.01}},
            {"mode": "m", "hit": False, "calibration": {"available": True, "confidence": 0.8, "correct": 0.0, "brier": 0.64}},
            {"mode": "m", "hit": True, "calibration": {"available": True, "confidence": 0.2, "correct": 1.0, "brier": 0.64}},
        ]

        summary = module.calibration_summary(rows, n_bins=5)

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["accuracy"], 2 / 3)
        self.assertEqual(
            round(summary["mean_confidence"], 6),
            round((0.9 + 0.8 + 0.2) / 3, 6),
        )
        self.assertGreater(summary["ece"], 0.0)
        self.assertEqual(round(summary["brier"], 6), round((0.01 + 0.64 + 0.64) / 3, 6))

    def test_build_matched_gate_accepts_only_accuracy_safe_calibration_gain(self) -> None:
        module = load_script()
        baseline = [
            {"id": "a", "mode": "qtrm_core_steps_8_no_evidence", "category": "answerable", "expected_unknown": False, "hit": True, "choice_scores": [{"choice": "A", "logprob": 3.0}, {"choice": "B", "logprob": 0.0}], "completion": "A", "answer_aliases": ["A"]},
            {"id": "b", "mode": "qtrm_core_steps_8_no_evidence", "category": "unknown", "expected_unknown": True, "hit": False, "choice_scores": [{"choice": "A", "logprob": 3.0}, {"choice": "B", "logprob": 0.0}], "completion": "A", "answer_aliases": ["B"]},
        ]
        candidate = [
            {"id": "a", "mode": "qtrm_core_steps_8_no_evidence", "category": "answerable", "expected_unknown": False, "hit": True, "choice_scores": [{"choice": "A", "logprob": 1.0}, {"choice": "B", "logprob": 0.0}], "completion": "A", "answer_aliases": ["A"]},
            {"id": "b", "mode": "qtrm_core_steps_8_no_evidence", "category": "unknown", "expected_unknown": True, "hit": False, "choice_scores": [{"choice": "A", "logprob": 0.2}, {"choice": "B", "logprob": 0.0}], "completion": "A", "answer_aliases": ["B"]},
        ]

        gate = module.build_matched_metacognitive_gate(
            baseline,
            candidate,
            baseline_label="no_warmup",
            candidate_label="warmup",
            n_bins=5,
        )

        comparison = gate["mode_comparisons"]["qtrm_core_steps_8_no_evidence"]
        self.assertEqual(gate["status"], "accepted")
        self.assertEqual(comparison["accuracy_delta"], 0.0)
        self.assertLess(comparison["brier_delta"], 0.0)
        self.assertIn("candidate_ece_not_worse", gate["passed_checks"])
        self.assertIn("category", gate["field_comparisons"])
        self.assertIn("answerable", gate["field_comparisons"]["category"])
        self.assertIn("expected_unknown", gate["field_comparisons"])

    def test_build_matched_gate_rejects_when_critical_qtrm_mode_worsens(self) -> None:
        module = load_script()
        baseline = []
        candidate = []
        for idx in range(4):
            baseline.append(
                {
                    "id": f"donor-{idx}",
                    "mode": "donor_only_no_evidence",
                    "hit": bool(idx % 2),
                    "choice_scores": [
                        {"choice": "A", "logprob": 0.1},
                        {"choice": "B", "logprob": 0.0},
                    ],
                    "completion": "A",
                    "answer_aliases": ["A" if idx % 2 else "B"],
                }
            )
            candidate.append({**baseline[-1]})
        baseline.append(
            {
                "id": "qtrm-bad",
                "mode": "qtrm_core_steps_8_qtrm_only_no_evidence",
                "hit": False,
                "choice_scores": [
                    {"choice": "A", "logprob": 0.1},
                    {"choice": "B", "logprob": 0.0},
                ],
                "completion": "A",
                "answer_aliases": ["B"],
            }
        )
        candidate.append(
            {
                "id": "qtrm-bad",
                "mode": "qtrm_core_steps_8_qtrm_only_no_evidence",
                "hit": False,
                "choice_scores": [
                    {"choice": "A", "logprob": 5.0},
                    {"choice": "B", "logprob": 0.0},
                ],
                "completion": "A",
                "answer_aliases": ["B"],
            }
        )

        gate = module.build_matched_metacognitive_gate(
            baseline,
            candidate,
            baseline_label="baseline",
            candidate_label="candidate",
            n_bins=5,
        )

        self.assertEqual(gate["status"], "rejected")
        self.assertIn("critical_mode_qtrm_core_steps_8_qtrm_only_no_evidence_ece_worse", gate["failed_checks"])

    def test_qtrm_core_profile_filters_global_checks_to_core_modes(self) -> None:
        module = load_script()

        def row(row_id: str, mode: str, *, correct: float, confidence: float) -> dict:
            return {
                "id": row_id,
                "mode": mode,
                "calibration": {
                    "available": True,
                    "confidence": confidence,
                    "correct": correct,
                    "brier": (confidence - correct) ** 2,
                },
            }

        baseline = [
            row("core-full", "qtrm_core_steps_8_no_evidence", correct=1.0, confidence=0.60),
            row("core-only", "qtrm_core_steps_8_qtrm_only_no_evidence", correct=0.0, confidence=0.60),
            row("donor", "donor_only_no_evidence", correct=1.0, confidence=0.90),
        ]
        candidate = [
            row("core-full", "qtrm_core_steps_8_no_evidence", correct=1.0, confidence=1.00),
            row("core-only", "qtrm_core_steps_8_qtrm_only_no_evidence", correct=0.0, confidence=0.00),
            row("donor", "donor_only_no_evidence", correct=0.0, confidence=0.95),
        ]

        strict_gate = module.build_matched_metacognitive_gate(
            baseline,
            candidate,
            baseline_label="baseline",
            candidate_label="candidate",
            n_bins=5,
            gate_profile="strict",
        )
        qtrm_core_gate = module.build_matched_metacognitive_gate(
            baseline,
            candidate,
            baseline_label="baseline",
            candidate_label="candidate",
            n_bins=5,
            gate_profile="qtrm_core",
        )

        self.assertEqual(strict_gate["status"], "rejected")
        self.assertEqual(qtrm_core_gate["status"], "accepted")
        self.assertEqual(qtrm_core_gate["gate_profile"], "qtrm_core")
        self.assertEqual(qtrm_core_gate["profile_record_count"], 4)
        self.assertEqual(
            qtrm_core_gate["included_modes"],
            [
                "qtrm_core_steps_8_no_evidence",
                "qtrm_core_steps_8_qtrm_only_no_evidence",
            ],
        )
        self.assertNotIn("donor_only_no_evidence", qtrm_core_gate["mode_comparisons"])

    def test_fused_profile_focuses_low_donor_and_full_qtrm_modes(self) -> None:
        module = load_script()

        def row(row_id: str, mode: str, *, correct: float, confidence: float) -> dict:
            return {
                "id": row_id,
                "mode": mode,
                "calibration": {
                    "available": True,
                    "confidence": confidence,
                    "correct": correct,
                    "brier": (confidence - correct) ** 2,
                },
            }

        baseline = [
            row("full", "qtrm_core_steps_8_no_evidence", correct=1.0, confidence=0.60),
            row("low-donor", "qtrm_core_steps_8_low_donor_no_evidence", correct=0.0, confidence=0.60),
            row("qtrm-only", "qtrm_core_steps_8_qtrm_only_no_evidence", correct=1.0, confidence=0.95),
        ]
        candidate = [
            row("full", "qtrm_core_steps_8_no_evidence", correct=1.0, confidence=1.00),
            row("low-donor", "qtrm_core_steps_8_low_donor_no_evidence", correct=0.0, confidence=0.00),
            row("qtrm-only", "qtrm_core_steps_8_qtrm_only_no_evidence", correct=0.0, confidence=0.95),
        ]

        fused_gate = module.build_matched_metacognitive_gate(
            baseline,
            candidate,
            baseline_label="baseline",
            candidate_label="candidate",
            n_bins=5,
            gate_profile="fused",
        )

        self.assertEqual(fused_gate["status"], "accepted")
        self.assertEqual(fused_gate["gate_profile"], "fused")
        self.assertEqual(
            fused_gate["included_modes"],
            [
                "qtrm_core_steps_8_no_evidence",
                "qtrm_core_steps_8_low_donor_no_evidence",
            ],
        )
        self.assertNotIn("qtrm_core_steps_8_qtrm_only_no_evidence", fused_gate["mode_comparisons"])

    def test_report_writer_creates_json_and_markdown(self) -> None:
        module = load_script()
        row = {
            "id": "a",
            "mode": "qtrm_core_steps_8_no_evidence",
            "hit": True,
            "choice_scores": [{"choice": "A", "logprob": 1.0}, {"choice": "B", "logprob": 0.0}],
            "completion": "A",
            "answer_aliases": ["A"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_path = tmp_path / "baseline.jsonl"
            candidate_path = tmp_path / "candidate.jsonl"
            json_out = tmp_path / "summary.json"
            md_out = tmp_path / "summary.md"
            baseline_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            candidate_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            gate = module.write_metacognitive_gate_report(
                baseline_jsonl=str(baseline_path),
                candidate_jsonl=str(candidate_path),
                baseline_label="baseline",
                candidate_label="candidate",
                markdown_out=str(md_out),
                json_out=str(json_out),
                n_bins=5,
            )

            self.assertEqual(gate["record_count"], 2)
            self.assertTrue(json_out.exists())
            self.assertTrue(md_out.exists())


if __name__ == "__main__":
    unittest.main()
