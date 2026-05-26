from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "562_build_past_success_doubt_report.py"


def load_module():
    spec = importlib.util.spec_from_file_location("past_success_doubt_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PastSuccessDoubtReportTests(unittest.TestCase):
    def test_ptrm_summary_is_classified_as_search_verifier_not_general_lm(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "history": [
                            {
                                "eval": {
                                    "mean_selected_accuracy_oracle_depth": 0.93359375,
                                    "mean_oracle_accuracy": 0.9401041666666666,
                                    "mean_packed_register_answer_accuracy_oracle_depth": 0.9934895833333334,
                                }
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            row = module.extract_ptrm_success_row(path, label="Stage58B")

        self.assertEqual(row["label"], "Stage58B")
        self.assertEqual(row["metric_family"], "selected_oracle_search")
        self.assertAlmostEqual(row["selected_accuracy"], 0.93359375)
        self.assertAlmostEqual(row["oracle_accuracy"], 0.9401041666666666)
        self.assertIn("candidate", row["plain_language_proves"])
        self.assertIn("free language generation", row["does_not_prove"])

    def test_language_loss_log_is_not_converted_into_accuracy_claim(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "language.log"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"step": 1, "eval_loss": 6.5}),
                        json.dumps(
                            {
                                "initial_eval_loss": 6.5,
                                "final_eval_loss": 2.13,
                                "plain_language_read": "teacher forced heldout loss",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            row = module.extract_language_loss_row(path, label="Stage94 raw-byte")

        self.assertEqual(row["metric_family"], "teacher_forced_loss")
        self.assertAlmostEqual(row["initial_eval_loss"], 6.5)
        self.assertAlmostEqual(row["final_eval_loss"], 2.13)
        self.assertNotIn("accuracy", row["plain_language_proves"].lower())
        self.assertIn("free generation", row["does_not_prove"])

    def test_language_loss_log_handles_pretty_summary_after_jsonl(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "language.log"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"step": 1, "loss": 6.5}),
                        json.dumps({"step": 400, "loss": 2.7}),
                        json.dumps(
                            {
                                "initial_eval_loss": 6.5,
                                "final_eval_loss": 2.13,
                                "eval_loss_history": [{"loss": 6.5}, {"loss": 2.13}],
                            },
                            indent=2,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            row = module.extract_language_loss_row(path, label="pretty")

        self.assertEqual(row["metric_family"], "teacher_forced_loss")
        self.assertAlmostEqual(row["initial_eval_loss"], 6.5)
        self.assertAlmostEqual(row["final_eval_loss"], 2.13)

    def test_report_contains_required_and_recommended_comparison_rows(self) -> None:
        module = load_module()
        report = module.build_past_success_doubt_report(
            old_rows=[
                {
                    "label": "Stage58B",
                    "metric_family": "selected_oracle_search",
                    "selected_accuracy": 0.93359375,
                    "oracle_accuracy": 0.9401041666666666,
                    "exact_metric": "selected=0.9336, oracle=0.9401",
                    "causal_ingredient": "candidate diversity plus verifier-selected compact answers",
                    "plain_language_proves": "candidate search/verifier works on compact arithmetic",
                    "does_not_prove": "free language generation",
                }
            ],
            current_rows=[
                {
                    "label": "Stage94 raw-byte",
                    "metric_family": "teacher_forced_loss",
                    "initial_eval_loss": 6.5,
                    "final_eval_loss": 2.13,
                    "plain_language_proves": "teacher-forced CE can drop",
                    "does_not_prove": "free generation",
                }
            ],
        )

        self.assertIn("required_comparison_row", report)
        template = report["required_comparison_row"]
        self.assertEqual(
            list(template),
            [
                "old_success",
                "exact_metric",
                "causal_ingredient",
                "missing_in_current_run",
                "smallest_restoration_test",
            ],
        )
        self.assertIn("recommended_comparison_row", report)
        recommended = report["recommended_comparison_row"]
        self.assertEqual(recommended["old_success"], "Stage58B")
        self.assertIn("selected=0.9336", recommended["exact_metric"])
        self.assertIn("candidate diversity", recommended["causal_ingredient"])
        self.assertIn("free generation", recommended["missing_in_current_run"])
        self.assertIn("selected-vs-oracle", recommended["smallest_restoration_test"])
        self.assertEqual(report["launch_recommendation"], "do_not_launch_long_run_until_restoration_gate_exists")
        markdown = module.render_markdown(report)
        self.assertIn("selected/oracle", markdown)
        self.assertIn("not general LM ability", markdown)
        self.assertIn("Recommended Comparison Row", markdown)


if __name__ == "__main__":
    unittest.main()
