from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def _load_script():
    path = Path("scripts/398_score_m7a_answer_only_gate.py")
    spec = importlib.util.spec_from_file_location("m7a_answer_only_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class M7AAnswerOnlyGateTests(unittest.TestCase):
    def test_rejects_prompt_echo_and_empty_dominance(self):
        module = _load_script()
        with TemporaryDirectory() as tmp:
            eval_report = Path(tmp) / "eval.json"
            out = Path(tmp) / "gate.json"
            eval_report.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "cases": 256,
                            "accuracy": 0.003,
                            "invalid_pred_rate": 0.98,
                            "prompt_echo_rate": 1.0,
                            "pred_answer_histogram": {"<empty>": 251, "A": 5},
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = module.build_arg_parser().parse_args(
                ["--eval-report", str(eval_report), "--out-json", str(out)]
            )

            report = module.score_gate(args)

        self.assertFalse(report["accepted"])
        self.assertIn("invalid_pred_rate_le_max", report["reject_reasons"])
        self.assertIn("prompt_echo_rate_le_max", report["reject_reasons"])
        self.assertIn("max_pred_fraction_le_max", report["reject_reasons"])

    def test_accepts_valid_balanced_answer_only_distribution(self):
        module = _load_script()
        with TemporaryDirectory() as tmp:
            eval_report = Path(tmp) / "eval.json"
            out = Path(tmp) / "gate.json"
            eval_report.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "cases": 100,
                            "accuracy": 0.25,
                            "invalid_pred_rate": 0.0,
                            "prompt_echo_rate": 0.0,
                            "pred_answer_histogram": {"A": 25, "B": 25, "C": 25, "D": 25},
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = module.build_arg_parser().parse_args(
                ["--eval-report", str(eval_report), "--out-json", str(out)]
            )

            report = module.score_gate(args)

        self.assertTrue(report["accepted"])


if __name__ == "__main__":
    unittest.main()
