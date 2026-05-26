import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path("scripts/537_raw_intelligence_gate.py")
    spec = importlib.util.spec_from_file_location("raw_intelligence_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RawIntelligenceGateTests(unittest.TestCase):
    def test_prefixlm_eval_only_is_not_raw_intelligence_claim(self):
        module = load_module()
        rows = [
            {"step": 1, "loss": 9.0, "target_tokens_seen": 4},
            {"step": 10, "eval_loss": 4.5, "eval_target_tokens": 12},
        ]

        report = module.build_report_from_rows(rows)

        self.assertFalse(report["claim_raw_intelligence"])
        self.assertIn("language_body", report["covered_or_wired_axes"])
        self.assertIn("ood_generalization", report["weak_proxy_axes"])
        self.assertIn("reasoning", report["missing_axes"])
        self.assertIn("working_memory", report["missing_axes"])
        self.assertIn("metacognitive_control", report["missing_axes"])

    def test_token_verifier_rows_wire_verifier_axis(self):
        module = load_module()
        rows = [
            {
                "step": 2,
                "loss": 11.1,
                "lm_loss": 11.0,
                "token_verifier_loss": 0.7,
                "token_verifier_accuracy": 0.5,
            }
        ]

        axes = {axis.axis: axis for axis in module.assess_raw_intelligence(rows)}

        self.assertEqual(axes["verifier_judgment"].status, "wired_smoke")
        self.assertIn("token_verifier_loss=0.7", axes["verifier_judgment"].evidence)

    def test_cli_reads_json_lines_and_writes_report(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            out_path = Path(tmp) / "report.json"
            log_path.write_text(
                "\n".join(
                    [
                        "plain text",
                        json.dumps({"step": 1, "loss": 8.0}),
                        json.dumps({"step": 2, "eval_loss": 4.0}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = module.build_report([log_path])
            out_path.write_text(json.dumps(report), encoding="utf-8")
            loaded = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertFalse(loaded["claim_raw_intelligence"])
        self.assertIn("language_body", loaded["covered_or_wired_axes"])

    def test_report_json_loss_history_is_parsed_as_rows(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "loss_history": [
                            {
                                "step": 2,
                                "token_verifier_loss": 0.6,
                                "token_verifier_accuracy": 0.75,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rows = module.read_jsonl(report_path)
            report = module.build_report([report_path])

        self.assertEqual(rows[0]["token_verifier_loss"], 0.6)
        self.assertIn("verifier_judgment", report["covered_or_wired_axes"])

    def test_verifier_selection_report_marks_selection_tested(self):
        module = load_module()
        rows = [
            {
                "claim": "verifier_selected_candidate_beats_raw_lm",
                "raw_lm_top1_accuracy": 0.1,
                "verifier_selected_accuracy": 0.2,
                "verifier_gain": 0.1,
            }
        ]

        axes = {axis.axis: axis for axis in module.assess_raw_intelligence(rows)}
        report = module.build_report_from_rows(rows)

        self.assertEqual(axes["verifier_judgment"].status, "selection_tested")
        self.assertIn("verifier_judgment", report["covered_or_wired_axes"])
        self.assertIn("verifier_gain=0.1", axes["verifier_judgment"].evidence)


if __name__ == "__main__":
    unittest.main()
