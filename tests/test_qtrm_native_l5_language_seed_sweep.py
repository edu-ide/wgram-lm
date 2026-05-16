import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path("scripts/341_qtrm_native_l5_language_seed_sweep.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_l5_language_seed_sweep", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QTRMNativeL5LanguageSeedSweepTests(unittest.TestCase):
    def test_l5c_standard_command_uses_text_nonregression_gate(self):
        module = load_module()

        command = module.l5c_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5c/seed_011"),
            seed=11,
            profile="standard",
        )

        self.assertIn("scripts/336_train_qtrm_native_text_probe.py", command)
        self.assertIn("--accepted-decision", command)
        self.assertIn("accepted_l5_language_nonregression", command)
        self.assertIn("--baseline-steps", command)
        self.assertIn("--max-full-vs-baseline-loss-ratio", command)
        self.assertIn("--seed", command)
        self.assertIn("11", command)

    def test_summarize_reports_rejects_when_baseline_ratio_is_high(self):
        module = load_module()
        reports = [
            {
                "seed": 1,
                "accepted": True,
                "eval_metrics": {"loss_ratios": {"full_vs_baseline": 1.01}},
            },
            {
                "seed": 2,
                "accepted": True,
                "eval_metrics": {"loss_ratios": {"full_vs_baseline": 1.41}},
            },
        ]

        summary = module.summarize_reports(
            reports,
            min_pass_rate=1.0,
            max_baseline_ratio=1.35,
            min_seeds=1,
        )

        self.assertFalse(summary["accepted"])
        self.assertEqual(summary["max_full_vs_baseline"], 1.41)
        self.assertEqual(summary["reject_reasons"], ["seed_baseline_ratio_above_threshold"])

    def test_reuse_existing_report_summarizes_without_running_training(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp)
            seed_dir = out_root / "seed_001"
            seed_dir.mkdir()
            (seed_dir / "report.json").write_text(
                json.dumps(
                    {
                        "accepted": True,
                        "decision": "accepted_l5_language_nonregression",
                        "eval_metrics": {"loss_ratios": {"full_vs_baseline": 1.01}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = module.build_arg_parser().parse_args(
                [
                    "--out-root",
                    tmp,
                    "--seeds",
                    "1",
                    "--reuse-existing",
                    "--min-seeds",
                    "1",
                ]
            )
            summary = module.run_sweep(args)

        self.assertTrue(summary["accepted"])
        self.assertEqual(summary["decision"], "accepted_l5c_seed_stability")
        self.assertEqual(summary["reports"][0]["seed"], 1)


if __name__ == "__main__":
    unittest.main()
