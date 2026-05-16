import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path("scripts/340_qtrm_native_l5_seed_sweep.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_l5_seed_sweep", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QTRMNativeL5SeedSweepTests(unittest.TestCase):
    def test_l5_standard_command_uses_multifamily_gate(self):
        module = load_module()

        command = module.l5_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5/seed_011"),
            seed=11,
            eval_seed=1011,
            profile="standard",
        )

        self.assertIn("scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py", command)
        self.assertIn("--task-families", command)
        self.assertIn("modchain,revchain,modchain,revchain,checksum", command)
        self.assertIn("--eval-task-families", command)
        self.assertIn("modchain,revchain,checksum", command)
        self.assertIn("--accepted-decision", command)
        self.assertIn("accepted_l5_multifamily", command)
        self.assertIn("--accept-min-family-exact", command)
        self.assertIn("--seed", command)
        self.assertIn("11", command)

    def test_summarize_reports_rejects_when_min_family_exact_is_low(self):
        module = load_module()
        reports = [
            {
                "seed": 1,
                "accepted": True,
                "decisive_metrics": {
                    "full_generation_exact": 0.62,
                    "min_family_generation_exact": 0.39,
                },
            },
            {
                "seed": 2,
                "accepted": True,
                "decisive_metrics": {
                    "full_generation_exact": 0.61,
                    "min_family_generation_exact": 0.42,
                },
            },
        ]

        summary = module.summarize_reports(
            reports,
            min_pass_rate=1.0,
            min_exact=0.60,
            min_family_exact=0.40,
            min_seeds=1,
        )

        self.assertFalse(summary["accepted"])
        self.assertEqual(summary["min_full_generation_exact"], 0.61)
        self.assertEqual(summary["min_family_generation_exact"], 0.39)
        self.assertEqual(summary["reject_reasons"], ["seed_family_exact_below_threshold"])

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
                        "decision": "accepted_l5_multifamily",
                        "decisive_metrics": {
                            "full_generation_exact": 0.62,
                            "min_family_generation_exact": 0.41,
                        },
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
        self.assertEqual(summary["decision"], "accepted_l5_seed_stability")
        self.assertEqual(summary["reports"][0]["seed"], 1)


if __name__ == "__main__":
    unittest.main()
