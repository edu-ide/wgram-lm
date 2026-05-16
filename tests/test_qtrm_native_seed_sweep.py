import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path("scripts/339_qtrm_native_seed_sweep.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_seed_sweep", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QTRMNativeSeedSweepTests(unittest.TestCase):
    def test_l4_standard_command_includes_seed_and_canonical_capacity(self):
        module = load_module()

        command = module.l4_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/sweep/seed_011"),
            seed=11,
            eval_seed=1011,
            profile="standard",
        )

        self.assertIn("scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py", command)
        self.assertIn("--seed", command)
        self.assertIn("11", command)
        self.assertIn("--eval-seed", command)
        self.assertIn("1011", command)
        self.assertIn("--steps", command)
        self.assertIn("8000", command)
        self.assertIn("--d-model", command)
        self.assertIn("128", command)
        self.assertIn("--active-len-curriculum", command)

    def test_summarize_reports_tracks_pass_rate_and_min_exact(self):
        module = load_module()
        reports = [
            {
                "seed": 1,
                "accepted": True,
                "decision": "accepted_l4_mixed_text_reasoning",
                "decisive_metrics": {"full_generation_exact": 0.74},
            },
            {
                "seed": 2,
                "accepted": False,
                "decision": "rejected",
                "decisive_metrics": {"full_generation_exact": 0.61},
            },
        ]

        summary = module.summarize_reports(
            reports,
            min_pass_rate=0.75,
            min_exact=0.70,
            min_seeds=1,
        )

        self.assertFalse(summary["accepted"])
        self.assertEqual(summary["pass_count"], 1)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["min_full_generation_exact"], 0.61)
        self.assertEqual(
            summary["reject_reasons"],
            ["pass_rate_below_threshold", "seed_exact_below_threshold"],
        )

    def test_summarize_reports_rejects_when_any_seed_exact_is_below_threshold(self):
        module = load_module()
        reports = [
            {
                "seed": 1,
                "accepted": True,
                "decision": "accepted_l4_mixed_text_reasoning",
                "decisive_metrics": {"full_generation_exact": 0.74},
            },
            {
                "seed": 2,
                "accepted": True,
                "decision": "accepted_l4_mixed_text_reasoning",
                "decisive_metrics": {"full_generation_exact": 0.69},
            },
        ]

        summary = module.summarize_reports(
            reports,
            min_pass_rate=1.0,
            min_exact=0.70,
            min_seeds=1,
        )

        self.assertFalse(summary["accepted"])
        self.assertEqual(summary["pass_rate"], 1.0)
        self.assertEqual(summary["min_full_generation_exact"], 0.69)
        self.assertEqual(summary["reject_reasons"], ["seed_exact_below_threshold"])

    def test_summarize_reports_rejects_when_seed_count_is_too_low(self):
        module = load_module()
        reports = [
            {
                "seed": 338,
                "accepted": True,
                "decision": "accepted_l4_mixed_text_reasoning",
                "decisive_metrics": {"full_generation_exact": 0.746},
            }
        ]

        summary = module.summarize_reports(
            reports,
            min_pass_rate=1.0,
            min_exact=0.70,
            min_seeds=3,
        )

        self.assertFalse(summary["accepted"])
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["reject_reasons"], ["seed_count_below_threshold"])

    def test_dry_run_writes_summary_without_executing_training(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            args = module.build_arg_parser().parse_args(
                [
                    "--out-root",
                    tmp,
                    "--seeds",
                    "1",
                    "2",
                    "--dry-run",
                ]
            )
            summary = module.run_sweep(args)
            path = Path(tmp) / "seed_sweep_summary.json"
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(summary["decision"], "dry_run")
        self.assertEqual(len(loaded["commands"]), 2)
        self.assertEqual(loaded["commands"][0]["seed"], 1)

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
                        "decision": "accepted_l4_mixed_text_reasoning",
                        "decisive_metrics": {"full_generation_exact": 0.73},
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
        self.assertEqual(summary["reports"][0]["seed"], 1)
        self.assertEqual(summary["reports"][0]["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
