import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path("scripts/343_qtrm_native_l5d_placement_seed_sweep.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_l5d_placement_seed_sweep", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QTRMNativeL5DPlacementSeedSweepTests(unittest.TestCase):
    def test_sweep_command_uses_backbone_compare_runner_and_limited_candidates(self):
        module = load_module()

        command = module.compare_seed_command(
            python_bin=".venv/bin/python",
            out_root=Path("local_eval/l5d_seed/seed_337"),
            seed=337,
            eval_seed=9337,
            profile="short",
            candidates="mha_etd,official_fla_think",
        )

        self.assertIn("scripts/342_qtrm_native_l5d_backbone_compare.py", command)
        self.assertIn("--candidates", command)
        self.assertIn("mha_etd,official_fla_think", command)
        self.assertIn("--seed", command)
        self.assertIn("337", command)
        self.assertIn("--eval-seed", command)
        self.assertIn("9337", command)

    def test_parser_accepts_standard_profile_for_scaled_seed_sweeps(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(["--profile", "standard"])

        self.assertEqual(args.profile, "standard")

    def test_summarize_seed_reports_accepts_promoted_target_for_all_seeds(self):
        module = load_module()
        reports = [
            {
                "seed": 337,
                "winner": "official_fla_think",
                "candidate_promotions": {
                    "official_fla_think": {
                        "promoted": True,
                        "full_generation_exact": 0.08,
                        "full_exact_delta_vs_mha": 0.06,
                        "causal_ok": True,
                        "backend_ok": True,
                    }
                },
            },
            {
                "seed": 338,
                "winner": "official_fla_think",
                "candidate_promotions": {
                    "official_fla_think": {
                        "promoted": True,
                        "full_generation_exact": 0.07,
                        "full_exact_delta_vs_mha": 0.03,
                        "causal_ok": True,
                        "backend_ok": True,
                    }
                },
            },
        ]

        summary = module.summarize_seed_reports(
            reports,
            target_candidate="official_fla_think",
            min_seeds=2,
            min_promoted_rate=1.0,
            min_delta_vs_mha=0.0,
        )

        self.assertTrue(summary["accepted"])
        self.assertEqual(summary["decision"], "accepted_l5d_placement_seed_stability")
        self.assertEqual(summary["target_candidate"], "official_fla_think")
        self.assertEqual(summary["promoted_count"], 2)
        self.assertEqual(summary["min_delta_vs_mha"], 0.03)

    def test_summarize_seed_reports_rejects_when_any_seed_not_promoted(self):
        module = load_module()
        reports = [
            {
                "seed": 337,
                "winner": "official_fla_think",
                "candidate_promotions": {
                    "official_fla_think": {
                        "promoted": True,
                        "full_generation_exact": 0.08,
                        "full_exact_delta_vs_mha": 0.06,
                        "causal_ok": True,
                        "backend_ok": True,
                    }
                },
            },
            {
                "seed": 338,
                "winner": "mha_etd",
                "candidate_promotions": {
                    "official_fla_think": {
                        "promoted": False,
                        "full_generation_exact": 0.04,
                        "full_exact_delta_vs_mha": -0.01,
                        "causal_ok": False,
                        "backend_ok": True,
                    }
                },
            },
        ]

        summary = module.summarize_seed_reports(
            reports,
            target_candidate="official_fla_think",
            min_seeds=2,
            min_promoted_rate=1.0,
            min_delta_vs_mha=0.0,
        )

        self.assertFalse(summary["accepted"])
        self.assertIn("promoted_rate_below_threshold", summary["reject_reasons"])
        self.assertIn("seed_delta_below_threshold", summary["reject_reasons"])

    def test_reuse_existing_compare_summaries_without_running_training(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp)
            seed_dir = out_root / "seed_337"
            seed_dir.mkdir()
            (seed_dir / "backbone_compare_summary.json").write_text(
                json.dumps(
                    {
                        "winner": "official_fla_think",
                        "candidate_promotions": {
                            "official_fla_think": {
                                "promoted": True,
                                "full_generation_exact": 0.08,
                                "full_exact_delta_vs_mha": 0.06,
                                "causal_ok": True,
                                "backend_ok": True,
                            }
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
                    "337",
                    "--reuse-existing",
                    "--min-seeds",
                    "1",
                ]
            )
            summary = module.run_sweep(args)

        self.assertTrue(summary["accepted"])
        self.assertEqual(summary["reports"][0]["seed"], 337)


if __name__ == "__main__":
    unittest.main()
