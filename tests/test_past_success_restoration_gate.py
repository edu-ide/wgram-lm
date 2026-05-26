from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "564_check_past_success_restoration_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("past_success_restoration_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PastSuccessRestorationGateTests(unittest.TestCase):
    def test_complete_gate_requires_language_generation_search_and_depth_signals(self) -> None:
        module = load_module()

        report = module.build_restoration_gate_report(
            teacher_forced_reports=[
                {"metric_family": "teacher_forced_loss", "final_eval_loss": 2.13}
            ],
            generation_reports=[
                {
                    "first_response": {
                        "accuracy": 0.25,
                        "eoa_top1_fraction": 0.0,
                        "gold_probability": 0.11,
                    },
                    "generation": {
                        "exact_fraction": 0.125,
                        "repeated_token_loop_fraction": 0.0,
                        "starts_with_eoa_fraction": 0.0,
                        "ended_with_eoa_fraction": 0.75,
                        "samples": [{"generated": "water", "gold": "water"}],
                    },
                    "response_continuation": {
                        "continuation_accuracy": 0.5,
                        "eos_targets": 2,
                        "eos_top1_accuracy": 0.5,
                    },
                }
            ],
            search_reports=[
                {
                    "metric_family": "selected_oracle_search",
                    "selected_accuracy": 0.93359375,
                    "oracle_accuracy": 0.9401041666666666,
                }
            ],
            depth_reports=[
                {
                    "probe_type": "blt_depth_residual_probe",
                    "depth_summaries": [
                        {"think_steps": 1, "loss": 3.0},
                        {"think_steps": 4, "loss": 2.5},
                    ],
                    "passed_checks": ["deepest_loss_beats_shallowest"],
                    "failed_checks": [],
                }
            ],
            require_search_split=True,
        )

        self.assertTrue(report["all_required_signals_present"])
        self.assertEqual(report["launch_recommendation"], "restoration_gate_exists_review_metrics")
        signal_by_name = {signal["name"]: signal for signal in report["signals"]}
        self.assertTrue(signal_by_name["teacher_forced_heldout_loss"]["present"])
        self.assertTrue(signal_by_name["free_generation_samples"]["present"])
        self.assertTrue(signal_by_name["first_response_token_rank_or_topk"]["present"])
        self.assertTrue(signal_by_name["repetition_and_eos_rate"]["present"])
        self.assertTrue(signal_by_name["selected_vs_oracle_split_when_search_is_used"]["present"])
        self.assertTrue(signal_by_name["depth_or_recurrent_core_off_ablation"]["present"])
        self.assertIn("시험지를 한 번에 본다", report["plain_korean_read"])
        self.assertEqual(report["current_checkpoint_recommendation"], "review_metric_quality_before_promotion")

    def test_missing_generation_and_depth_blocks_long_run(self) -> None:
        module = load_module()

        report = module.build_restoration_gate_report(
            teacher_forced_reports=[
                {"metric_family": "teacher_forced_loss", "final_eval_loss": 2.13}
            ],
            generation_reports=[],
            search_reports=[],
            depth_reports=[],
            require_search_split=False,
        )

        self.assertFalse(report["all_required_signals_present"])
        self.assertEqual(
            report["launch_recommendation"],
            "do_not_launch_long_run_missing_restoration_signals",
        )
        missing = report["missing_required_signals"]
        self.assertIn("free_generation_samples", missing)
        self.assertIn("first_response_token_rank_or_topk", missing)
        self.assertIn("depth_or_recurrent_core_off_ablation", missing)
        self.assertNotIn("selected_vs_oracle_split_when_search_is_used", missing)
        self.assertEqual(report["current_checkpoint_recommendation"], "not_interpretable")

    def test_observable_but_poor_generation_or_depth_rejects_current_checkpoint(self) -> None:
        module = load_module()

        report = module.build_restoration_gate_report(
            teacher_forced_reports=[{"final_eval_loss": 2.56}],
            generation_reports=[
                {
                    "first_response": {"accuracy": 1.0, "eoa_top1_fraction": 0.0},
                    "generation": {
                        "exact_fraction": 0.0,
                        "ended_with_eoa_fraction": 0.0,
                        "starts_with_eoa_fraction": 0.0,
                        "repeated_token_loop_fraction": 0.0,
                        "samples": [{"generated": "The euu", "gold": "water"}],
                    },
                    "response_continuation": {
                        "continuation_accuracy": 0.05,
                        "eos_targets": 8,
                        "eos_top1_accuracy": 0.0,
                    },
                }
            ],
            search_reports=[],
            depth_reports=[
                {
                    "depth_summaries": [
                        {"think_steps": 1, "loss": 2.56},
                        {"think_steps": 8, "loss": 2.61},
                    ],
                    "accepted": False,
                    "failed_checks": ["no_depth_loss_gain"],
                }
            ],
            require_search_split=False,
        )

        self.assertTrue(report["all_required_signals_present"])
        self.assertEqual(report["current_checkpoint_recommendation"], "do_not_promote_current_checkpoint")
        self.assertIn("free_generation_exact_zero", report["metric_warnings"])
        self.assertIn("generation_never_reaches_eos", report["metric_warnings"])
        self.assertIn("low_response_continuation_accuracy", report["metric_warnings"])
        self.assertIn("eos_teacher_forced_top1_zero", report["metric_warnings"])
        self.assertIn("depth_probe_rejected", report["metric_warnings"])

    def test_cli_loads_json_files_and_writes_report(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher = root / "teacher.json"
            generation = root / "generation.json"
            depth = root / "depth.json"
            out = root / "gate.json"
            teacher.write_text(json.dumps({"final_eval_loss": 1.5}), encoding="utf-8")
            generation.write_text(
                json.dumps(
                    {
                        "first_response": {"accuracy": 0.5},
                        "generation": {
                            "samples": [{"generated": "ok"}],
                            "repeated_token_loop_fraction": 0.0,
                            "starts_with_eoa_fraction": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            depth.write_text(
                json.dumps(
                    {
                        "depth_summaries": [
                            {"think_steps": 1, "loss": 2.0},
                            {"think_steps": 2, "loss": 1.8},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                module.main(
                    [
                        "--teacher-forced-report",
                        str(teacher),
                        "--generation-report",
                        str(generation),
                        "--depth-report",
                        str(depth),
                        "--out-json",
                        str(out),
                    ]
                )
            payload = json.loads(out.read_text(encoding="utf-8"))

        self.assertTrue(payload["all_required_signals_present"])
        self.assertEqual(payload["launch_recommendation"], "restoration_gate_exists_review_metrics")


if __name__ == "__main__":
    unittest.main()
