from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


def _load_script():
    path = Path("scripts/372_qtrm_native_27b_milestone_status.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_27b_milestone_status", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QTRMNative27BMilestoneStatusTests(unittest.TestCase):
    def test_bridge_signal_is_partial_when_family_accuracy_misses_floor(self):
        module = _load_script()
        report = {
            "after_eval": {
                "gain": 0.064,
                "base_accuracy": 0.056,
                "core_accuracy": 0.121,
            },
            "after_family_summary": {
                "min_gain": 0.011,
                "min_core_accuracy": 0.093,
            },
            "after_language": {"top1_agreement": 1.0},
        }

        status = module.bridge_causal_signal(report)

        self.assertEqual(status["status"], "partial")
        self.assertFalse(status["accepted"])
        self.assertTrue(status["checks"]["gain_ge_0_05"])
        self.assertTrue(status["checks"]["min_family_gain_ge_0_01"])
        self.assertFalse(status["checks"]["min_family_core_accuracy_ge_0_10"])
        self.assertTrue(status["diagnostic_only"])

    def test_build_status_keeps_native_milestones_pending_without_native_report(self):
        module = _load_script()

        status = module.build_status()

        self.assertTrue(status["milestones"]["M0_TARGET_CONTRACT"]["accepted"])
        self.assertFalse(status["milestones"]["M2_NATIVE_TINY_LM"]["accepted"])
        self.assertIn("QTRM-Native-2B/3B", status["project_goal"])
        self.assertIn("gpqa_diamond", status["qwen36_27b_targets"])
        self.assertIn("M8_NATIVE_2B_3B_PUBLIC_BENCH_WIN", status["milestone_schedule"])
        self.assertIn("fast_path_estimate", status["milestone_schedule"]["M1_BRIDGE_CAUSAL_SIGNAL"])
        self.assertIn("M8_NATIVE_2B_3B_PUBLIC_BENCH_WIN", status["compute_plan"])
        self.assertEqual(status["compute_plan"]["M2_NATIVE_TINY_LM"]["4090_feasibility"], "yes")
        self.assertIn("published Qwen3.6", status["compute_plan"]["M5_QWEN36_EVAL_HARNESS"]["preferred_compute"])
        self.assertIn("public-target", status["compute_plan"]["M5_QWEN36_EVAL_HARNESS"]["4090_feasibility"])
        self.assertIn("M_A_RECURSIVE_CORE_REASONING_PROOF", status["next_action"])

    def test_bridge_stability_report_overrides_single_seed_report(self):
        module = _load_script()
        single_seed_report = {
            "after_eval": {"gain": 0.08},
            "after_family_summary": {
                "min_gain": 0.02,
                "min_core_accuracy": 0.12,
            },
            "after_language": {"top1_agreement": 1.0},
        }
        stability_report = {
            "accepted": False,
            "summary": {
                "num_seeds": 3,
                "num_accepted": 1,
                "min_gain": 0.03,
                "mean_gain": 0.05,
                "min_family_gain": -0.01,
                "mean_family_gain": 0.0,
                "min_family_core_accuracy": 0.08,
                "mean_family_core_accuracy": 0.10,
                "min_language_top1_agreement": 1.0,
            },
        }

        status = module.bridge_causal_signal(single_seed_report, stability_report)

        self.assertEqual(status["status"], "rejected")
        self.assertFalse(status["accepted"])
        self.assertTrue(status["stability_required"])
        self.assertEqual(status["metrics"]["num_accepted"], 1)

    def test_rejected_diagnostic_bridge_pivots_next_action_to_core_reasoning_proof(self):
        module = _load_script()
        stability_report = {
            "accepted": False,
            "summary": {
                "num_seeds": 3,
                "num_accepted": 1,
                "min_gain": 0.03,
                "mean_gain": 0.05,
                "min_family_gain": -0.01,
                "mean_family_gain": 0.0,
                "min_family_core_accuracy": 0.08,
                "mean_family_core_accuracy": 0.10,
                "min_language_top1_agreement": 1.0,
            },
        }

        status = module.build_status(bridge_stability_report=stability_report)

        self.assertIn("M_A_RECURSIVE_CORE_REASONING_PROOF", status["next_action"])

    def test_native_language_report_accepts_m2_and_leaves_m3_partial_without_reset(self):
        module = _load_script()
        report = {
            "accepted": True,
            "decision": "accepted_qtrm_native_language_bootstrap",
            "vocab_size": 8192,
            "reject_reasons": [],
            "pretrained_init": {"runtime_donor": False},
            "eval_metrics": {
                "think_eval_loss": 0.1,
                "think0_loss": 2.0,
                "thinking_block_off_loss": 2.0,
                "loss_ratios": {"full_vs_best_shallow_depth": 0.5},
                "sample_degeneracy": {
                    "unique_chars": 30.0,
                    "max_run_fraction": 0.02,
                },
            },
        }

        status = module.build_status(native_report=report)

        self.assertTrue(status["milestones"]["M2_NATIVE_TINY_LM"]["accepted"])
        self.assertEqual(status["milestones"]["M3_NATIVE_CORE_CAUSALITY"]["status"], "partial")
        self.assertFalse(status["milestones"]["M3_NATIVE_CORE_CAUSALITY"]["accepted"])
        self.assertEqual(status["milestones"]["M4_NATIVE_LANGUAGE_BOOTSTRAP"]["status"], "blocked")
        self.assertFalse(status["milestones"]["M4_NATIVE_LANGUAGE_BOOTSTRAP"]["accepted"])

    def test_native_core_causality_accepts_when_reset_degrades(self):
        module = _load_script()
        report = {
            "accepted": True,
            "decision": "accepted_qtrm_native_language_bootstrap",
            "eval_metrics": {
                "think_eval_loss": 0.1,
                "think0_loss": 2.0,
                "thinking_block_off_loss": 2.0,
                "loss_ratios": {"full_vs_best_shallow_depth": 0.5},
                "sample_degeneracy": {
                    "unique_chars": 30.0,
                    "max_run_fraction": 0.02,
                },
                "state_reset_ablation": {"loss": 1.4},
            },
        }

        status = module.native_core_causality_status(report)

        self.assertEqual(status["status"], "accepted")
        self.assertTrue(status["accepted"])
        self.assertTrue(status["metrics"]["reset_or_corruption_degrades"])

    def test_build_status_uses_separate_native_core_report(self):
        module = _load_script()
        language_report = {
            "accepted": True,
            "decision": "accepted_qtrm_native_language_bootstrap",
            "vocab_size": 8192,
            "eval_metrics": {
                "think_eval_loss": 0.1,
                "think0_loss": 2.0,
                "thinking_block_off_loss": 2.0,
                "loss_ratios": {"full_vs_best_shallow_depth": 0.5},
                "sample_degeneracy": {
                    "unique_chars": 30.0,
                    "max_run_fraction": 0.02,
                },
            },
        }
        core_report = {
            "accepted": True,
            "decision": "accepted_qtrm_native_core_causality",
            "eval_metrics": {
                "think_eval_loss": 0.1,
                "think0_loss": 2.0,
                "thinking_block_off_loss": 2.0,
                "loss_ratios": {"full_vs_best_shallow_depth": 0.5},
                "state_reset_ablation": {"loss": 1.4},
            },
        }

        status = module.build_status(
            native_report=language_report,
            native_core_report=core_report,
        )

        self.assertTrue(status["milestones"]["M2_NATIVE_TINY_LM"]["accepted"])
        self.assertTrue(status["milestones"]["M3_NATIVE_CORE_CAUSALITY"]["accepted"])

    def test_eval_manifest_accepts_m5(self):
        module = _load_script()
        manifest = {
            "accepted": True,
            "comparison_mode": "public_qwen36_target_scores",
            "direct_qwen36_rerun_required": False,
            "benchmark_map": {"gpqa_diamond": {}},
            "qtrm_native": {"artifacts": [{"path": "report.json"}]},
            "acceptance_checks": {
                "qwen_source_url_present": True,
                "all_targets_have_mapping": True,
            },
            "limitations": [],
        }

        status = module.build_status(eval_manifest=manifest)

        self.assertTrue(status["milestones"]["M5_QWEN36_EVAL_HARNESS"]["accepted"])
        self.assertIn(
            "accepted public-target manifest",
            status["milestone_schedule"]["M5_QWEN36_EVAL_HARNESS"]["actual_duration"],
        )
        self.assertFalse(
            status["milestones"]["M5_QWEN36_EVAL_HARNESS"]["metrics"][
                "direct_qwen36_rerun_required"
            ]
        )

    def test_m6_report_rejected_without_qwen_baseline_keeps_next_action_on_m6(self):
        module = _load_script()
        m6_report = {
            "accepted": False,
            "suite_id": "qtrm_native_text_reasoning_modchain_revchain_checksum_program4_mod32",
            "best_qtrm_native": {
                "full_generation_exact": 0.61,
                "core_gain": 0.59,
                "ablation_drop": 0.57,
                "min_family_generation_exact": 0.42,
            },
            "qwen36_baseline": None,
            "acceptance_checks": {"qwen36_baseline_present": False},
            "reject_reasons": ["qwen36_baseline_present"],
        }

        native_report = {
            "accepted": True,
            "decision": "accepted_qtrm_native_language_bootstrap",
            "vocab_size": 8192,
            "eval_metrics": {
                "think_eval_loss": 0.1,
                "think0_loss": 2.0,
                "thinking_block_off_loss": 2.0,
                "loss_ratios": {"full_vs_best_shallow_depth": 0.5},
                "sample_degeneracy": {
                    "unique_chars": 30.0,
                    "max_run_fraction": 0.02,
                },
            },
        }
        core_report = {
            "accepted": True,
            "decision": "accepted_qtrm_native_core_causality",
            "eval_metrics": {
                "think_eval_loss": 0.1,
                "think0_loss": 2.0,
                "thinking_block_off_loss": 2.0,
                "loss_ratios": {"full_vs_best_shallow_depth": 0.5},
                "state_reset_ablation": {"loss": 1.4},
            },
        }
        status = module.build_status(
            bridge_stability_report={"accepted": False, "summary": {}},
            native_report=native_report,
            native_core_report=core_report,
            core_reasoning_report={
                "accepted": True,
                "decision": "accepted_l5_multifamily",
                "decisive_metrics": {
                    "full_generation_exact": 0.61,
                    "min_family_generation_exact": 0.42,
                    "full_minus_think0": 0.59,
                    "full_minus_worst_ablation": 0.57,
                },
            },
            eval_manifest={"accepted": True},
            m6_report=m6_report,
        )

        m6 = status["milestones"]["M6_NATIVE_SMALL_BEATS_27B_ON_SCOPED_RAW_REASONING"]
        self.assertEqual(m6["status"], "rejected")
        self.assertFalse(m6["accepted"])
        self.assertEqual(m6["metrics"]["qtrm_score"], 0.61)
        self.assertIn("qwen36_baseline_present", m6["reject_reasons"])
        self.assertIn(
            "rejected until matched Qwen3.6 scoped baseline exists",
            status["milestone_schedule"]["M6_NATIVE_SMALL_BEATS_27B_ON_SCOPED_RAW_REASONING"][
                "actual_duration"
            ],
        )
        self.assertIn("M6_NATIVE_SMALL_BEATS_27B", status["next_action"])

    def test_markdown_renderer_includes_benchmarks_and_milestones(self):
        module = _load_script()
        status = module.build_status()

        markdown = module.render_markdown(status)

        self.assertIn("QTRM-Native 27B Target Status", markdown)
        self.assertIn("gpqa_diamond", markdown)
        self.assertIn("Core-to-LM-to-Healing Dependency", markdown)
        self.assertIn("Fast-Path Schedule", markdown)
        self.assertIn("Fast-Path Strategy", markdown)
        self.assertIn("Compute Plan", markdown)
        self.assertIn("M8_NATIVE_2B_3B_PUBLIC_BENCH_WIN", markdown)

    def test_core_to_lm_to_healing_dependency_accepts_only_in_order(self):
        module = _load_script()
        language_report = {
            "accepted": True,
            "decision": "accepted_qtrm_native_language_bootstrap",
            "vocab_size": 16000,
            "pretrained_init": {"runtime_donor": False},
            "eval_metrics": {},
        }
        core_reasoning_report = {
            "accepted": True,
            "decision": "accepted_l5_multifamily",
            "decisive_metrics": {
                "full_generation_exact": 0.61,
                "min_family_generation_exact": 0.42,
                "full_minus_think0": 0.59,
                "full_minus_worst_ablation": 0.57,
            },
        }

        status = module.build_status(
            native_report=language_report,
            core_reasoning_report=core_reasoning_report,
        )

        deps = status["core_to_lm_to_healing_dependencies"]
        self.assertTrue(deps["M_A_RECURSIVE_CORE_REASONING_PROOF"]["accepted"])
        self.assertTrue(deps["M_B_CORE_TO_LM_ATTACHMENT"]["accepted"])
        self.assertTrue(deps["M_C_LANGUAGE_HEALING_AFTER_CORE"]["accepted"])
        self.assertTrue(status["milestones"]["M4_NATIVE_LANGUAGE_BOOTSTRAP"]["accepted"])

    def test_m7_public_benchmark_report_rejected_when_below_parity(self):
        module = _load_script()

        status = module.build_status(
            m7_report={
                "accepted": False,
                "benchmark_id": "mmlu_pro",
                "benchmark_name": "MMLU-Pro",
                "qwen36_target_score": 0.862,
                "parity_floor": 0.842,
                "min_cases_for_parity": 256,
                "metrics": {
                    "accuracy": 0.10,
                    "cases": 64,
                    "invalid_pred_rate": 0.90,
                    "prompt_echo_rate": 1.0,
                    "pred_answer_histogram": {"<empty>": 60, "A": 4},
                    "by_category": {"math": {"accuracy": 0.10}},
                },
                "limitations": ["subset only"],
            }
        )

        m7 = status["milestones"]["M7_NATIVE_2B_3B_PUBLIC_BENCH_PARITY"]
        self.assertEqual(m7["status"], "rejected")
        self.assertFalse(m7["accepted"])
        self.assertEqual(m7["metrics"]["benchmark_id"], "mmlu_pro")
        self.assertEqual(m7["metrics"]["prompt_echo_rate"], 1.0)
        self.assertEqual(m7["metrics"]["pred_answer_histogram"]["<empty>"], 60)
        self.assertIn("score_ge_parity_floor", m7["reject_reasons"])
        self.assertIn("cases_ge_min", m7["reject_reasons"])
        self.assertIn(
            "public benchmark parity rejected",
            status["milestone_schedule"]["M7_NATIVE_2B_3B_PUBLIC_BENCH_PARITY"][
                "actual_duration"
            ],
        )

    def test_m7_public_benchmark_report_accepted_when_inside_parity_band(self):
        module = _load_script()

        status = module.build_status(
            m7_report={
                "accepted": True,
                "benchmark_id": "mmlu_pro",
                "benchmark_name": "MMLU-Pro",
                "qwen36_target_score": 0.862,
                "parity_floor": 0.842,
                "min_cases_for_parity": 256,
                "metrics": {"accuracy": 0.85, "cases": 256},
            }
        )

        m7 = status["milestones"]["M7_NATIVE_2B_3B_PUBLIC_BENCH_PARITY"]
        self.assertEqual(m7["status"], "accepted")
        self.assertTrue(m7["accepted"])


if __name__ == "__main__":
    unittest.main()
