from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "591_wgram_v2_fastlane.py"


def load_module():
    spec = importlib.util.spec_from_file_location("wgram_v2_fastlane", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class WGRAMV2FastlaneTests(unittest.TestCase):
    def test_fastlane_plan_is_single_core_recipe_not_comparison_sweep(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            sampled = Path(tmp) / "sampled"
            sampled.mkdir()
            args = module.build_arg_parser().parse_args(
                [
                    "--sampled-data",
                    str(sampled),
                    "--out-root",
                    str(Path(tmp) / "runs"),
                    "--run-name",
                    "unit_fastlane",
                    "--dry-run",
                ]
            )

            plan = module.build_fastlane_plan(args)
            encoded = json.dumps(plan, ensure_ascii=False)

        self.assertEqual(plan["experiment_type"], "single_core_wgram_v2_fastlane")
        self.assertEqual(plan["comparison_policy"], "non_core_comparisons_deferred")
        self.assertEqual(plan["recipe"]["imta_trajectories"], 3)
        self.assertEqual(plan["recipe"]["grad_accum_steps"], 4)
        self.assertEqual(plan["recipe"]["effective_tokens_per_optimizer_step"], 512)
        self.assertEqual(plan["recipe"]["lr_schedule"], "warmup_cosine")
        self.assertEqual(plan["recipe"]["optimizer_warmup_steps"], 9)
        self.assertEqual(plan["recipe"]["min_lr_ratio"], 0.1)
        self.assertTrue(plan["recipe"]["tensorboard_logdir"].endswith("/unit_fastlane"))
        self.assertEqual(plan["recipe"]["aim_repo"], "/tmp/wgram_aim")
        self.assertEqual(plan["recipe"]["aim_experiment"], "wgram_v2_fastlane")
        self.assertEqual(plan["recipe"]["max_position_embeddings"], 4096)
        self.assertEqual(plan["recipe"]["max_response_position_embeddings"], 1024)
        self.assertFalse(plan["recipe"]["tie_input_output_embeddings"])
        self.assertTrue(plan["recipe"]["use_response_phase_embeddings"])
        self.assertFalse(plan["recipe"]["force_fixed_boundaries"])
        self.assertGreater(plan["recipe"]["own_latent_prediction_weight"], 0.0)
        self.assertGreater(plan["recipe"]["imta_route_min_probability"], 0.0)
        self.assertGreater(plan["recipe"]["imta_route_entropy_floor"], 0.0)
        self.assertGreater(plan["recipe"]["imta_route_entropy_weight"], 0.0)
        self.assertGreater(plan["recipe"]["imta_route_balance_weight"], 0.0)
        self.assertEqual(plan["recipe"]["repeat_unlikelihood_weight"], 0.0)
        self.assertGreater(plan["recipe"]["premature_stop_loss_weight"], 0.0)
        self.assertGreater(plan["recipe"]["response_start_loss_weight"], 0.0)
        self.assertGreater(plan["recipe"]["response_start_stop_margin_weight"], 0.0)
        self.assertEqual(plan["recipe"]["response_continue_stop_margin_weight"], 0.0)
        self.assertEqual(plan["recipe"]["response_continue_stop_margin_start_after"], 195)
        self.assertEqual(plan["recipe"]["response_continue_stop_margin_warmup_steps"], 105)
        self.assertGreater(plan["recipe"]["response_body_loss_weight"], 0.0)
        self.assertGreater(plan["recipe"]["response_stop_loss_weight"], 0.0)
        self.assertGreater(plan["recipe"]["response_stop_loss_start_after"], 1)
        self.assertGreater(plan["recipe"]["response_stop_loss_warmup_steps"], 0)
        self.assertEqual(plan["recipe"]["response_stop_loss_start_after"], 195)
        self.assertEqual(plan["recipe"]["response_stop_loss_warmup_steps"], 105)
        self.assertEqual(plan["recipe"]["token_maturation_steps"], 2)
        self.assertGreater(plan["recipe"]["token_maturation_aux_loss_weight"], 0.0)
        self.assertTrue(plan["recipe"]["answer_memory"])
        self.assertEqual(plan["recipe"]["answer_memory_steps"], 2)
        self.assertEqual(plan["recipe"]["answer_memory_plan_tokens"], 4)
        self.assertEqual(plan["recipe"]["answer_memory_plan_layers"], 1)
        self.assertFalse(plan["recipe"]["answer_memory_prompt_context"])
        self.assertEqual(plan["recipe"]["answer_memory_prompt_context_gate_init"], -1.0)
        self.assertGreater(plan["recipe"]["answer_memory_aux_loss_weight"], 0.0)
        self.assertTrue(plan["recipe"]["answer_memory_confidence_gate"])
        self.assertEqual(plan["recipe"]["answer_memory_confidence_mode"], "topk_mass")
        self.assertEqual(plan["recipe"]["answer_memory_confidence_topk"], 5)
        self.assertGreater(plan["recipe"]["answer_memory_confidence_floor"], 0.0)
        self.assertEqual(plan["recipe"]["answer_memory_stop_margin_loss_weight"], 0.0)
        self.assertGreater(plan["recipe"]["answer_memory_stop_margin"], 0.0)
        self.assertEqual(plan["recipe"]["answer_memory_commitment_scale"], 1.0)
        self.assertFalse(plan["recipe"]["answer_memory_commitment_confidence_gate"])
        self.assertGreater(plan["recipe"]["answer_prefix_commitment_loss_weight"], 0.0)
        self.assertEqual(plan["recipe"]["answer_memory_commitment_start_after"], 105)
        self.assertEqual(plan["recipe"]["answer_memory_commitment_warmup_steps"], 90)
        self.assertEqual(plan["recipe"]["answer_memory_injection_start_after"], 195)
        self.assertEqual(plan["recipe"]["answer_memory_injection_warmup_steps"], 105)
        self.assertTrue(plan["recipe"]["adaptive_latent_bridge"])
        self.assertEqual(plan["recipe"]["init_from_blt_checkpoint"], "")
        self.assertEqual(plan["recipe"]["init_from_v2_checkpoint"], "")
        self.assertGreater(plan["recipe"]["stability_activation_clip_value"], 0.0)
        self.assertGreater(plan["recipe"]["self_rollout_loss_weight"], 0.0)
        self.assertGreater(plan["recipe"]["self_rollout_max_tokens"], 0)
        self.assertTrue(plan["recipe"]["balanced_response_sampler"])
        self.assertEqual(plan["recipe"]["generation_repetition_penalty"], 1.0)
        self.assertEqual(plan["recipe"]["core_implementation"], "official_gated_delta2")
        self.assertTrue(plan["recipe"]["official_gdn2_force_chunk_eval"])
        self.assertIn("590_train_wgram_v2_prefixlm.py", plan["train_command"])
        self.assertIn("--require-promotion-ready", plan["train_command"])
        self.assertIn("--imta-trajectories 3", plan["train_command"])
        self.assertIn("--grad-accum-steps 4", plan["train_command"])
        self.assertIn("--lr-schedule warmup_cosine", plan["train_command"])
        self.assertIn("--optimizer-warmup-steps 9", plan["train_command"])
        self.assertIn("--tensorboard-logdir /tmp/wgram_eval/unit_fastlane", plan["train_command"])
        self.assertIn("--aim-repo /tmp/wgram_aim", plan["train_command"])
        self.assertIn("--aim-experiment wgram_v2_fastlane", plan["train_command"])
        self.assertIn("--max-position-embeddings 4096", plan["train_command"])
        self.assertIn("--max-response-position-embeddings 1024", plan["train_command"])
        self.assertIn("--disable-tied-input-output-embeddings", plan["train_command"])
        self.assertNotIn("--disable-response-phase-embeddings", plan["train_command"])
        self.assertIn("--dynamic-boundary-threshold", plan["train_command"])
        self.assertIn("--own-latent-prediction-weight", plan["train_command"])
        self.assertIn("--imta-route-min-probability", plan["train_command"])
        self.assertIn("--imta-route-entropy-floor", plan["train_command"])
        self.assertIn("--imta-route-entropy-weight", plan["train_command"])
        self.assertIn("--imta-route-balance-weight", plan["train_command"])
        self.assertIn("--repeat-unlikelihood-weight", plan["train_command"])
        self.assertIn("--premature-stop-loss-weight", plan["train_command"])
        self.assertIn("--response-start-loss-weight", plan["train_command"])
        self.assertIn("--response-start-stop-margin-weight", plan["train_command"])
        self.assertIn("--response-continue-stop-margin-weight", plan["train_command"])
        self.assertIn("--response-continue-stop-margin-start-after 195", plan["train_command"])
        self.assertIn("--response-continue-stop-margin-warmup-steps 105", plan["train_command"])
        self.assertIn("--response-body-loss-weight", plan["train_command"])
        self.assertIn("--response-stop-loss-weight", plan["train_command"])
        self.assertIn("--response-stop-loss-start-after 195", plan["train_command"])
        self.assertIn("--response-stop-loss-warmup-steps 105", plan["train_command"])
        self.assertIn("--token-maturation-steps 2", plan["train_command"])
        self.assertIn("--token-maturation-aux-loss-weight", plan["train_command"])
        self.assertIn("--answer-memory-steps 2", plan["train_command"])
        self.assertIn("--answer-memory-plan-tokens 4", plan["train_command"])
        self.assertIn("--answer-memory-plan-layers 1", plan["train_command"])
        self.assertIn("--no-answer-memory-prompt-context", plan["train_command"])
        self.assertIn("--answer-memory-prompt-context-gate-init -1.0", plan["train_command"])
        self.assertIn("--answer-memory-aux-loss-weight", plan["train_command"])
        self.assertIn("--answer-memory-confidence-mode topk_mass", plan["train_command"])
        self.assertIn("--answer-memory-confidence-topk 5", plan["train_command"])
        self.assertIn("--answer-memory-confidence-floor", plan["train_command"])
        self.assertIn("--answer-memory-stop-margin-loss-weight", plan["train_command"])
        self.assertIn("--answer-memory-stop-margin", plan["train_command"])
        self.assertIn("--answer-memory-commitment-scale 1.0", plan["train_command"])
        self.assertIn("--no-answer-memory-commitment-confidence-gate", plan["train_command"])
        self.assertIn("--answer-prefix-commitment-loss-weight", plan["train_command"])
        self.assertIn("--answer-memory-commitment-start-after 105", plan["train_command"])
        self.assertIn("--answer-memory-commitment-warmup-steps 90", plan["train_command"])
        self.assertIn("--answer-memory-injection-start-after 195", plan["train_command"])
        self.assertIn("--answer-memory-injection-warmup-steps 105", plan["train_command"])
        self.assertIn("--adaptive-latent-bridge-gate-init", plan["train_command"])
        self.assertIn("--byte-residual-gate-init", plan["train_command"])
        self.assertIn("--latent-residual-gate-init", plan["train_command"])
        self.assertIn("--stability-activation-clip-value", plan["train_command"])
        self.assertIn("--self-rollout-loss-weight", plan["train_command"])
        self.assertIn("--balanced-response-sampler", plan["train_command"])
        self.assertIn("--generation-repetition-penalty", plan["eval_command"])
        self.assertNotIn("--force-fixed-boundaries", plan["train_command"])
        self.assertNotIn("--official-gdn2-fused-recurrent-eval", plan["train_command"])
        self.assertNotIn("K=1", encoded)
        self.assertNotIn("ablation", encoded.lower())
        self.assertNotIn("forced_choice", encoded)
        self.assertNotIn("candidate_rerank", encoded)

    def test_fastlane_smoke_plan_is_explicitly_non_promotion(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            sampled = Path(tmp) / "sampled"
            sampled.mkdir()
            args = module.build_arg_parser().parse_args(
                [
                    "--sampled-data",
                    str(sampled),
                    "--out-root",
                    str(Path(tmp) / "runs"),
                    "--run-name",
                    "unit_smoke",
                    "--smoke",
                    "--dry-run",
                ]
            )

            plan = module.build_fastlane_plan(args)

        self.assertEqual(plan["recipe"]["runtime_profile"], "smoke")
        self.assertEqual(plan["recipe"]["core_implementation"], "torch_smoke")
        self.assertEqual(plan["recipe"]["grad_accum_steps"], 1)
        self.assertIn("--allow-torch-smoke-core", plan["train_command"])
        self.assertIn("--grad-accum-steps 1", plan["train_command"])
        self.assertNotIn("--require-promotion-ready", plan["train_command"])

    def test_fastlane_duplicate_guard_requires_force(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            sampled = Path(tmp) / "sampled"
            sampled.mkdir()
            out_root = Path(tmp) / "runs"
            run_dir = out_root / "duplicate"
            run_dir.mkdir(parents=True)
            (run_dir / "fastlane_manifest.json").write_text("{}", encoding="utf-8")
            args = module.build_arg_parser().parse_args(
                [
                    "--sampled-data",
                    str(sampled),
                    "--out-root",
                    str(out_root),
                    "--run-name",
                    "duplicate",
                    "--dry-run",
                ]
            )

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                module.assert_not_duplicate(args)

    def test_generation_repetition_stats_flags_diluted_long_low_diversity_loops(self) -> None:
        from wgram_lm.v2.generation import generation_repetition_stats

        stats = generation_repetition_stats(
            [31108, 236, 114, 236, 48, 121, 48, 236, 48, 236, 48, 315, 236]
            + [48, 236] * 12
            + [39, 11]
        )

        self.assertTrue(stats["loop_like"])
        self.assertLessEqual(stats["unique_fraction"], 0.25)
        self.assertGreaterEqual(stats["long_low_diversity_fraction"], 0.35)

    def test_generation_repetition_stats_flags_medium_low_diversity_answer_drift(self) -> None:
        from wgram_lm.v2.generation import generation_repetition_stats

        stats = generation_repetition_stats(
            [31108, 236, 114, 236, 48, 121, 48, 236, 48, 236, 48, 315, 236]
            + [48, 236] * 5
            + [11]
        )

        self.assertTrue(stats["loop_like"])
        self.assertLessEqual(stats["unique_fraction"], 0.30)
        self.assertGreaterEqual(stats["max_token_count_fraction"], 0.35)

    def test_first_token_consistency_flags_teacher_forced_autoregressive_mismatch(self) -> None:
        from wgram_lm.v2.generation import first_token_consistency_stats

        stats = first_token_consistency_stats(
            [7, 11],
            {
                "available": True,
                "top1_id": 5,
                "gold_token_id": 5,
                "top5_ids": [5, 7, 9],
            },
            deterministic_free_decode=True,
        )

        self.assertTrue(stats["available"])
        self.assertTrue(stats["consistency_required"])
        self.assertFalse(stats["consistency_pass"])
        self.assertFalse(stats["matches_teacher_forced_top1"])
        self.assertTrue(stats["matches_teacher_forced_top5"])
        self.assertFalse(stats["matches_gold"])

    def test_first_token_consistency_does_not_require_match_for_stochastic_decode(self) -> None:
        from wgram_lm.v2.generation import first_token_consistency_stats

        stats = first_token_consistency_stats(
            [7],
            {"available": True, "top1_id": 5, "gold_token_id": 5, "top5_ids": [5, 7]},
            deterministic_free_decode=False,
        )

        self.assertFalse(stats["consistency_required"])
        self.assertTrue(stats["consistency_pass"])


if __name__ == "__main__":
    unittest.main()
