from pathlib import Path
import unittest


class WorkspaceEvidencePathScriptTests(unittest.TestCase):
    def test_script_runs_workspace_injection_causality_modes(self):
        script = Path("scripts/117_run_workspace_evidence_path_probe.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("--evidence-injection workspace", script)
        self.assertIn("qtrm_residual_with_evidence", script)
        self.assertIn("qtrm_workspace_off_with_evidence", script)
        self.assertIn("qtrm_core_off_with_evidence", script)
        self.assertIn("qtrm_workspace_gate_off_with_evidence", script)
        self.assertIn("qtrm_workspace_memory_off_with_evidence", script)
        self.assertIn("qtrm_evidence_bottleneck_off_with_evidence", script)
        self.assertIn("scripts/113_build_expanded_ablation_proof.py", script)

    def test_train_script_uses_workspace_evidence_config_and_init_checkpoint(self):
        script = Path("scripts/118_run_workspace_evidence_path_train.sh").read_text(
            encoding="utf-8"
        )
        config = Path("configs/qwen35_2b_4090_workspace_evidence_path_s050.yaml").read_text(
            encoding="utf-8"
        )

        self.assertIn("qwen35_2b_4090_workspace_evidence_path_s050.yaml", script)
        self.assertIn("memory_gated_workspace_s050/last.pt", script)
        self.assertIn("--init-checkpoint", script)
        self.assertIn("scripts/117_run_workspace_evidence_path_probe.sh", script)
        self.assertIn("workspace_evidence_injection: true", config)

    def test_repeatguard_config_enables_conservative_unlikelihood_loss(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_workspace_evidence_repeatguard_s050.yaml")

        self.assertTrue(cfg.train.workspace_evidence_injection)
        self.assertEqual(cfg.model.qtrm_logits_scale, 0.5)
        self.assertEqual(cfg.model.donor_logits_scale, 1.0)
        self.assertGreater(cfg.train.loss_repeat_unlikelihood_weight, 0.0)
        self.assertLessEqual(cfg.train.loss_repeat_unlikelihood_weight, 0.05)
        self.assertIn("repeatguard", cfg.train.out_dir)

    def test_repeatguard_script_uses_repeatguard_config_and_post_eval(self):
        script = Path("scripts/119_run_workspace_evidence_repeatguard_train.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("qwen35_2b_4090_workspace_evidence_repeatguard_s050.yaml", script)
        self.assertIn("memory_gated_workspace_s050/last.pt", script)
        self.assertIn("--init-checkpoint", script)
        self.assertIn("scripts/117_run_workspace_evidence_path_probe.sh", script)
        self.assertIn("workspace-evidence-repeatguard-trained-ablation", script)

    def test_preference_config_enables_pairwise_alignment_loss(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_workspace_evidence_preference_s050.yaml")

        self.assertTrue(cfg.train.workspace_evidence_injection)
        self.assertEqual(cfg.train.loss_repeat_unlikelihood_weight, 0.0)
        self.assertGreater(cfg.train.loss_preference_weight, 0.0)
        self.assertGreater(cfg.train.preference_margin, 0.0)
        self.assertIn("preference", cfg.train.out_dir)

    def test_preference_script_uses_pairwise_data_and_post_eval(self):
        script = Path("scripts/120_run_workspace_evidence_preference_train.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("qwen35_2b_4090_workspace_evidence_preference_s050.yaml", script)
        self.assertIn("memory_self_improvement_preferences_analysis.jsonl", script)
        self.assertIn("memory_gated_workspace_s050/last.pt", script)
        self.assertIn("--init-checkpoint", script)
        self.assertIn("scripts/117_run_workspace_evidence_path_probe.sh", script)
        self.assertIn("scripts/121_eval_preference_pairs.py", script)
        self.assertIn("workspace-evidence-preference-trained-ablation", script)

    def test_preference_repeatguard_config_balances_alignment_and_repetition_guard(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_workspace_evidence_preference_repeatguard_s050.yaml"
        )

        self.assertTrue(cfg.train.workspace_evidence_injection)
        self.assertGreater(cfg.train.loss_preference_weight, 0.0)
        self.assertGreater(cfg.train.loss_repeat_unlikelihood_weight, 0.0)
        self.assertLessEqual(cfg.train.loss_repeat_unlikelihood_weight, 0.05)
        self.assertLessEqual(cfg.train.steps, 500)
        self.assertIn("preference_repeatguard", cfg.train.out_dir)

    def test_preference_repeatguard_script_uses_pairwise_data_and_post_eval(self):
        script = Path(
            "scripts/122_run_workspace_evidence_preference_repeatguard_train.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("qwen35_2b_4090_workspace_evidence_preference_repeatguard_s050.yaml", script)
        self.assertIn("memory_self_improvement_preferences_analysis.jsonl", script)
        self.assertIn("memory_gated_workspace_s050/last.pt", script)
        self.assertIn("--init-checkpoint", script)
        self.assertIn("scripts/117_run_workspace_evidence_path_probe.sh", script)
        self.assertIn("scripts/121_eval_preference_pairs.py", script)
        self.assertIn("workspace-evidence-preference-repeatguard-trained-ablation", script)

    def test_bounded_preference_config_preserves_donor_with_capped_residual(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_workspace_evidence_bounded_preference_s050.yaml"
        )

        self.assertTrue(cfg.train.workspace_evidence_injection)
        self.assertEqual(cfg.model.donor_logits_scale, 1.0)
        self.assertEqual(cfg.model.qtrm_residual_clamp, 1.0)
        self.assertTrue(cfg.model.qtrm_residual_gate_enabled)
        self.assertGreaterEqual(cfg.model.qtrm_residual_gate_min, 0.05)
        self.assertGreater(cfg.train.loss_student_lm_weight, 0.0)
        self.assertGreater(cfg.train.loss_donor_kl_weight, 0.0)
        self.assertGreater(cfg.train.loss_preference_weight, 0.0)
        self.assertGreater(cfg.train.loss_repeat_unlikelihood_weight, 0.0)
        self.assertLessEqual(cfg.train.steps, 300)

    def test_bounded_preference_script_uses_clean_init_and_step_checkpoints(self):
        script = Path(
            "scripts/123_run_workspace_evidence_bounded_preference_train.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("qwen35_2b_4090_workspace_evidence_bounded_preference_s050.yaml", script)
        self.assertIn("memory_self_improvement_preferences_analysis.jsonl", script)
        self.assertIn("memory_gated_workspace_s050/last.pt", script)
        self.assertIn("--init-checkpoint", script)
        self.assertIn("SAVE_EVERY=\"${SAVE_EVERY:-100}\"", script)
        self.assertIn("scripts/121_eval_preference_pairs.py", script)
        self.assertIn("workspace-evidence-bounded-preference-trained-ablation", script)

    def test_bounded_checkpoint_compare_script_keeps_midpoint_candidates(self):
        script = Path(
            "scripts/124_compare_bounded_preference_checkpoints.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("step_000100.pt", script)
        self.assertIn("step_000200.pt", script)
        self.assertIn("last.pt", script)
        self.assertIn("MAX_CASES=\"${MAX_CASES:-4}\"", script)
        self.assertIn("scripts/117_run_workspace_evidence_path_probe.sh", script)
        self.assertIn("workspace-evidence-bounded-preference-${name}-quick-ablation", script)

    def test_counterfactual_config_adds_workspace_causality_loss(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_workspace_evidence_counterfactual_s050.yaml"
        )

        self.assertTrue(cfg.train.workspace_evidence_injection)
        self.assertGreater(cfg.train.loss_workspace_contrastive_weight, 0.0)
        self.assertGreater(cfg.train.workspace_contrastive_margin, 0.0)
        self.assertGreater(cfg.train.loss_preference_weight, 0.0)
        self.assertTrue(cfg.model.core_context_enabled)
        self.assertIn("counterfactual", cfg.train.out_dir)

    def test_counterfactual_script_builds_data_before_training(self):
        script = Path(
            "scripts/126_run_workspace_evidence_counterfactual_train.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("qwen35_2b_4090_workspace_evidence_counterfactual_s050.yaml", script)
        self.assertIn("scripts/125_build_workspace_counterfactual_preferences.py", script)
        self.assertIn("memory_self_improvement_preferences_workspace_counterfactual.jsonl", script)
        self.assertIn("--init-checkpoint", script)
        self.assertIn("scripts/117_run_workspace_evidence_path_probe.sh", script)
        self.assertIn("workspace-evidence-counterfactual-trained-ablation", script)

    def test_logical_causal_bottleneck_config_gates_residual_with_evidence_heads(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_logical_causal_bottleneck_s050.yaml"
        )

        self.assertTrue(cfg.train.workspace_evidence_injection)
        self.assertTrue(cfg.model.evidence_bottleneck_enabled)
        self.assertTrue(cfg.model.evidence_bottleneck_suppress_without_workspace)
        self.assertGreater(cfg.train.loss_logical_evidence_weight, 0.0)
        self.assertGreater(cfg.train.loss_causal_evidence_gate_weight, 0.0)
        self.assertGreater(cfg.train.loss_workspace_contrastive_weight, 0.0)
        self.assertIn("logical_causal_bottleneck", cfg.train.out_dir)

    def test_logical_causal_bottleneck_script_reuses_counterfactual_data_and_proof(self):
        script = Path(
            "scripts/127_run_logical_causal_bottleneck_train.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("qwen35_2b_4090_logical_causal_bottleneck_s050.yaml", script)
        self.assertIn("scripts/125_build_workspace_counterfactual_preferences.py", script)
        self.assertIn("memory_self_improvement_preferences_workspace_counterfactual.jsonl", script)
        self.assertIn("memory_gated_workspace_s050/last.pt", script)
        self.assertIn("--init-checkpoint", script)
        self.assertIn("scripts/121_eval_preference_pairs.py", script)
        self.assertIn("scripts/117_run_workspace_evidence_path_probe.sh", script)
        self.assertIn("logical-causal-bottleneck-trained-ablation", script)

    def test_lewm_core_world_model_config_enables_action_conditioned_latent_prediction(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_lewm_core_world_model_probe_s050.yaml")

        self.assertTrue(cfg.model.core_world_model_enabled)
        self.assertEqual(cfg.model.outer_steps, 3)
        self.assertGreater(cfg.train.loss_core_world_model_weight, 0.0)
        self.assertGreater(cfg.model.core_world_model_sigreg_weight, 0.0)
        self.assertTrue(cfg.model.evidence_bottleneck_enabled)
        self.assertIn("lewm_core_world_model", cfg.train.out_dir)

    def test_lewm_core_world_model_script_uses_counterfactual_data_and_post_eval(self):
        script = Path("scripts/128_run_lewm_core_world_model_probe.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("qwen35_2b_4090_lewm_core_world_model_probe_s050.yaml", script)
        self.assertIn("scripts/125_build_workspace_counterfactual_preferences.py", script)
        self.assertIn("memory_self_improvement_preferences_workspace_counterfactual.jsonl", script)
        self.assertIn("memory_gated_workspace_s050/last.pt", script)
        self.assertIn("--init-checkpoint", script)
        self.assertIn("scripts/121_eval_preference_pairs.py", script)
        self.assertIn("scripts/117_run_workspace_evidence_path_probe.sh", script)
        self.assertIn("lewm-core-world-model-trained-ablation", script)

    def test_workspace_answer_bottleneck_config_forces_residual_through_latent_path(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_workspace_answer_bottleneck_s050.yaml")

        self.assertTrue(cfg.train.workspace_evidence_injection)
        self.assertTrue(cfg.model.answer_bottleneck_enabled)
        self.assertTrue(cfg.model.core_context_enabled)
        self.assertTrue(cfg.model.workspace_memory_gate_enabled)
        self.assertEqual(cfg.model.donor_logits_scale, 1.0)
        self.assertLessEqual(cfg.train.steps, 500)
        self.assertIn("workspace_answer_bottleneck", cfg.train.out_dir)

    def test_workspace_answer_bottleneck_script_runs_swap_and_root_gates(self):
        script = Path("scripts/129_run_workspace_answer_bottleneck_train.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("qwen35_2b_4090_workspace_answer_bottleneck_s050.yaml", script)
        self.assertIn("memory_gated_workspace_s050/last.pt", script)
        self.assertIn("--init-checkpoint", script)
        self.assertIn("scripts/117_run_workspace_evidence_path_probe.sh", script)
        self.assertIn("scripts/build_workspace_counterfactual_eval_cases.py", script)
        self.assertIn("workspace-answer-bottleneck-trained-root-gate", script)
        self.assertIn("workspace-answer-bottleneck-swap-root-gate", script)

    def test_workspace_answer_bottleneck_causal_script_trains_from_bottleneck_checkpoint(self):
        script = Path("scripts/149_run_workspace_answer_bottleneck_causal_train.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("qwen35_2b_4090_workspace_answer_bottleneck_causal_s050.yaml", script)
        self.assertIn("workspace_answer_bottleneck_s050/last.pt", script)
        self.assertIn("scripts/125_build_workspace_counterfactual_preferences.py", script)
        self.assertIn("memory_self_improvement_preferences_workspace_counterfactual.jsonl", script)
        self.assertIn("SHORT_ANSWER_GOVERNOR", script)
        self.assertIn("scripts/117_run_workspace_evidence_path_probe.sh", script)
        self.assertIn("workspace-answer-bottleneck-causal-root-gate", script)
        self.assertIn("workspace-answer-bottleneck-causal-swap-root-gate", script)

    def test_evidence_span_copy_gate_script_runs_normal_and_swap_gates(self):
        script = Path("scripts/151_run_evidence_span_copy_gate.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "qwen35_2b_4090_evidence_span_reader_wsno_hardneg_s050.yaml", script
        )
        self.assertIn(
            "qwen35_2b_4090_evidence_span_reader_wsno_hardneg_s050/last.pt", script
        )
        self.assertIn("--answer-channel evidence_span_copy", script)
        self.assertIn("--evidence-injection workspace", script)
        self.assertIn("qtrm_evidence_span_reader_off_with_evidence", script)
        self.assertIn("scripts/build_workspace_counterfactual_eval_cases.py", script)
        self.assertIn("evidence-span-copy-hardneg-normal-root-gate", script)
        self.assertIn("evidence-span-copy-hardneg-swap-root-gate", script)


if __name__ == "__main__":
    unittest.main()
