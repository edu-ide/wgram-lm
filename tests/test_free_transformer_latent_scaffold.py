from pathlib import Path
import unittest

import yaml


class FreeTransformerLatentScaffoldTests(unittest.TestCase):
    def test_config_enables_posterior_prior_latent_path(self):
        path = Path(
            "configs/"
            "qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_"
            "core_state_only_kiss_free_transformer_latent_s040.yaml"
        )
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

        model = cfg["model"]
        self.assertTrue(model["answer_state_loop_enabled"])
        self.assertTrue(model["answer_state_loop_core_state_only_enabled"])
        self.assertTrue(model["answer_state_loop_next_token_decoder_enabled"])
        self.assertEqual(model["answer_state_loop_next_token_decoder_gate_min"], 1.0)
        self.assertTrue(model["answer_state_loop_free_transformer_latent_enabled"])
        self.assertTrue(
            model["answer_state_loop_free_transformer_posterior_train_enabled"]
        )
        self.assertEqual(model["answer_state_loop_free_transformer_gate_min"], 1.0)

    def test_training_script_exposes_free_latent_losses(self):
        text = Path("scripts/196_train_pure_recursive_depth_supervised.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("--answer-free-transformer-latent-kl-weight", text)
        self.assertIn("--answer-free-transformer-latent-final-contrast-weight", text)
        self.assertIn("answer_state_loop_free_transformer_latent_kl", text)
        self.assertIn("answer_free_transformer_latent_kl_loss", text)
        self.assertIn("answer_free_transformer_latent_final_contrast", text)
        self.assertIn("disable_answer_state_loop_free_transformer_latent=True", text)

        wrapper = Path("scripts/332_run_free_transformer_latent_smoke.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('DEPTH_STEPS="${DEPTH_STEPS:-8}"', wrapper)
        self.assertIn('MAX_TARGET_TOKENS="${MAX_TARGET_TOKENS:-8}"', wrapper)
        self.assertIn('LATER_TOKEN_WEIGHT="${LATER_TOKEN_WEIGHT:-1.0}"', wrapper)
        self.assertIn('RUN_GATE="${RUN_GATE:-1}"', wrapper)
        self.assertIn('SAVE_EVERY="${SAVE_EVERY:-0}"', wrapper)
        self.assertIn('--depth-steps "$DEPTH_STEPS"', wrapper)
        self.assertIn('--save-every "$SAVE_EVERY"', wrapper)
        self.assertIn('--causal-prefix-later-token-weight "$LATER_TOKEN_WEIGHT"', wrapper)
        self.assertIn("--causal-prefix-skip-leading-whitespace-targets", wrapper)
        self.assertIn('if [[ "$RUN_GATE" == "0" ]]', wrapper)

    def test_eval_and_gate_have_free_latent_off_ablation(self):
        eval_text = Path("scripts/192_eval_raw_intelligence.py").read_text(
            encoding="utf-8"
        )
        gate_text = Path("scripts/330_run_mixed_noncopy_lm_gate.py").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "qtrm_core_steps_(\\d+)_answer_free_transformer_latent_off_no_evidence",
            eval_text,
        )
        self.assertIn("disable_answer_state_loop_free_transformer_latent", eval_text)
        self.assertIn("answer_free_transformer_latent_off", gate_text)
        self.assertIn("--min-answer-free-transformer-latent-drop", gate_text)


if __name__ == "__main__":
    unittest.main()
