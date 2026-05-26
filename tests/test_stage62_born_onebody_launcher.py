from pathlib import Path
import subprocess
import unittest


class Stage62BornOneBodyLauncherTests(unittest.TestCase):
    def test_launcher_preserves_born_onebody_pvgram_contract(self):
        launcher = Path("scripts/launch_stage62a_local_born_onebody_pvgram.sh")

        self.assertTrue(launcher.exists(), "Stage62A born-one-body launcher is missing")
        text = launcher.read_text(encoding="utf-8")

        self.assertIn("setsid", text)
        self.assertIn('PYTHON="${PYTHON:-.venv/bin/python}"', text)
        self.assertIn("337_train_qtrm_native_mixed_text_reasoning_probe.py", text)
        self.assertIn("--backbone trm_qwen35_3to1", text)
        self.assertIn("DELTA_BACKEND", text)
        self.assertIn("official_gated_delta2", text)
        self.assertIn('--delta-backend "${DELTA_BACKEND}"', text)
        self.assertIn("STRICT_BACKENDS", text)
        self.assertIn("--strict-backends", text)
        self.assertIn("--eval-gram-trajectory-search", text)
        self.assertIn("--gram-lprm-loss-weight", text)
        self.assertIn("--gram-lprm-ranking-loss-weight", text)
        self.assertIn("--gram-candidate-topk-per-trajectory", text)
        self.assertIn("GRAM_CANDIDATE_SELECTOR", text)
        self.assertIn("--gram-candidate-selector", text)
        self.assertIn("--gram-attractor-iterations", text)
        self.assertIn("--gram-attractor-step-scale", text)
        self.assertIn("GRAM_LCV_LATENT_DIM", text)
        self.assertIn("GRAM_LCV_TEMPERATURE", text)
        self.assertIn("--gram-lcv-latent-dim", text)
        self.assertIn("--gram-lcv-temperature", text)
        self.assertIn("GRAM_TRACE_MAX_LEN", text)
        self.assertIn("--gram-trace-max-len", text)
        self.assertIn("GRAM_TRACE_CONSISTENCY_WEIGHT", text)
        self.assertIn("--gram-trace-consistency-weight", text)
        self.assertIn("--gram-candidate-score-mode", text)
        self.assertIn("GRAM_CANDIDATE_SCORE_TRAIN_BODY", text)
        self.assertIn("--gram-candidate-score-train-body", text)
        self.assertIn("ACCEPTED_DECISION", text)
        self.assertIn("accepted_stage62a_born_onebody_pvgram", text)
        self.assertIn('--accepted-decision "${ACCEPTED_DECISION}"', text)
        self.assertNotIn("mamba3", text.lower())

    def test_launcher_shell_syntax_is_valid(self):
        result = subprocess.run(
            ["bash", "-n", "scripts/launch_stage62a_local_born_onebody_pvgram.sh"],
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_stage66_launcher_starts_hrmtext_style_pretraining(self):
        launcher = Path("scripts/launch_stage66_local_hrmtext_pvgram_pretrain.sh")

        self.assertTrue(launcher.exists(), "Stage66 HRM-Text pretrain launcher is missing")
        text = launcher.read_text(encoding="utf-8")

        self.assertIn("Stage66 HRM-Text-style", text)
        self.assertIn("TARGET_LEVEL", text)
        self.assertIn("accepted_stage66_hrmtext_pvgram_pretrain", text)
        self.assertIn("GRAM_CANDIDATE_SELECTOR", text)
        self.assertIn("lprm_head", text)
        self.assertIn("GRAM_CANDIDATE_SCORE_MODE", text)
        self.assertIn("candidate_forward", text)
        self.assertIn("GRAM_TRACE_CONSISTENCY_WEIGHT", text)
        self.assertIn("TRITON_PTXAS_PATH", text)
        self.assertIn("REQUIRED_TRITON_PTXAS_PATH", text)
        self.assertIn("missing required ptxas contract", text)
        self.assertNotIn('REQUIRED_TRITON_PTXAS_PATH="${REQUIRED_TRITON_PTXAS_PATH:-/usr/local/cuda-13.2/bin/ptxas}"', text)
        self.assertIn("LEARNING_EFFICIENCY_FACTOR", text)
        self.assertIn("HRM_TEXT_BASELINE_REPORT", text)
        self.assertIn("532_compare_learning_efficiency_claim.py", text)
        self.assertIn("launch_stage62a_local_born_onebody_pvgram.sh", text)
        self.assertNotIn("mamba3", text.lower())

        result = subprocess.run(
            ["bash", "-n", str(launcher)],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
