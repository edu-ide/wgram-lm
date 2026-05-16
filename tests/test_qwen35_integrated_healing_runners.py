from __future__ import annotations

import unittest
from pathlib import Path


class Qwen35IntegratedHealingRunnerTests(unittest.TestCase):
    def test_healing_tune_runner_uses_language_kl_and_partial_layer23(self):
        script = Path("scripts/388_run_qwen35_integrated_healing_tune.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("--mandatory-core", script)
        self.assertIn("--unfreeze-qwen-layer-indices", script)
        self.assertIn("23", script)
        self.assertIn("--language-kl-weight", script)
        self.assertIn("bfloat16", script)

    def test_healing_language_gate_uses_mandatory_core_generation(self):
        script = Path("scripts/389_run_qwen35_integrated_healing_language_gate.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("--mandatory-core", script)
        self.assertIn("--max-new-tokens", script)
        self.assertIn("--core-adapter-dim", script)
        self.assertIn("--residual-scale", script)


if __name__ == "__main__":
    unittest.main()
