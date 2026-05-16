from __future__ import annotations

import unittest
from pathlib import Path


class Qwen35IntegratedPartialUnfreezeRunnerTests(unittest.TestCase):
    def test_runner_uses_partial_unfreeze_without_full_qwen_train_flag(self):
        script = Path("scripts/387_run_qwen35_integrated_partial_unfreeze_gate.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("--core-impl qwen_layer_wrapped", script)
        self.assertIn("--mandatory-core", script)
        self.assertIn("--unfreeze-qwen-layer-indices", script)
        self.assertIn("--qwen-lr", script)
        self.assertIn("--init-checkpoint", script)
        self.assertNotIn("--train-qwen", script)


if __name__ == "__main__":
    unittest.main()
