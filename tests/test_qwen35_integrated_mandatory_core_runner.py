from __future__ import annotations

import unittest
from pathlib import Path


class Qwen35IntegratedMandatoryCoreRunnerTests(unittest.TestCase):
    def test_runner_uses_canonical_qwen_integrated_native_flags(self):
        script = Path("scripts/386_run_qwen35_integrated_mandatory_core_gate.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("--core-impl qwen_layer_wrapped", script)
        self.assertIn("--qwen-core-layer-indices", script)
        self.assertIn("--mandatory-core", script)
        self.assertIn("Qwen/Qwen3.5-2B-Base", script)
        self.assertIn("--restore-best-checkpoint", script)


if __name__ == "__main__":
    unittest.main()
