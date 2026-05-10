from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "305_eval_donor_renderer_baseline.py"
    spec = importlib.util.spec_from_file_location("donor_renderer_baseline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DonorRendererBaselineTests(unittest.TestCase):
    def test_prompt_styles(self) -> None:
        module = _load_module()
        case = {"prompt": "P", "question": "Q"}
        self.assertEqual(module.build_prompt(case, "raw"), "P")
        self.assertIn("Q", module.build_prompt(case, "minimal"))
        self.assertIn("Final answer:", module.build_prompt(case, "numeric_strict"))


if __name__ == "__main__":
    unittest.main()
