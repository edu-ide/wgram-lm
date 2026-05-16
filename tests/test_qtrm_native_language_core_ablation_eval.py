from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


def _load_script():
    path = Path("scripts/374_eval_qtrm_native_language_core_ablation.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_language_core_ablation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QTRMNativeLanguageCoreAblationEvalTests(unittest.TestCase):
    def test_parser_requires_checkpoint_and_accepts_overrides(self):
        module = _load_script()

        args = module.build_arg_parser().parse_args(
            [
                "--checkpoint",
                "model.pt",
                "--device",
                "cpu",
                "--eval-depth-sweep",
                "0,1,4",
                "--eval-think-steps",
                "4",
            ]
        )

        self.assertEqual(args.checkpoint, "model.pt")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.eval_depth_sweep, "0,1,4")
        self.assertEqual(args.eval_think_steps, 4)


if __name__ == "__main__":
    unittest.main()
