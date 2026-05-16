from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def load_module():
    path = Path("scripts/367_eval_qwen_backbone_language_gate.py")
    spec = importlib.util.spec_from_file_location("qwen_backbone_language_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QwenBackboneLanguageGateTests(unittest.TestCase):
    def test_default_prompts_cover_english_and_korean(self):
        module = load_module()

        prompts = module.default_language_prompts()

        self.assertGreaterEqual(len(prompts), 8)
        self.assertTrue(any("Explain" in prompt for prompt in prompts))
        self.assertTrue(any("한국어" in prompt or "설명" in prompt for prompt in prompts))

    def test_repetition_and_unique_ratio_helpers(self):
        module = load_module()

        self.assertEqual(module.max_repeated_token_run([1, 1, 2, 2, 2, 3]), 3)
        self.assertEqual(module.max_repeated_token_run([]), 0)
        self.assertAlmostEqual(module.unique_ratio([1, 1, 2, 3]), 0.75)
        self.assertEqual(module.unique_ratio([]), 0.0)

    def test_parser_defaults_to_qwen_transition(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(
            ["--checkpoint", "/tmp/last_core.pt"]
        )

        self.assertEqual(args.core_impl, "qwen_layer_wrapped")
        self.assertEqual(args.qwen_core_layer_indices, "3")
        self.assertEqual(args.core_adapter_dim, 64)
        self.assertEqual(args.core_delta_adapter_mode, "add")
        self.assertFalse(args.mandatory_core)


if __name__ == "__main__":
    unittest.main()
