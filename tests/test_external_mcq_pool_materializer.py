from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import unittest


def load_module():
    path = Path("scripts/393_materialize_external_mcq_pool.py")
    spec = importlib.util.spec_from_file_location("external_mcq_pool", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ExternalMCQPoolMaterializerTests(unittest.TestCase):
    def test_normalizes_arc_style_choices(self):
        module = load_module()

        row = module.normalize_hf_mcq_row(
            {
                "id": "case-1",
                "question": "What should be recorded?",
                "choices": {"label": ["A", "B"], "text": ["Nothing", "Details"]},
                "answerKey": "B",
                "_row_idx": 7,
            },
            dataset="allenai/ai2_arc",
            config="ARC-Challenge",
            split="validation",
            category="science",
            benchmark_id="external_mcq",
        )

        assert row is not None
        self.assertEqual(row["answer"], "B")
        self.assertEqual(row["answer_index"], 1)
        self.assertIn("A. Nothing", row["qtrm_prompt"])
        self.assertIn("B. Details", row["qtrm_prompt"])

    def test_normalizes_commonsense_style_choices(self):
        module = load_module()

        row = module.normalize_hf_mcq_row(
            {
                "id": "case-2",
                "question": "A revolving door is used at a what?",
                "choices": {"label": ["A", "B", "C"], "text": ["bank", "library", "park"]},
                "answerKey": "A",
            },
            dataset="tau/commonsense_qa",
            config="default",
            split="validation",
            category="commonsense",
            benchmark_id="external_mcq",
        )

        assert row is not None
        self.assertEqual(row["options"], ["bank", "library", "park"])
        self.assertEqual(row["answer"], "A")


if __name__ == "__main__":
    unittest.main()
