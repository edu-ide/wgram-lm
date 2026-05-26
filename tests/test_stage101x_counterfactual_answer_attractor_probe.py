from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "600_build_stage101x_counterfactual_answer_attractor_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101x_counterfactual_answer_attractor_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101XCounterfactualAnswerAttractorProbeTests(unittest.TestCase):
    def test_rows_hide_feature_labels_and_flip_answer(self) -> None:
        module = load_module()
        rows = module.counterfactual_answer_attractor_rows()

        self.assertEqual(32, len(rows))
        for row in rows:
            self.assertNotIn("world_a_targets", row)
            self.assertNotIn("world_b_targets", row)
            self.assertIn(row["original_answer"], [" yes", " no"])
            self.assertIn(row["counterfactual_answer"], [" yes", " no"])
            self.assertNotEqual(row["original_answer"], row["counterfactual_answer"])
            self.assertIn("Real world:", row["original_prompt"])
            self.assertIn("Imagined change:", row["counterfactual_prompt"])

    def test_contract_balances_original_answer_positions(self) -> None:
        module = load_module()
        rows = module.counterfactual_answer_attractor_rows()
        contract = module.counterfactual_answer_attractor_contract(rows)

        self.assertEqual({" yes": 16, " no": 16}, contract["original_answer_counts"])
        self.assertEqual({" yes": 16, " no": 16}, contract["counterfactual_answer_counts"])
        self.assertEqual(
            {
                "source_reliability": 8,
                "evidence_relevance": 8,
                "detail_sufficiency": 8,
                "conflict_status": 8,
            },
            contract["pair_feature_counts"],
        )


if __name__ == "__main__":
    unittest.main()
