from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "594_eval_stage101w7_siamese_counterfactual_energy.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101w7_siamese_energy", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101W7SiameseCounterfactualEnergyEvalTests(unittest.TestCase):
    def test_world_prompt_scores_one_world_without_ab_choice(self) -> None:
        module = load_module()
        prompt = module.world_answerability_prompt(
            {
                "source_claim": "platform is 4.",
                "world_a": "rumor says 4.",
                "world_b": "official board says 4.",
            },
            world_key="world_b",
        )

        self.assertIn("Claim: platform is 4.", prompt)
        self.assertIn("World: official board says 4.", prompt)
        self.assertIn("Can answer now?", prompt)
        self.assertNotIn("World A:", prompt)
        self.assertNotIn("World B:", prompt)
        self.assertNotIn("A or B", prompt)

    def test_pair_rows_use_energy_gap_and_opposite_blocked_world(self) -> None:
        module = load_module()
        pair = {
            "id": "pair0_answerable_world",
            "task": "stage101w6_answerable_world_icl",
            "twin_pair_id": "pair0",
            "source": "stage101w6_counterfactual_twin_probe",
            "source_claim": "platform is 4.",
            "world_a": "rumor says 4.",
            "world_b": "official board says 4.",
            "answerable_world": " B",
            "blocked_world": " A",
        }

        rows = module.rows_from_pair_energy(pair, think_steps=16, energy_a=-0.5, energy_b=1.25)

        self.assertEqual(2, len(rows))
        answerable = rows[0]
        blocked = rows[1]
        self.assertEqual("stage101w7_siamese_answerable_world", answerable["task"])
        self.assertEqual(" B", answerable["predicted_answer"])
        self.assertTrue(answerable["correct"])
        self.assertAlmostEqual(1.75, answerable["normalized_margin"])
        self.assertEqual("stage101w7_siamese_blocked_world", blocked["task"])
        self.assertEqual(" A", blocked["predicted_answer"])
        self.assertTrue(blocked["correct"])
        self.assertAlmostEqual(1.75, blocked["normalized_margin"])

    def test_unique_answerable_pairs_deduplicates_probe_rows(self) -> None:
        module = load_module()
        rows = [
            {"twin_pair_id": "p0", "stage101w6_chain_step": "answerable_world"},
            {"twin_pair_id": "p0", "stage101w6_chain_step": "blocked_world"},
            {"twin_pair_id": "p1", "stage101w6_chain_step": "answerable_world"},
        ]

        pairs = module.unique_answerable_pairs(rows)

        self.assertEqual(["p0", "p1"], [row["twin_pair_id"] for row in pairs])


if __name__ == "__main__":
    unittest.main()
