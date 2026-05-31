from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "566_eval_blt_candidate_rerank_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("blt_candidate_rerank_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BLTCandidateRerankGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_micro_math_solver_covers_verified_templates(self) -> None:
        cases = {
            "Solve carefully. What is 501 - 338?": "163",
            "Solve carefully. What is 797 + 959?": "1756",
            "Solve carefully. What is 29 times 3?": "87",
            "Solve for x: 6x + 46 = -8.": "-9",
            "Find the least common multiple of 30 and 23.": "690",
            "Find the greatest common divisor of 78 and 39.": "39",
            "Compute binom(12,1).": "12",
            "A box has 8 bags with 7 marbles each, plus 5 extra marbles. How many marbles are there?": "61",
            "Solve carefully. What is 1/3 + 2/7?": "13/21",
        }

        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(self.module.solve_micro_math_instruction(prompt), expected)

    def test_micro_math_solver_ignores_unknown_prompt(self) -> None:
        self.assertIsNone(self.module.solve_micro_math_instruction("Explain gravity in one sentence."))

    def test_self_consistency_selects_first_majority_candidate(self) -> None:
        answer_interface = self.module.load_answer_interface_module()

        index, value = self.module.choose_self_consistency(
            ["Final answer: 4", "Final answer: 7", "answer = 7."],
            answer_interface,
        )

        self.assertEqual(index, 1)
        self.assertEqual(value, "7")

    def test_normalized_answer_handles_box_end_and_fraction(self) -> None:
        answer_interface = self.module.load_answer_interface_module()

        self.assertEqual(
            self.module.normalized_answer(answer_interface, "Final answer: \\frac{13}{21}<|box_end|>"),
            "13/21",
        )


if __name__ == "__main__":
    unittest.main()
