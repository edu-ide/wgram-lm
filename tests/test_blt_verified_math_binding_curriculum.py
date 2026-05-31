from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "582_build_blt_verified_math_binding_curriculum.py"


def load_module():
    spec = importlib.util.spec_from_file_location("blt_verified_math_binding_curriculum", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BLTVerifiedMathBindingCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_binding_response_copies_operands_and_answer(self) -> None:
        response = self.module.binding_response("Solve carefully. What is 501 - 338?")

        self.assertIn("Operands: 501, 338.", response)
        self.assertIn("Operation: subtract.", response)
        self.assertIn("Equation: 501 - 338 = 163.", response)
        self.assertTrue(response.endswith("Final answer: 163"))

    def test_binding_response_handles_linear_and_fraction(self) -> None:
        self.assertIn(
            "Final answer: -9",
            self.module.binding_response("Solve for x: 6x + 46 = -8."),
        )
        self.assertIn(
            "Final answer: 13/21",
            self.module.binding_response("Solve carefully. What is 1/3 + 2/7?"),
        )

    def test_binding_response_returns_none_for_unknown(self) -> None:
        self.assertIsNone(self.module.binding_response("Write a poem about prime numbers."))


if __name__ == "__main__":
    unittest.main()
