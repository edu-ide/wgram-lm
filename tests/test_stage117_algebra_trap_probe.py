import importlib.util
from pathlib import Path
import sys
import unittest


def load_module():
    path = Path("scripts/626_build_algebra_trap_preference_probe.py")
    spec = importlib.util.spec_from_file_location("stage117_algebra_trap_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage117AlgebraTrapProbeTests(unittest.TestCase):
    def test_build_rows_has_all_formats_and_excludes_heldout_answer_pairs(self):
        module = load_module()
        rows = module.build_rows(rows_per_format=3, seed=117)
        self.assertEqual(12, len(rows))
        tasks = {row["task"] for row in rows}
        self.assertEqual(
            {
                "repetitive_answer/algebra/original",
                "repetitive_answer/algebra/v2fmt",
                "repetitive_answer/algebra/numbered",
                "repetitive_answer/algebra/instruction",
            },
            tasks,
        )
        for row in rows:
            fmt = row["task"].split("/")[-1]
            correct = int(row["intelligence_answer"].strip())
            wrong = int(row["parrot_answer"].strip())
            self.assertNotEqual(correct, wrong)
            self.assertNotIn((fmt, correct, wrong), module.HELDOUT_FINALS)
            self.assertIn("a", row["prompt"])

    def test_fixed_wrong_values_can_target_specific_formats(self):
        module = load_module()
        rows = module.build_rows(
            rows_per_format=4,
            seed=118,
            formats=("original", "instruction"),
            fixed_wrong_by_format={"original": 83, "instruction": 13},
        )
        self.assertEqual(8, len(rows))
        wrong_by_task = {
            row["task"].split("/")[-1]: int(row["parrot_answer"].strip())
            for row in rows
        }
        self.assertEqual(83, wrong_by_task["original"])
        self.assertEqual(13, wrong_by_task["instruction"])


if __name__ == "__main__":
    unittest.main()
