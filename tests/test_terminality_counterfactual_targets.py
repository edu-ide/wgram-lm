import importlib.util
from pathlib import Path
import tempfile
import unittest


def load_script():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "233_build_terminality_counterfactual_targets.py"
    )
    spec = importlib.util.spec_from_file_location("terminality_counterfactuals", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TerminalityCounterfactualTargetTests(unittest.TestCase):
    def test_arithmetic_terminal_depth2_uses_terminal_compose(self):
        module = load_script()
        row = {
            "id": "arith-chain-1",
            "task_family": "arithmetic_chain",
            "question": "Evaluate the expression ((7 + 3) * 2) - 3.",
            "solver_trace": [
                {"depth": 1, "state_text": "10"},
                {"depth": 2, "state_text": "20"},
                {"depth": 4, "state_text": "17"},
                {"depth": 8, "state_text": "17"},
            ],
        }

        out = module.arithmetic_terminal_depth2(row)

        self.assertEqual(out["answer_aliases"], ["20"])
        self.assertEqual(out["transition_state_codes"]["2"], 1)
        self.assertEqual(out["transition_finality_targets"]["2"], 1)
        self.assertNotIn("- 3", out["question"])
        self.assertEqual(out["task_family"], "arithmetic_chain")

    def test_symbolic_nonterminal_depth3_uses_nonterminal_compose(self):
        module = load_script()
        row = {
            "id": "symbolic-binding-1",
            "task_family": "symbolic_binding",
            "question": (
                "Mapping facts: A maps to red; red maps to blue; blue maps to C. "
                "Starting at A, apply the mapping two times."
            ),
        }

        out = module.symbolic_nonterminal_depth3(row)

        self.assertEqual(out["answer_aliases"], ["C"])
        self.assertEqual(out["transition_state_codes"]["2"], 2)
        self.assertEqual(out["transition_state_codes"]["4"], 3)
        self.assertEqual(out["transition_finality_targets"]["2"], 0)
        self.assertIn("three times", out["question"])

    def test_build_file_keeps_base_and_adds_two_counterfactuals(self):
        module = load_script()
        rows = [
            {
                "id": "arith-chain-1",
                "task_family": "arithmetic_chain",
                "question": "Compute ((7 + 3) * 2) - 3.",
                "solver_trace": [
                    {"depth": 1, "state_text": "10"},
                    {"depth": 2, "state_text": "20"},
                    {"depth": 4, "state_text": "17"},
                    {"depth": 8, "state_text": "17"},
                ],
            },
            {
                "id": "symbolic-binding-1",
                "task_family": "symbolic_binding",
                "question": "A maps to red; red maps to blue; blue maps to C.",
            },
            {
                "id": "boolean-1",
                "task_family": "boolean_logic",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inp = tmp_path / "in.jsonl"
            out = tmp_path / "out.jsonl"
            inp.write_text(
                "\n".join(__import__("json").dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            summary = module.build_terminality_counterfactual_file(
                base_train_jsonl=inp,
                output_jsonl=out,
            )

            self.assertEqual(summary["base_rows"], 3)
            self.assertEqual(summary["added_rows"], 2)
            self.assertEqual(summary["total_rows"], 5)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
