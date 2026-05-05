from pathlib import Path
import importlib.util
import unittest


def _load_module():
    path = Path("scripts/222_infer_qtrm_primitive_transition_answer.py")
    spec = importlib.util.spec_from_file_location("qtrm_primitive_answer_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QTRMPrimitiveTransitionAnswerRuntimeTests(unittest.TestCase):
    def test_build_runtime_row_keeps_prompt_without_solver_trace(self):
        module = _load_module()

        row = module.build_runtime_row("Question: Compute ((1 + 2) * 3) - 2.\nAnswer:")

        self.assertEqual(row["prompt"], "Question: Compute ((1 + 2) * 3) - 2.\nAnswer:")
        self.assertNotIn("solver_trace", row)

    def test_runtime_report_from_operations_answers_without_gold_trace(self):
        module = _load_module()

        report = module.runtime_report_from_operations(
            {"prompt": "Question: Compute ((207 + 3) * 2) - 3.\nAnswer:"},
            ["add_operands", "multiply_sum", "subtract_offset", "hold_final"],
        )

        self.assertEqual(report["answer"], "417")
        self.assertEqual(
            report["predicted_operations"],
            ["add_operands", "multiply_sum", "subtract_offset", "hold_final"],
        )
        self.assertEqual(
            report["executed_operations"],
            ["add_operands", "multiply_sum", "subtract_offset"],
        )


if __name__ == "__main__":
    unittest.main()
