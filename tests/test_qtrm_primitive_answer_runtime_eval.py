from pathlib import Path
import importlib.util
import unittest


def _load_module():
    path = Path("scripts/223_eval_qtrm_primitive_answer_runtime.py")
    spec = importlib.util.spec_from_file_location("qtrm_primitive_answer_runtime_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QTRMPrimitiveAnswerRuntimeEvalTests(unittest.TestCase):
    def test_runtime_row_from_eval_row_removes_solver_trace(self):
        module = _load_module()

        runtime_row = module.runtime_row_from_eval_row(
            {
                "prompt": "Question?\nAnswer:",
                "chosen": "A",
                "solver_trace": [{"operation": "hold_final"}],
            }
        )

        self.assertEqual(runtime_row["prompt"], "Question?\nAnswer:")
        self.assertNotIn("solver_trace", runtime_row)
        self.assertNotIn("chosen", runtime_row)

    def test_score_answer_report_uses_chosen_or_answer_alias(self):
        module = _load_module()

        exact = module.score_answer_report(
            {"id": "case-1", "chosen": "417", "task_family": "arithmetic_chain"},
            {"answer": "417", "predicted_operations": ["add_operands"]},
        )
        miss = module.score_answer_report(
            {"id": "case-2", "answer": "TRUE", "task_family": "boolean_logic"},
            {"answer": "FALSE", "predicted_operations": ["not_q"]},
        )

        self.assertTrue(exact["answer_exact_match"])
        self.assertFalse(miss["answer_exact_match"])
        self.assertEqual(exact["target_answer"], "417")

    def test_summarize_answer_results_counts_by_family(self):
        module = _load_module()

        summary = module.summarize_answer_results(
            [
                {"task_family": "arithmetic_chain", "answer_exact_match": True},
                {"task_family": "arithmetic_chain", "answer_exact_match": False},
                {"task_family": "list_transform", "answer_exact_match": True},
            ]
        )

        self.assertEqual(summary["answer_exact"], "2/3")
        self.assertEqual(summary["by_family"]["arithmetic_chain"]["answer_exact"], "1/2")
        self.assertEqual(summary["by_family"]["list_transform"]["answer_accuracy"], 1.0)

    def test_runtime_report_from_operations_records_executor_error(self):
        module = _load_module()

        report = module.runtime_report_from_operations_safe(
            {"question": "Compute ((1 + 2) * 3) - 2."},
            ["not_q"],
        )

        self.assertEqual(report["answer"], "")
        self.assertIn("unsupported boolean question", report["error"])

    def test_arg_parser_accepts_state_constrained_operations_flag(self):
        module = _load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "config.yaml",
                "--checkpoint",
                "last.pt",
                "--data-jsonl",
                "cases.jsonl",
                "--out-json",
                "out.json",
                "--state-constrained-operations",
            ]
        )

        self.assertTrue(args.state_constrained_operations)


if __name__ == "__main__":
    unittest.main()
