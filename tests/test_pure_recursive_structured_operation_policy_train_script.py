import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "218_train_pure_recursive_structured_operation_policy.py"
    )
    spec = importlib.util.spec_from_file_location("structured_operation_policy_train", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PureRecursiveStructuredOperationPolicyTrainScriptTests(unittest.TestCase):
    def test_load_trace_rows_rejects_shortcuts(self):
        module = _load_module()
        row = {
            "type": "pure_recursive_solver_trace",
            "task_family": "arithmetic_chain",
            "depth": 1,
            "trace_index": 0,
            "operation": "add_operands",
            "memoryos_allowed": True,
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "shortcut"):
                module.load_trace_rows(path)

    def test_row_tensors_encode_structured_metadata(self):
        module = _load_module()
        from qtrm_mm.agentic.solver_state_machine import OperationVocab

        row = {
            "task_family": "arithmetic_chain",
            "depth": 2,
            "trace_index": 1,
            "operation": "multiply_sum",
        }
        family_vocab = OperationVocab.build(["arithmetic_chain"])
        trace_vocab = OperationVocab.build(["1"])
        depth_vocab = OperationVocab.build(["2"])
        operation_vocab = OperationVocab.build(["multiply_sum"])

        tensors = module.row_tensors(
            row,
            family_vocab,
            trace_vocab,
            depth_vocab,
            operation_vocab,
            device="cpu",
        )

        self.assertEqual(tensors.family_id.item(), 0)
        self.assertEqual(tensors.trace_index_id.item(), 0)
        self.assertEqual(tensors.depth_id.item(), 0)
        self.assertEqual(tensors.label.item(), 0)

    def test_parser_accepts_structured_policy_args(self):
        module = _load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--train-jsonl",
                "train.jsonl",
                "--eval-jsonl",
                "eval.jsonl",
                "--out-dir",
                "local_eval/structured-op",
                "--steps",
                "8",
            ]
        )

        self.assertEqual(args.steps, 8)


if __name__ == "__main__":
    unittest.main()
