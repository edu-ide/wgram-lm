import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "217_train_pure_recursive_operation_policy.py"
    )
    spec = importlib.util.spec_from_file_location("operation_policy_train", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PureRecursiveOperationPolicyTrainScriptTests(unittest.TestCase):
    def test_load_trace_rows_rejects_shortcuts(self):
        module = _load_module()
        row = {
            "type": "pure_recursive_solver_trace",
            "prompt": "Question?\nAnswer:",
            "operation": "hold_final",
            "target_state_text": "A",
            "memoryos_allowed": True,
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "shortcut"):
                module.load_trace_rows(path)

    def test_policy_row_tensors_shape_and_label(self):
        module = _load_module()
        from wgram_lm.agentic.solver_state_machine import CharVocab, OperationVocab

        row = {
            "question": "Compute ((7 + 3) * 2) - 3.",
            "depth": 1,
            "operation": "add_operands",
            "previous_state_text": "",
            "target_state_text": "10",
        }
        char_vocab = CharVocab.build(module.vocab_texts_for_rows([row]))
        op_vocab = OperationVocab.build(["add_operands", "hold_final"])

        tensors = module.row_tensors(
            row,
            char_vocab,
            op_vocab,
            max_input_len=64,
            previous_state="",
        )

        self.assertEqual(tuple(tensors.input_ids.shape), (64,))
        self.assertEqual(tensors.label.item(), op_vocab.encode("add_operands"))

    def test_parser_accepts_operation_policy_args(self):
        module = _load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--train-jsonl",
                "train.jsonl",
                "--eval-jsonl",
                "eval.jsonl",
                "--out-dir",
                "local_eval/op-policy",
                "--steps",
                "10",
            ]
        )

        self.assertEqual(args.train_jsonl, "train.jsonl")
        self.assertEqual(args.steps, 10)


if __name__ == "__main__":
    unittest.main()
