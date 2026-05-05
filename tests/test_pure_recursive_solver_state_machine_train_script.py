import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "216_train_pure_recursive_solver_state_machine.py"
    )
    spec = importlib.util.spec_from_file_location("solver_state_machine_train", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PureRecursiveSolverStateMachineTrainScriptTests(unittest.TestCase):
    def test_load_trace_rows_rejects_shortcuts(self):
        module = _load_module()
        row = {
            "type": "pure_recursive_solver_trace",
            "prompt": "Question?\nAnswer:",
            "operation": "shortcut",
            "target_state_text": "A",
            "retrieval_allowed": True,
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "shortcut"):
                module.load_trace_rows(path)

    def test_build_vocab_texts_includes_prompt_input_and_target_state(self):
        module = _load_module()
        rows = [
            {
                "prompt": "Question: Compute 1 + 2.\nAnswer:",
                "depth": 1,
                "operation": "add_operands",
                "previous_state_text": "",
                "target_state_text": "3",
            }
        ]

        texts = module.vocab_texts_for_rows(rows)

        self.assertTrue(any("Question: Compute" in text for text in texts))
        self.assertIn("3", texts)

    def test_row_tensors_use_padded_input_and_shifted_target(self):
        module = _load_module()
        from qtrm_mm.agentic.solver_state_machine import CharVocab

        row = {
            "prompt": "Question?\nAnswer:",
            "depth": 1,
            "operation": "hold_final",
            "previous_state_text": "",
            "target_state_text": "42",
        }
        vocab = CharVocab.build(module.vocab_texts_for_rows([row]))

        tensors = module.row_tensors(
            row,
            vocab,
            max_input_len=64,
            max_target_len=4,
            previous_state="",
        )

        self.assertEqual(tuple(tensors.input_ids.shape), (64,))
        self.assertEqual(tuple(tensors.attention_mask.shape), (64,))
        self.assertEqual(tensors.decoder_input_ids[0].item(), vocab.bos_id)
        self.assertEqual(tensors.labels.tolist()[-1], vocab.pad_id)

    def test_parser_accepts_probe_training_args(self):
        module = _load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--train-jsonl",
                "train.jsonl",
                "--eval-jsonl",
                "eval.jsonl",
                "--out-dir",
                "runs/state-machine",
                "--steps",
                "12",
                "--max-input-len",
                "96",
                "--max-target-len",
                "8",
            ]
        )

        self.assertEqual(args.train_jsonl, "train.jsonl")
        self.assertEqual(args.steps, 12)
        self.assertEqual(args.max_target_len, 8)


if __name__ == "__main__":
    unittest.main()
