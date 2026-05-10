from pathlib import Path
import importlib.util
import unittest


def load_module():
    path = Path("scripts/320_train_prompt_source_position_binder_probe.py")
    spec = importlib.util.spec_from_file_location("prompt_source_position_binder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PromptSourcePositionBinderProbeTests(unittest.TestCase):
    def test_source_position_targets_encode_even_positions(self):
        module = load_module()

        row = {
            "input_list": [14, 31, 10, 24, 27],
            "depth_targets": {"1": "14,10,24", "2": "28,20,48"},
        }

        self.assertEqual(
            module.source_position_targets(row, max_slots=4, position_vocab_size=8),
            (1, 3, 4, 0),
        )

    def test_source_position_targets_can_use_doubled_depth(self):
        module = load_module()

        row = {
            "input_list": [14, 31, 10, 24, 27],
            "depth_targets": {"1": "14,10,24", "2": "28,20,48"},
        }

        self.assertEqual(
            module.source_position_targets(
                row,
                max_slots=4,
                position_vocab_size=8,
                target_depth=2,
            ),
            (1, 3, 4, 0),
        )

    def test_source_position_targets_accept_singleton_scalar_list_result(self):
        module = load_module()

        row = {
            "task_family": "list_transform",
            "input_list": [27, 13, 29, 1, 2],
            "depth_targets": {"1": "2", "2": "4"},
        }

        self.assertEqual(
            module.source_position_targets(row, max_slots=4, position_vocab_size=8),
            (5, 0, 0, 0),
        )

    def test_binder_returns_slot_logits(self):
        import torch

        module = load_module()
        model = module.PromptSourcePositionBinder(
            input_dim=8,
            hidden_dim=16,
            max_slots=4,
            position_vocab_size=8,
        )

        logits = model(
            torch.randn(2, 5, 8),
            torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]),
        )

        self.assertEqual(logits.shape, (2, 4, 8))

    def test_numeric_value_ids_are_one_based_and_padded(self):
        module = load_module()

        ids, mask = module.numeric_value_ids(
            {"input_list": [14, 31, 10]},
            max_list_len=5,
            value_vocab_size=64,
        )

        self.assertEqual(ids, (15, 32, 11, 0, 0))
        self.assertEqual(mask, (1, 1, 1, 0, 0))

    def test_token_numeric_value_ids_marks_source_number_spans(self):
        module = load_module()

        row = {
            "prompt": "Question: From the list [14, 31], return evens.",
            "input_list": [14, 31],
        }
        offsets = [
            (0, 9),
            (10, 14),
            (15, 18),
            (19, 23),
            (24, 25),
            (25, 27),
            (27, 29),
            (29, 31),
            (31, 32),
        ]

        ids = module.token_numeric_value_ids(
            row,
            offsets=offsets,
            value_vocab_size=64,
        )

        self.assertEqual(ids, (0, 0, 0, 0, 0, 15, 0, 32, 0))

    def test_token_numeric_source_slots_collapse_token_pieces(self):
        module = load_module()

        row = {
            "prompt": "Question: From the list [14, 31], return evens.",
            "input_list": [14, 31],
        }
        offsets = [
            (0, 9),
            (10, 14),
            (15, 18),
            (19, 23),
            (24, 25),
            (25, 26),
            (26, 27),
            (27, 29),
            (29, 31),
            (31, 32),
        ]

        ids, mask = module.token_numeric_source_slot_ids(
            row,
            offsets=offsets,
            max_list_len=4,
            value_vocab_size=64,
        )

        self.assertEqual(ids, (15, 32, 0, 0))
        self.assertEqual(mask, (1, 1, 0, 0))

    def test_token_numeric_source_slot_token_ids_keep_tokenizer_id_coordinate(self):
        module = load_module()

        row = {
            "prompt": "Question: From the list [14, 31], return evens.",
            "input_list": [14, 31],
        }
        offsets = [
            (0, 9),
            (10, 14),
            (15, 18),
            (19, 23),
            (24, 25),
            (25, 26),
            (26, 27),
            (27, 29),
            (29, 31),
            (31, 32),
        ]
        input_ids = [101, 102, 103, 104, 105, 206, 207, 308, 309, 110]

        token_ids = module.token_numeric_source_slot_token_ids(
            row,
            offsets=offsets,
            input_ids=input_ids,
            max_list_len=4,
            value_vocab_size=64,
        )

        self.assertEqual(token_ids, (206, 309, 0, 0))

    def test_token_numeric_source_slot_token_spans_keep_all_token_pieces(self):
        module = load_module()

        row = {
            "prompt": "Question: From the list [14, 31], return evens.",
            "input_list": [14, 31],
        }
        offsets = [
            (0, 9),
            (10, 14),
            (15, 18),
            (19, 23),
            (24, 25),
            (25, 26),
            (26, 27),
            (29, 30),
            (30, 31),
            (31, 32),
        ]
        input_ids = [101, 102, 103, 104, 105, 206, 207, 308, 309, 110]

        span_ids, span_mask = module.token_numeric_source_slot_token_spans(
            row,
            offsets=offsets,
            input_ids=input_ids,
            max_list_len=4,
            max_token_pieces=3,
            value_vocab_size=64,
        )

        self.assertEqual(span_ids, ((206, 207, 0), (308, 309, 0), (0, 0, 0), (0, 0, 0)))
        self.assertEqual(span_mask, ((1, 1, 0), (1, 1, 0), (0, 0, 0), (0, 0, 0)))

    def test_parser_accepts_token_plus_numeric_input_source(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--train-jsonl",
                "train.jsonl",
                "--eval-jsonl",
                "eval.jsonl",
                "--out-dir",
                "out",
                "--input-source",
                "token_plus_numeric_value",
            ]
        )

        self.assertEqual(args.input_source, "token_plus_numeric_value")

    def test_parser_accepts_token_numeric_source_slots_input_source(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--train-jsonl",
                "train.jsonl",
                "--eval-jsonl",
                "eval.jsonl",
                "--out-dir",
                "out",
                "--input-source",
                "token_numeric_source_slots",
            ]
        )

        self.assertEqual(args.input_source, "token_numeric_source_slots")


if __name__ == "__main__":
    unittest.main()
