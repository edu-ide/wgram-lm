from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "565_eval_blt_generation_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("blt_generation_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _StepModel(torch.nn.Module):
    def __init__(self, next_ids: list[int], vocab_size: int = 512) -> None:
        super().__init__()
        self.next_ids = [int(value) for value in next_ids]
        self.vocab_size = int(vocab_size)
        self.calls = 0

    def forward_logits_and_decoder_hidden(self, input_ids, attention_mask, *, think_steps: int):
        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, self.vocab_size), -10.0, device=input_ids.device)
        index = min(self.calls, len(self.next_ids) - 1)
        self.calls += 1
        logits[:, -1, self.next_ids[index]] = 10.0
        return logits, torch.zeros((batch, seq_len, 4), device=input_ids.device)


class _ToyDataset:
    def __init__(self) -> None:
        self.row_indices = np.array([0], dtype=np.int64)
        self.inst_start = np.array([0], dtype=np.int64)
        self.inst_len = np.array([2], dtype=np.int64)
        self.resp_start = np.array([2], dtype=np.int64)
        self.resp_len = np.array([2], dtype=np.int64)
        self.tokens = np.array([2 + ord("Q"), 2 + ord("?"), 2 + ord("A"), 1], dtype=np.int64)

    def _slice_tokens(self, start: int, length: int):
        return self.tokens[int(start) : int(start) + int(length)]

    def __len__(self) -> int:
        return 1


class BLTGenerationGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_script_reuses_blt_checkpoint_loader_from_depth_probe(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("560_eval_blt_depth_residual_probe.py", source)
        self.assertIn("load_checkpoint_model", source)

    def test_byte_free_decoder_turns_shifted_byte_ids_into_text(self) -> None:
        text = self.module.decode_ids(
            None,
            [2 + ord("h"), 2 + ord("i"), 1],
            {"kind": "tokenizer_free_utf8_byte_shifted", "byte_offset": 2, "eos_token_id": 1},
        )

        self.assertEqual(text, "hi<eos>")

    def test_generate_one_stops_on_eos(self) -> None:
        generated = self.module.generate_one(
            model=_StepModel([5, 1]),
            prefix_ids=[2, 3],
            eoa_id=1,
            device=torch.device("cpu"),
            think_steps=2,
            seq_len=8,
            max_new_tokens=4,
        )

        self.assertEqual(generated, [5, 1])

    def test_generation_stats_reports_samples_and_repetition_fields(self) -> None:
        report = self.module.generation_stats(
            model=_StepModel([2 + ord("A"), 1]),
            dataset=_ToyDataset(),
            tokenizer=None,
            tokenizer_info={"kind": "tokenizer_free_utf8_byte_shifted", "byte_offset": 2, "eos_token_id": 1},
            eoa_id=1,
            device=torch.device("cpu"),
            think_steps=2,
            seq_len=8,
            max_rows=1,
            max_new_tokens=4,
        )

        self.assertEqual(report["rows"], 1)
        self.assertEqual(report["exact"], 1)
        self.assertEqual(report["exact_fraction"], 1.0)
        self.assertEqual(report["repeated_token_loop_fraction"], 0.0)
        self.assertEqual(report["samples"][0]["generated"], "A<eos>")

    def test_summarize_response_token_stats_separates_first_token_from_continuation_and_eos(self) -> None:
        labels = torch.tensor(
            [
                [self.module.IGNORE_LABEL_ID, 34, 67, 1],
                [self.module.IGNORE_LABEL_ID, 34, 68, 1],
            ],
            dtype=torch.long,
        )
        start_mask = torch.tensor(
            [
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=torch.bool,
        )
        logits = torch.full((2, 4, 128), -10.0)
        logits[:, 1, 34] = 10.0
        logits[0, 2, 67] = 10.0
        logits[1, 2, 69] = 10.0
        logits[:, 3, 2] = 10.0

        stats = self.module.summarize_response_token_logits(
            logits=logits,
            labels=labels,
            response_start_mask=start_mask,
            eoa_id=1,
            tokenizer=None,
            tokenizer_info={"kind": "tokenizer_free_utf8_byte_shifted", "byte_offset": 2, "eos_token_id": 1},
        )

        self.assertEqual(stats["positions"], 6)
        self.assertAlmostEqual(stats["accuracy"], 0.5)
        self.assertEqual(stats["continuation_positions"], 4)
        self.assertAlmostEqual(stats["continuation_accuracy"], 0.25)
        self.assertEqual(stats["eos_targets"], 2)
        self.assertAlmostEqual(stats["eos_top1_accuracy"], 0.0)
        self.assertEqual(stats["common_targets"][0]["decoded"], " ")


if __name__ == "__main__":
    unittest.main()
