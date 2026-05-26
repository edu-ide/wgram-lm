from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


def _load_script():
    path = Path("scripts/539_eval_prefixlm_generation_gate.py")
    spec = importlib.util.spec_from_file_location("prefixlm_generation_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class _FakeTokenizer:
    def __init__(self, mapping: dict[int, str]):
        self.mapping = mapping

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self.mapping[int(token_id)] for token_id in token_ids)


class _FakeDataset:
    def __init__(self):
        self.row_indices = np.array([0, 1, 2], dtype=np.int64)
        self.inst_start = np.array([0, 3, 6], dtype=np.int64)
        self.inst_len = np.array([3, 3, 3], dtype=np.int64)
        self.resp_len = np.array([1, 1, 1], dtype=np.int64)
        self.shifted_lengths = self.inst_len[self.row_indices] + self.resp_len[self.row_indices] - 1
        self.tokens = np.array([1, 10, 2, 1, 11, 2, 1, 10, 2], dtype=np.int64)

    def _slice_tokens(self, start: int, length: int):
        return self.tokens[int(start) : int(start) + int(length)]

    def __len__(self):
        return int(self.row_indices.shape[0])


class PrefixLMGenerationGateTests(unittest.TestCase):
    def test_condition_from_instruction_text_uses_condition_mapping(self):
        module = _load_script()
        tokenizer_info = {
            "boq": "<|im_start|>",
            "eoq": "<|im_end|>",
            "condition_mapping": {"direct": "<D>", "cot": "<C>"},
        }

        self.assertEqual(
            module.condition_from_instruction_text("<|im_start|><D>problem<|im_end|>", tokenizer_info),
            "direct",
        )
        self.assertEqual(
            module.condition_from_instruction_text("<|im_start|><C>problem<|im_end|>", tokenizer_info),
            "cot",
        )

    def test_filter_dataset_by_condition_keeps_matching_rows(self):
        module = _load_script()
        dataset = _FakeDataset()
        tokenizer = _FakeTokenizer({1: "<|im_start|>", 2: "<|im_end|>", 10: "<D>", 11: "<C>"})
        tokenizer_info = {
            "boq": "<|im_start|>",
            "eoq": "<|im_end|>",
            "condition_mapping": {"direct": "<D>", "cot": "<C>"},
        }

        summary = module.filter_dataset_by_condition(
            dataset,
            tokenizer=tokenizer,
            tokenizer_info=tokenizer_info,
            condition="direct",
        )

        self.assertEqual(dataset.row_indices.tolist(), [0, 2])
        self.assertEqual(summary["condition"], "direct")
        self.assertEqual(summary["rows_before"], 3)
        self.assertEqual(summary["rows_after"], 2)


if __name__ == "__main__":
    unittest.main()
