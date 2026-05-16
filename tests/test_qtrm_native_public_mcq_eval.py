from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch


def _load_script():
    path = Path("scripts/384_eval_qtrm_native_public_mcq.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_public_mcq_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QTRMNativePublicMCQEvalTests(unittest.TestCase):
    def test_normalize_mcq_answer_extracts_letters(self):
        module = _load_script()

        self.assertEqual(module.normalize_mcq_answer("A"), "A")
        self.assertEqual(module.normalize_mcq_answer("(C)"), "C")
        self.assertEqual(module.normalize_mcq_answer("Answer: h"), "H")
        self.assertEqual(module.normalize_mcq_answer("not sure"), "")

    def test_extract_answer_text_rejects_prompt_echo_options(self):
        module = _load_script()
        prompt = "User: Q\nOptions:\nA. one\nB. two\n\nAnswer:\nAssistant:"
        generated = "User: Q\nOptions:\nA. one\nB. two\n\nAnswer:\nAssistant:"

        answer_text, prompt_echo = module.extract_answer_text(generated, prompt)

        self.assertEqual(answer_text, "")
        self.assertTrue(prompt_echo)
        self.assertEqual(module.normalize_mcq_answer(answer_text), "")

    def test_extract_answer_text_uses_suffix_after_assistant_marker(self):
        module = _load_script()
        prompt = "User: Q\nAssistant:"
        generated = "User: Q\nAssistant: C"

        answer_text, prompt_echo = module.extract_answer_text(generated, prompt)

        self.assertEqual(answer_text, "C")
        self.assertFalse(prompt_echo)
        self.assertEqual(module.normalize_mcq_answer(answer_text), "C")

    def test_generate_answer_scores_only_new_suffix_tokens(self):
        module = _load_script()

        class FakeTokenizer:
            eos_token_id = None
            vocab_size = 4

            def encode(self, text):
                return [0, 1, 2]

            def decode(self, token_ids):
                # If the evaluator decodes the full prompt, this would include
                # prompt-like text before the answer and be rejected as echo.
                mapping = {0: "User:", 1: "Options:", 2: "Assistant:", 3: "C"}
                return "".join(mapping[int(token_id)] for token_id in token_ids)

        class FakeModel(torch.nn.Module):
            def forward(self, x, *, think_steps=0, **kwargs):
                logits = torch.zeros((x.shape[0], x.shape[1], 4), device=x.device)
                logits[:, -1, 3] = 1.0
                return logits

        completion, answer_text, pred, prompt_echo = module.generate_answer(
            SimpleNamespace(),
            SimpleNamespace(seq_len=16),
            FakeTokenizer(),
            FakeModel(),
            torch.device("cpu"),
            prompt="User: Q\nOptions:\nA. one\nB. two\n\nAnswer:\nAssistant:",
            think_steps=4,
            max_new_chars=1,
        )

        self.assertEqual(completion, "C")
        self.assertEqual(answer_text, "C")
        self.assertEqual(pred, "C")
        self.assertFalse(prompt_echo)

    def test_load_suite_validates_public_case_schema(self):
        module = _load_script()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "suite.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "benchmark_id": "mmlu_pro",
                        "case_id": "case-1",
                        "qtrm_prompt": "User: Q\nAssistant:",
                        "answer": "B",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = module.load_suite(path)

        self.assertEqual(rows[0]["answer"], "B")

    def test_score_rows_groups_by_category(self):
        module = _load_script()

        metrics = module.score_rows(
            [
                {"category": "math", "exact": True, "pred_answer": "A"},
                {"category": "math", "exact": False, "pred_answer": "", "prompt_echo": True},
                {"category": "physics", "exact": True, "pred_answer": "D"},
            ]
        )

        self.assertEqual(metrics["hits"], 2)
        self.assertEqual(metrics["cases"], 3)
        self.assertEqual(metrics["invalid_pred_count"], 1)
        self.assertEqual(metrics["prompt_echo_count"], 1)
        self.assertEqual(metrics["by_category"]["math"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
