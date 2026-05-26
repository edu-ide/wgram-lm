from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "544_eval_prefixlm_language_heldout_loss.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prefixlm_language_heldout_loss_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PrefixLMLanguageHeldoutLossEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_load_heldout_cases_validates_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "heldout.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "case_id": "ko_plain",
                        "family": "plain_qa",
                        "language": "ko",
                        "instruction": "하늘 색은?",
                        "response": "파란색입니다.",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            cases = self.module.load_heldout_cases(path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "ko_plain")
        self.assertEqual(cases[0].response, "파란색입니다.")

    def test_build_prefixlm_example_masks_instruction_tokens(self) -> None:
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                values = {
                    "question": [11, 12],
                    "answer": [21, 22],
                }[text]

                class Encoded:
                    ids = values

                return Encoded()

        class Helper:
            @staticmethod
            def build_instruction_ids(**kwargs):
                return [1, *kwargs["tokenizer"].encode(kwargs["instruction"]).ids, 2]

        case = self.module.HeldoutCase(
            case_id="toy",
            family="toy",
            language="en",
            instruction="question",
            response="answer",
        )

        example = self.module.build_prefixlm_example(
            helper=Helper,
            tokenizer=FakeTokenizer(),
            tokenizer_info={},
            case=case,
            condition="direct",
            eoa_id=99,
            seq_len=16,
            drop_overlength=True,
        )

        self.assertEqual(example["input_ids"], [1, 11, 12, 2, 21, 22])
        self.assertEqual(example["labels"], [-100, -100, -100, 21, 22, 99])
        self.assertEqual(example["target_tokens"], 3)

    def test_summarize_uses_token_weighted_loss(self) -> None:
        summary = self.module.summarize(
            [
                {
                    "language": "en",
                    "loss": 2.0,
                    "target_tokens": 2,
                    "correct_tokens": 1,
                },
                {
                    "language": "en",
                    "loss": 1.0,
                    "target_tokens": 6,
                    "correct_tokens": 3,
                },
            ],
            "language",
        )

        self.assertAlmostEqual(summary["en"]["loss"], 1.25)
        self.assertEqual(summary["en"]["target_tokens"], 8)
        self.assertAlmostEqual(summary["en"]["token_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
