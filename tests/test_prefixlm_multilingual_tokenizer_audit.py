from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "543_audit_prefixlm_multilingual_tokenizer.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prefixlm_multilingual_tokenizer_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> FakeEncoding:
        del add_special_tokens
        return FakeEncoding(list(range(len(text.split()))))


class PrefixLMMultilingualTokenizerAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_nonspace_char_count_ignores_spaces_and_newlines(self) -> None:
        self.assertEqual(self.module.nonspace_char_count("가 나\n다"), 3)

    def test_compute_case_stats_measures_token_fertility(self) -> None:
        case = {
            "case_id": "ko_simple",
            "language": "ko",
            "family": "korean_qa",
            "instruction": "서울 수도",
        }

        stats = self.module.compute_case_stats(case, FakeTokenizer())

        self.assertEqual(stats["case_id"], "ko_simple")
        self.assertEqual(stats["token_count"], 2)
        self.assertEqual(stats["nonspace_chars"], 4)
        self.assertEqual(stats["tokens_per_nonspace_char"], 0.5)

    def test_summarize_by_language_reports_mean_and_p95(self) -> None:
        summary = self.module.summarize_by(
            [
                {"language": "ko", "tokens_per_nonspace_char": 1.0},
                {"language": "ko", "tokens_per_nonspace_char": 2.0},
                {"language": "es", "tokens_per_nonspace_char": 0.5},
            ],
            "language",
        )

        self.assertEqual(summary["ko"]["cases"], 2)
        self.assertEqual(summary["ko"]["mean_tokens_per_nonspace_char"], 1.5)
        self.assertEqual(summary["ko"]["p95_tokens_per_nonspace_char"], 2.0)
        self.assertEqual(summary["es"]["mean_tokens_per_nonspace_char"], 0.5)

    def test_gate_summary_warns_on_fragmentation(self) -> None:
        gate = self.module.gate_summary(
            [
                {"language": "ko", "tokens_per_nonspace_char": 2.2},
                {"language": "en", "tokens_per_nonspace_char": 0.4},
            ],
            warn_threshold=1.5,
        )

        self.assertEqual(gate["status"], "warn")
        self.assertEqual(gate["languages_over_threshold"], ["ko"])


if __name__ == "__main__":
    unittest.main()
