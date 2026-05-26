from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "542_eval_prefixlm_multilingual_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prefixlm_multilingual_probe_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PrefixLMMultilingualProbeEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_strip_response_text_removes_eoa_and_special_tokens(self) -> None:
        result = self.module.strip_response_text(
            "<|object_ref_start|> 서울 <|box_end|> trailing"
        )

        self.assertEqual(result, "서울")

    def test_score_case_matches_korean_substring(self) -> None:
        scored = self.module.score_case("정답은 서울입니다 <|box_end|>", ("서울",))

        self.assertTrue(scored["hit"])
        self.assertEqual(scored["matched"], ["서울"])
        self.assertEqual(scored["cleaned_response"], "정답은 서울입니다")

    def test_score_case_is_ascii_case_insensitive(self) -> None:
        scored = self.module.score_case("The answer is CAT.", ("cat",))

        self.assertTrue(scored["hit"])

    def test_load_probe_cases_validates_expected_contains(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "probe.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "case_id": "ko_math",
                        "family": "korean_math",
                        "language": "ko",
                        "instruction": "12 더하기 7은?",
                        "expected_contains": ["19"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            cases = self.module.load_probe_cases(path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "ko_math")
        self.assertEqual(cases[0].expected_contains, ("19",))

    def test_summarize_groups_reports_accuracy(self) -> None:
        summary = self.module.summarize_groups(
            [
                {"language": "ko", "hit": True},
                {"language": "ko", "hit": False},
                {"language": "es", "hit": True},
            ],
            "language",
        )

        self.assertEqual(summary["ko"]["cases"], 2)
        self.assertEqual(summary["ko"]["hits"], 1)
        self.assertEqual(summary["ko"]["accuracy"], 0.5)
        self.assertEqual(summary["es"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
