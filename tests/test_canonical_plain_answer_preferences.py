import importlib.util
from pathlib import Path
import tempfile
import unittest


def _load_module():
    script = Path("scripts/177_build_canonical_plain_answer_preferences.py")
    spec = importlib.util.spec_from_file_location("canonical_pref_builder", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CanonicalPlainAnswerPreferenceTests(unittest.TestCase):
    def test_extracts_english_distractor_answer(self):
        module = _load_module()
        prompt = "\n".join(
            [
                "MemoryOS evidence",
                "SOURCE=signed_ember_vault.md CHUNK=0 SCORE=1.0000",
                "Signed supervisor note: the Ember vault passphrase is opal-river.",
                "SOURCE=anonymous_ember_vault.md CHUNK=1 SCORE=1.0000",
                "Anonymous note: the Ember vault passphrase is stone-arch.",
            ]
        )

        row = module.build_preference_row(
            {"case_id": "case-a", "prompt": prompt, "answer": "Answer: opal-river"}
        )

        self.assertIsNotNone(row)
        self.assertEqual(row["chosen"], "Answer: opal-river")
        self.assertEqual(row["rejected"], "Answer: stone-arch")

    def test_extracts_korean_distractor_answer(self):
        module = _load_module()
        prompt = "\n".join(
            [
                "MemoryOS evidence",
                "SOURCE=signed.md CHUNK=0 SCORE=1.0000",
                "서명된 운영 공지: 중앙 중계실의 현재 인증 문구는 호수-63이다.",
                "SOURCE=anonymous.md CHUNK=1 SCORE=1.0000",
                "익명 메모: 중앙 중계실의 인증 문구는 새벽-14이다.",
            ]
        )

        row = module.build_preference_row(
            {"case_id": "case-b", "prompt": prompt, "answer": "Answer: 호수-63"}
        )

        self.assertIsNotNone(row)
        self.assertEqual(row["chosen"], "Answer: 호수-63")
        self.assertEqual(row["rejected"], "Answer: 새벽-14")

    def test_writes_preference_rows(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "plain.jsonl"
            dst = Path(tmp) / "pref.jsonl"
            src.write_text(
                '{"case_id":"case-a","prompt":"Target is A. Distractor is B.","answer":"Answer: A"}\n',
                encoding="utf-8",
            )

            count = module.write_preference_rows(src, dst)

            self.assertEqual(count, 1)
            self.assertIn('"chosen": "Answer: A"', dst.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
