from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


def _load_script():
    path = Path("scripts/173_build_canonical_plain_answer_data.py")
    spec = importlib.util.spec_from_file_location("canonical_plain_answer_data", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CanonicalPlainAnswerDataTests(unittest.TestCase):
    def test_plain_answer_rows_match_eval_answer_contract(self) -> None:
        module = _load_script()
        row = module.build_canonical_plain_answer_row(
            {
                "id": "case-supported",
                "category": "direct",
                "instruction": "Use the signed source.",
                "question": "What is the archive access code?",
                "answer_aliases": ["VX-913"],
                "evidence": [
                    {
                        "source": "archive.md",
                        "chunk_id": 0,
                        "text": "The archive access code is VX-913.",
                    }
                ],
                "distractors": [
                    {
                        "source": "rumor.md",
                        "chunk_id": 1,
                        "text": "A rumor says the archive access code is PL-404.",
                    }
                ],
            },
            evidence_mode="all",
            top_k=2,
        )

        self.assertEqual(row["type"], "canonical_plain_answer")
        self.assertEqual(row["ssot_contract"], "single_visible_prompt_stream")
        self.assertEqual(row["answer"], "Answer: VX-913")
        self.assertNotIn("Verify:", row["answer"])
        self.assertNotIn("Decision:", row["answer"])
        self.assertNotIn("workspace_context", row)
        self.assertIn("Answer using only the evidence. Return only the short answer.", row["prompt"])
        self.assertIn("MemoryOS evidence", row["prompt"])
        self.assertIn("VX-913", row["prompt"])
        self.assertIn("PL-404", row["prompt"])
        self.assertEqual(row["metadata"]["answer_contract"], "plain_short_answer")
        self.assertEqual(row["metadata"]["answer_policy"], "greedy_autoregressive")

    def test_plain_answer_rows_abstain_as_unknown_for_missing_cases(self) -> None:
        module = _load_script()
        row = module.build_canonical_plain_answer_row(
            {
                "id": "case-missing",
                "category": "negative_missing",
                "question": "What is the Garnet override phrase?",
                "answer_aliases": ["redacted"],
                "expected_unknown": True,
                "evidence": [
                    {
                        "source": "signed.md",
                        "chunk_id": 0,
                        "text": "Signed notice: the Garnet override phrase is redacted.",
                    }
                ],
            }
        )

        self.assertEqual(row["answer"], "Answer: UNKNOWN")
        self.assertTrue(row["metadata"]["expected_unknown"])

    def test_plain_answer_cli_writes_jsonl(self) -> None:
        import tempfile

        module = _load_script()
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "cases.jsonl"
            out = Path(temp_dir) / "plain.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": "case-cli",
                        "category": "direct",
                        "question": "Which badge unlocks Dock 7?",
                        "answer_aliases": ["blue-iris"],
                        "evidence": [
                            {
                                "source": "dock.md",
                                "chunk_id": 0,
                                "text": "Dock 7 opens with the blue-iris badge.",
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            count = module.write_canonical_plain_answer_data(
                source,
                out,
                evidence_mode="target",
            )

            rows = [
                json.loads(line)
                for line in out.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(count, 1)
            self.assertEqual(rows[0]["case_id"], "case-cli")
            self.assertEqual(rows[0]["answer"], "Answer: blue-iris")


if __name__ == "__main__":
    unittest.main()
