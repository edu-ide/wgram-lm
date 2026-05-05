from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


def _load_script():
    path = Path("scripts/170_build_canonical_decision_token_data.py")
    spec = importlib.util.spec_from_file_location("canonical_decision_token_data", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CanonicalDecisionTokenDataTests(unittest.TestCase):
    def test_canonical_decision_tokens_keep_evidence_in_single_prompt_stream(self) -> None:
        module = _load_script()
        row = module.build_canonical_decision_token_row(
            {
                "id": "case-supported",
                "category": "direct",
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

        self.assertEqual(row["type"], "canonical_decision_tokens")
        self.assertEqual(row["ssot_contract"], "single_visible_prompt_stream")
        self.assertNotIn("workspace_context", row)
        self.assertIn("MemoryOS evidence", row["prompt"])
        self.assertIn("VX-913", row["prompt"])
        self.assertIn("PL-404", row["prompt"])
        self.assertIn("Verify:", row["answer"])
        self.assertIn("Decision: ANSWER", row["answer"])
        self.assertIn("Answer: VX-913", row["answer"])
        self.assertEqual(row["metadata"]["answer_policy"], "greedy_autoregressive")
        self.assertEqual(row["metadata"]["decision_target"], "ANSWER")
        self.assertFalse(row["metadata"]["expected_unknown"])

    def test_canonical_decision_tokens_abstain_when_answer_is_unknown(self) -> None:
        module = _load_script()
        row = module.build_canonical_decision_token_row(
            {
                "id": "case-missing",
                "category": "negative_missing",
                "question": "What is the Garnet override phrase?",
                "answer_aliases": ["UNKNOWN"],
                "evidence": [
                    {
                        "source": "signed.md",
                        "chunk_id": 0,
                        "text": "Signed notice: the Garnet override phrase is redacted.",
                    }
                ],
            }
        )

        self.assertEqual(row["metadata"]["verification_label"], "missing")
        self.assertEqual(row["metadata"]["decision_target"], "ABSTAIN")
        self.assertIn("Verify: missing", row["answer"])
        self.assertIn("Decision: ABSTAIN", row["answer"])
        self.assertTrue(row["answer"].rstrip().endswith("Answer: UNKNOWN"))

    def test_canonical_decision_token_cli_writes_jsonl(self) -> None:
        import tempfile

        module = _load_script()
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "cases.jsonl"
            out = Path(temp_dir) / "decision_tokens.jsonl"
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

            count = module.write_canonical_decision_token_data(
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
            self.assertEqual(rows[0]["metadata"]["decision_target"], "ANSWER")
            self.assertIn("Answer: blue-iris", rows[0]["answer"])


if __name__ == "__main__":
    unittest.main()
