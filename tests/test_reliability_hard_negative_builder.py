from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def _load_module():
    path = Path("scripts/188_build_reliability_hard_negative_preferences.py")
    spec = importlib.util.spec_from_file_location("reliability_hardneg", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReliabilityHardNegativeBuilderTests(unittest.TestCase):
    def test_builds_unknown_temporal_and_conflict_preferences_only(self) -> None:
        module = _load_module()
        cases = [
            {
                "id": "unknown-case",
                "category": "negative_authority_location_ko_synth",
                "question": "암구호는?",
                "answer_aliases": ["UNKNOWN", "unknown"],
                "evidence": [{"source": "signed.md", "text": "요청한 암구호는 이 기록에 없다."}],
                "distractors": [{"source": "anon.md", "text": "암구호는 바다-26이다."}],
            },
            {
                "id": "temporal-case",
                "category": "temporal_conflict_ko_synth",
                "question": "현재 코드는?",
                "answer_aliases": ["새벽-14", "새벽 14"],
                "evidence": [{"source": "current.md", "text": "현재 코드는 새벽-14이다."}],
                "distractors": [{"source": "old.md", "text": "이전 코드는 바다-26이다."}],
            },
            {
                "id": "plain-multihop",
                "category": "multi_hop_synth",
                "question": "Owner?",
                "answer_aliases": ["Tao Lin"],
                "evidence": [{"source": "a.md", "text": "Owner is Tao Lin."}],
                "distractors": [{"source": "b.md", "text": "Owner is Nora Vale."}],
            },
        ]
        records = [
            {
                "id": "unknown-case",
                "mode": "qtrm_residual_with_evidence",
                "hit": False,
                "completion": "Answer: 바다-26",
            },
            {
                "id": "unknown-case",
                "mode": "donor_only_with_evidence",
                "hit": False,
                "completion": "Answer: 비공개",
            },
            {
                "id": "temporal-case",
                "mode": "qtrm_residual_with_evidence",
                "hit": True,
                "completion": "Answer: 새벽-14",
            },
            {
                "id": "temporal-case",
                "mode": "donor_only_with_evidence",
                "hit": False,
                "completion": "Answer: 바다-26",
            },
            {
                "id": "temporal-case",
                "mode": "qtrm_core_off_with_evidence",
                "hit": False,
                "completion": "Answer: 바다-26",
            },
            {
                "id": "plain-multihop",
                "mode": "qtrm_residual_with_evidence",
                "hit": False,
                "completion": "Answer: Nora Vale",
            },
        ]

        rows = module.build_reliability_preference_rows(cases, records)
        by_case = {row["case_id"]: row for row in rows}

        self.assertIn("unknown-case", by_case)
        self.assertIn("temporal-case", by_case)
        self.assertNotIn("plain-multihop", by_case)
        self.assertEqual(by_case["unknown-case"]["chosen"], "Answer: UNKNOWN")
        self.assertEqual(by_case["unknown-case"]["rejected"], "Answer: 바다-26")
        self.assertIn("repair_qtrm_miss", by_case["unknown-case"]["metadata"]["reliability_reasons"])
        self.assertEqual(by_case["temporal-case"]["chosen"], "Answer: 새벽-14")
        self.assertEqual(by_case["temporal-case"]["rejected"], "Answer: 바다-26")
        self.assertIn(
            "strengthen_qtrm_win_over_donor",
            by_case["temporal-case"]["metadata"]["reliability_reasons"],
        )

    def test_cli_writes_rows_and_summary(self) -> None:
        module = _load_module()
        case = {
            "id": "unknown-case",
            "category": "negative_missing_synth",
            "question": "Which phrase?",
            "answer_aliases": ["UNKNOWN"],
            "evidence": [{"source": "signed.md", "text": "The phrase is absent."}],
            "distractors": [{"source": "anon.md", "text": "The phrase is amber-field."}],
        }
        records = [
            {
                "id": "unknown-case",
                "mode": "qtrm_residual_with_evidence",
                "hit": False,
                "completion": "Answer: amber-field",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cases_path = Path(tmp) / "cases.jsonl"
            eval_path = Path(tmp) / "eval.jsonl"
            out_path = Path(tmp) / "prefs.jsonl"
            cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
            eval_path.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")

            module.main(
                [
                    "--cases-jsonl",
                    str(cases_path),
                    "--eval-jsonl",
                    str(eval_path),
                    "--out-jsonl",
                    str(out_path),
                ]
            )

            written = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0]["type"], "reliability_hard_negative_preference")


if __name__ == "__main__":
    unittest.main()
