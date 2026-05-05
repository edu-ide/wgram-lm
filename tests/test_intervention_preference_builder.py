from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def _load_module():
    path = Path("scripts/180_build_intervention_preferences.py")
    spec = importlib.util.spec_from_file_location("intervention_preferences", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InterventionPreferenceBuilderTests(unittest.TestCase):
    def test_builds_preserve_and_allow_rows_from_eval_records(self) -> None:
        module = _load_module()
        cases = [
            {
                "id": "a",
                "category": "authority_conflict_synth",
                "question": "Which passphrase opens the Ember vault?",
                "answer_aliases": ["opal-river"],
                "evidence": [{"source": "signed.md", "text": "The passphrase is opal-river."}],
                "distractors": [{"source": "anon.md", "text": "The passphrase is stone-arch."}],
            },
            {
                "id": "b",
                "category": "negative_authority_location_ko_synth",
                "question": "암구호는?",
                "answer_aliases": ["UNKNOWN"],
                "evidence": [{"source": "signed.md", "text": "The answer is redacted."}],
                "distractors": [{"source": "anon.md", "text": "The answer is private."}],
            },
        ]
        records = [
            {"id": "a", "mode": "donor_only_with_evidence", "hit": True, "completion": "Answer: opal-river"},
            {"id": "a", "mode": "qtrm_residual_with_evidence", "hit": False, "completion": "Answer: stone-arch"},
            {"id": "a", "mode": "qtrm_core_off_with_evidence", "hit": True, "completion": "Answer: opal-river"},
            {"id": "b", "mode": "donor_only_with_evidence", "hit": False, "completion": "Answer: private"},
            {"id": "b", "mode": "qtrm_residual_with_evidence", "hit": True, "completion": "Answer: UNKNOWN"},
            {"id": "b", "mode": "qtrm_core_off_with_evidence", "hit": False, "completion": "Answer: private"},
        ]

        rows = module.build_intervention_preference_rows(cases, records)
        reasons = {row["metadata"]["intervention_reason"] for row in rows}

        self.assertIn("preserve_donor", reasons)
        self.assertIn("allow_qtrm", reasons)
        preserve = next(row for row in rows if row["metadata"]["intervention_reason"] == "preserve_donor")
        self.assertEqual(preserve["chosen"], "Answer: opal-river")
        self.assertEqual(preserve["rejected"], "Answer: stone-arch")
        self.assertIn("Which passphrase", preserve["prompt"])

    def test_cli_writes_jsonl(self) -> None:
        module = _load_module()
        case = {
            "id": "a",
            "question": "Which passphrase opens the Ember vault?",
            "answer_aliases": ["opal-river"],
            "evidence": [{"source": "signed.md", "text": "The passphrase is opal-river."}],
            "distractors": [{"source": "anon.md", "text": "The passphrase is stone-arch."}],
        }
        records = [
            {"id": "a", "mode": "donor_only_with_evidence", "hit": True, "completion": "Answer: opal-river"},
            {"id": "a", "mode": "qtrm_residual_with_evidence", "hit": False, "completion": "Answer: stone-arch"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cases_path = Path(tmp) / "cases.jsonl"
            eval_path = Path(tmp) / "eval.jsonl"
            out_path = Path(tmp) / "prefs.jsonl"
            cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
            eval_path.write_text(
                "".join(json.dumps(row) + "\n" for row in records),
                encoding="utf-8",
            )

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
            self.assertEqual(written[0]["type"], "intervention_policy_preference")


if __name__ == "__main__":
    unittest.main()
