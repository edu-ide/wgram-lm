import importlib.util
from pathlib import Path
import tempfile
import unittest


def _load_module():
    path = Path("scripts/186_build_clean_intervention_preferences.py")
    spec = importlib.util.spec_from_file_location("clean_intervention_preferences", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CleanInterventionPreferencesTests(unittest.TestCase):
    def test_clean_rows_use_short_canonical_chosen_answers(self):
        module = _load_module()
        rows = [
            {
                "type": "intervention_policy_preference",
                "case_id": "c1",
                "prompt": "Question: test",
                "chosen": "Answer: noisy answer with SOURCE=x",
                "rejected": "Answer: wrong",
                "preference_weight": 1.0,
                "metadata": {"answer_aliases": ["opal-river", "opal river"]},
            },
            {
                "type": "intervention_policy_preference",
                "case_id": "c2",
                "prompt": "Question: missing",
                "chosen": "Answer: noisy UNKNOWN trace",
                "rejected": "Answer: 비공개",
                "preference_weight": 1.0,
                "metadata": {"answer_aliases": ["UNKNOWN", "unknown"]},
            },
        ]

        cleaned = module.clean_rows(rows)

        self.assertEqual(cleaned[0]["chosen"], "Answer: opal-river")
        self.assertEqual(cleaned[1]["chosen"], "Answer: UNKNOWN")
        self.assertTrue(cleaned[0]["metadata"]["clean_intervention_preference"])

    def test_cli_writes_jsonl(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.jsonl"
            dst = Path(tmp) / "out.jsonl"
            src.write_text(
                '{"prompt":"Q","chosen":"Answer: noisy","rejected":"Answer: wrong",'
                '"metadata":{"answer_aliases":["A"]}}\n',
                encoding="utf-8",
            )

            module.main(["--input-jsonl", str(src), "--output-jsonl", str(dst)])

            self.assertIn('"chosen": "Answer: A"', dst.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
