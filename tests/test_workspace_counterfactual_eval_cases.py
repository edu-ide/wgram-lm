import json
import tempfile
import unittest
from pathlib import Path


def load_workspace_counterfactual_module():
    import importlib.util

    script = Path("scripts/build_workspace_counterfactual_eval_cases.py")
    spec = importlib.util.spec_from_file_location("workspace_counterfactual_eval_cases", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class WorkspaceCounterfactualEvalCaseTests(unittest.TestCase):
    def test_build_workspace_swap_cases_keep_question_but_require_unknown(self):
        build_workspace_swap_cases = load_workspace_counterfactual_module().build_workspace_swap_cases

        cases = [
            {
                "id": "case-a",
                "question": "What is the vault code?",
                "answer_aliases": ["VX-913"],
                "evidence": [{"source": "a.md", "text": "The vault code is VX-913."}],
                "distractors": [],
            },
            {
                "id": "case-b",
                "question": "Who maintains Bay Opal?",
                "answer_aliases": ["Mira Sol"],
                "evidence": [{"source": "b.md", "text": "Bay Opal is maintained by Mira Sol."}],
                "distractors": [{"source": "b2.md", "text": "Bay Neon is maintained by Ilya Chen."}],
            },
        ]

        swapped = build_workspace_swap_cases(cases)

        self.assertEqual(len(swapped), 2)
        self.assertEqual(swapped[0]["question"], "What is the vault code?")
        self.assertEqual(swapped[0]["answer_aliases"], ["UNKNOWN", "unknown"])
        self.assertTrue(swapped[0]["expected_unknown"])
        self.assertEqual(swapped[0]["evidence"], [])
        self.assertIn("Bay Opal is maintained", swapped[0]["distractors"][0]["text"])
        self.assertNotIn("VX-913", json.dumps(swapped[0]["distractors"], ensure_ascii=False))

    def test_script_writes_limited_swap_cases(self):
        module = load_workspace_counterfactual_module()

        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "cases.jsonl"
            out_path = Path(tmp) / "swap.jsonl"
            rows = [
                {
                    "id": "case-a",
                    "question": "What is A?",
                    "answer_aliases": ["Alpha"],
                    "evidence": [{"source": "a.md", "text": "A is Alpha."}],
                },
                {
                    "id": "case-b",
                    "question": "What is B?",
                    "answer_aliases": ["Beta"],
                    "evidence": [{"source": "b.md", "text": "B is Beta."}],
                },
            ]
            in_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            count = module.write_workspace_swap_cases(str(in_path), str(out_path), max_cases=1)

            written = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(count, 1)
            self.assertEqual(written[0]["id"], "case-a__workspace_swap")


if __name__ == "__main__":
    unittest.main()
