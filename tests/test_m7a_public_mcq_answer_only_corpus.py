from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def _load_script():
    path = Path("scripts/397_build_m7a_public_mcq_answer_only_corpus.py")
    spec = importlib.util.spec_from_file_location("m7a_public_mcq_corpus", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class M7APublicMCQAnswerOnlyCorpusTests(unittest.TestCase):
    def test_answer_record_text_appends_single_letter_after_prompt(self):
        module = _load_script()
        row = {
            "qtrm_prompt": "User: Q\nOptions:\nA. one\nB. two\n\nAnswer:\nAssistant:",
            "answer": "b",
        }

        text = module.answer_record_text(row)

        self.assertTrue(text.endswith(" B\n"))
        self.assertIn("Assistant: B", text)

    def test_build_corpus_writes_text_records_and_repair_seeds(self):
        module = _load_script()
        with TemporaryDirectory() as tmp:
            suite = Path(tmp) / "suite.jsonl"
            out_jsonl = Path(tmp) / "corpus.jsonl"
            out_json = Path(tmp) / "report.json"
            suite.write_text(
                json.dumps(
                    {
                        "benchmark_id": "mmlu_pro",
                        "case_id": "case-1",
                        "category": "math",
                        "qtrm_prompt": "User: Q\nAssistant:",
                        "answer": "C",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = module.build_arg_parser().parse_args(
                [
                    "--suite-jsonl",
                    str(suite),
                    "--out-jsonl",
                    str(out_jsonl),
                    "--out-json",
                    str(out_json),
                    "--repeats",
                    "2",
                    "--repair-seed-count",
                    "1",
                ]
            )

            report = module.build_corpus(args)

            rows = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(report["records"], 2)
        self.assertEqual(rows[0]["answer"], "C")
        self.assertIn("Assistant: C", rows[0]["text"])
        self.assertIn("User: Q", report["repair_seed_texts"])


if __name__ == "__main__":
    unittest.main()
