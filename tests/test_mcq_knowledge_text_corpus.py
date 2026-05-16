from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def _load_script():
    path = Path("scripts/406_build_mcq_knowledge_text_corpus.py")
    spec = importlib.util.spec_from_file_location("mcq_knowledge_text_corpus", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MCQKnowledgeTextCorpusTests(unittest.TestCase):
    def test_record_text_uses_correct_option_content(self):
        module = _load_script()
        row = {
            "question": "Which color is grass?",
            "options": ["red", "green", "blue"],
            "answer": "B",
            "category": "science",
        }

        text = module.record_text(row)

        self.assertIn("Which color is grass?", text)
        self.assertIn("Assistant: green", text)
        self.assertNotIn("Assistant: B", text)

    def test_build_records_repeats_rows(self):
        module = _load_script()
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "rows.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "question": "Q?",
                        "options": ["wrong", "right"],
                        "answer": "B",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            records = module.build_records(
                module.argparse.Namespace(
                    source_jsonl=[str(source)],
                    max_rows=0,
                    repeats=3,
                    seed=1,
                    shuffle=False,
                )
            )

        self.assertEqual(len(records), 3)
        self.assertTrue(all("Assistant: right" in record["text"] for record in records))


if __name__ == "__main__":
    unittest.main()
