from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def _load_script():
    path = Path("scripts/404_materialize_aj_remap_mcq.py")
    spec = importlib.util.spec_from_file_location("aj_remap_mcq", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AJRemapMCQMaterializerTests(unittest.TestCase):
    def test_remap_row_builds_ten_options_and_preserves_correct_text(self):
        module = _load_script()
        rows = [
            {
                "question": "Which item is correct?",
                "options": ["alpha", "beta", "gamma", "delta"],
                "answer": "C",
                "category": "test",
            },
            {
                "question": "Distractor source?",
                "options": ["red", "green", "blue", "orange"],
                "answer": "A",
                "category": "test",
            },
            {
                "question": "More distractors?",
                "options": ["one", "two", "three", "four"],
                "answer": "B",
                "category": "test",
            },
            {
                "question": "Even more?",
                "options": ["cat", "dog", "eel", "fox"],
                "answer": "D",
                "category": "test",
            },
            {
                "question": "Fifth source?",
                "options": ["north", "south", "east", "west"],
                "answer": "A",
                "category": "test",
            },
            {
                "question": "Sixth source?",
                "options": ["iron", "copper", "silver", "gold"],
                "answer": "B",
                "category": "test",
            },
        ]
        pool = module.wrong_option_pool(rows)

        remapped = module.remap_row(
            rows[0],
            pool=pool,
            rng=module.random.Random(7),
            source_index=0,
            repeat_index=0,
        )

        self.assertEqual(len(remapped["options"]), 10)
        self.assertEqual(remapped["options"][remapped["answer_index"]], "gamma")
        self.assertIn(f'{remapped["answer"]}. gamma', remapped["qtrm_prompt"])

    def test_build_cases_appends_anchor_rows(self):
        module = _load_script()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            anchor = root / "anchor.jsonl"
            rows = []
            for index in range(6):
                rows.append(
                    {
                        "question": f"Q{index}?",
                        "options": [f"a{index}", f"b{index}", f"c{index}", f"d{index}"],
                        "answer": "A",
                        "category": "test",
                    }
                )
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            anchor.write_text(
                json.dumps(
                    {
                        "question": "Anchor?",
                        "options": ["a", "b"],
                        "answer": "B",
                        "qtrm_prompt": "User: Anchor\nAssistant:",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            cases = module.build_cases(
                module.argparse.Namespace(
                    source_jsonl=[str(source)],
                    anchor_jsonl=[str(anchor)],
                    augment_repeats=1,
                    max_augmented_cases=3,
                    max_cases=0,
                    seed=11,
                    shuffle=False,
                )
            )

        self.assertEqual(len(cases), 4)
        self.assertEqual(len(cases[0]["options"]), 10)
        self.assertEqual(cases[-1]["question"], "Anchor?")

    def test_clean_text_removes_jsonl_breaking_control_whitespace(self):
        module = _load_script()

        self.assertEqual(
            module.clean_text("body\u0085mass\nindex\tvalue"),
            "body mass index value",
        )


if __name__ == "__main__":
    unittest.main()
