from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def _load_script():
    path = Path("scripts/199_build_verified_reasoning_dataset.py")
    spec = importlib.util.spec_from_file_location("verified_reasoning_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class VerifiedReasoningDatasetTests(unittest.TestCase):
    def test_gsm8k_converter_uses_final_gold_answer_only(self) -> None:
        from qtrm_mm.data.verified_reasoning import convert_verified_row

        row = {
            "question": "Natalia sold 48 clips in April and half as many in May. Total?",
            "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 72",
        }

        converted = convert_verified_row(row, adapter="gsm8k", source_name="gsm8k", row_index=7)

        self.assertEqual(converted["answer"], "72")
        self.assertIn("71", converted["choices"])
        self.assertIn("73", converted["choices"])
        self.assertIn("Natalia sold", converted["prompt"])
        self.assertEqual(converted["source_dataset"], "gsm8k")
        self.assertEqual(converted["retrieval_allowed"], False)
        self.assertEqual(converted["memoryos_allowed"], False)
        self.assertNotIn("Natalia sold 48/2", converted.get("trace_summary", ""))

    def test_numina_converter_rejects_invalid_verification_flags(self) -> None:
        from qtrm_mm.data.verified_reasoning import convert_verified_row

        with self.assertRaises(ValueError):
            convert_verified_row(
                {
                    "problem": "Find x.",
                    "answer": "3",
                    "problem_is_valid": "No",
                    "solution_is_valid": "Yes",
                },
                adapter="numina_math_verifiable",
                source_name="numina",
                row_index=0,
            )

    def test_math_answer_converter_adds_generic_distractors_for_latex_answers(self) -> None:
        from qtrm_mm.data.verified_reasoning import convert_verified_row

        converted = convert_verified_row(
            {
                "problem": "Find the area.",
                "answer": "\\frac{3840}{289}",
                "subject": "Geometry",
            },
            adapter="math_answer",
            source_name="math500",
            row_index=0,
        )

        self.assertIn("\\frac{3840}{289}", converted["choices"])
        self.assertIn("0", converted["choices"])
        self.assertGreaterEqual(len(converted["choices"]), 2)

    def test_proofwriter_converter_maps_options_to_canonical_labels(self) -> None:
        from qtrm_mm.data.verified_reasoning import convert_verified_row

        row = {
            "context": "Bob is cold. If something is cold then it is blue.",
            "question": "Based on the above information, is Bob blue?",
            "answer": "A",
            "options": ["A) True", "B) False", "C) Unknown"],
        }

        converted = convert_verified_row(row, adapter="proofwriter", source_name="proof", row_index=1)

        self.assertEqual(converted["answer"], "TRUE")
        self.assertEqual(converted["choices"], ["TRUE", "FALSE", "UNKNOWN"])
        self.assertIn("Bob is cold", converted["prompt"])

    def test_clutrr_converter_keeps_relation_target(self) -> None:
        from qtrm_mm.data.verified_reasoning import convert_verified_row

        row = {
            "story": "[Ashley]'s daughter, [Lillian], helped [Nicholas].",
            "query": "('Ashley', 'Nicholas')",
            "target_text": "son",
        }

        converted = convert_verified_row(row, adapter="clutrr", source_name="clutrr", row_index=2)

        self.assertEqual(converted["answer"], "son")
        self.assertIn("daughter", converted["choices"])
        self.assertIn("relationship", converted["prompt"].lower())
        self.assertEqual(converted["task_family"], "relation_reasoning")

    def test_builder_writes_prompt_only_rows_from_local_jsonl(self) -> None:
        module = _load_script()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "gsm8k.jsonl"
            out = Path(tmp) / "out.jsonl"
            source.write_text(
                json.dumps({"question": "What is 2+2?", "answer": "#### 4"}) + "\n",
                encoding="utf-8",
            )

            stats = module.build_verified_reasoning_dataset(
                sources=[
                    module.SourceSpec(
                        name="unit_gsm8k",
                        dataset="local",
                        config="local",
                        split="train",
                        adapter="gsm8k",
                        local_jsonl=str(source),
                    )
                ],
                out_path=out,
                max_rows_per_source=10,
            )

            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(stats["written"], 1)
            self.assertEqual(rows[0]["answer"], "4")
            self.assertEqual(rows[0]["evidence"], [])
            self.assertEqual(rows[0]["distill_policy"], "verified_dataset_no_teacher_imitation")

    def test_builder_interleaves_sources_for_small_eval_slices(self) -> None:
        module = _load_script()
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.jsonl"
            second = Path(tmp) / "b.jsonl"
            out = Path(tmp) / "out.jsonl"
            first.write_text(
                "\n".join(
                    [
                        json.dumps({"question": "A1?", "answer": "#### 1"}),
                        json.dumps({"question": "A2?", "answer": "#### 2"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            second.write_text(
                "\n".join(
                    [
                        json.dumps({"question": "B1?", "answer": "#### 3"}),
                        json.dumps({"question": "B2?", "answer": "#### 4"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            module.build_verified_reasoning_dataset(
                sources=[
                    module.SourceSpec(
                        name="a",
                        dataset="local",
                        config="local",
                        split="train",
                        adapter="gsm8k",
                        local_jsonl=str(first),
                    ),
                    module.SourceSpec(
                        name="b",
                        dataset="local",
                        config="local",
                        split="train",
                        adapter="gsm8k",
                        local_jsonl=str(second),
                    ),
                ],
                out_path=out,
                max_rows_per_source=2,
            )

            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(
                [row["source_dataset"] for row in rows],
                ["a", "b", "a", "b"],
            )

    def test_default_sources_include_verified_raw_reasoning_sets(self) -> None:
        from qtrm_mm.data.verified_reasoning import DEFAULT_VERIFIED_SOURCES

        self.assertIn("gsm8k_train", DEFAULT_VERIFIED_SOURCES)
        self.assertIn("math500_test", DEFAULT_VERIFIED_SOURCES)
        self.assertIn("numina_verifiable_train", DEFAULT_VERIFIED_SOURCES)
        self.assertIn("proofwriter_validation", DEFAULT_VERIFIED_SOURCES)
        self.assertIn("clutrr_train", DEFAULT_VERIFIED_SOURCES)


if __name__ == "__main__":
    unittest.main()
