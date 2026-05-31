import json
from pathlib import Path
import tempfile
import unittest


class HRMTextSourceMixTests(unittest.TestCase):
    def test_verified_row_converts_to_hrm_text_instruction_response(self) -> None:
        from wgram_lm.data.hrm_text_source_mix import verified_to_hrm_text_row

        row = verified_to_hrm_text_row(
            {
                "prompt": "Answer with only the final answer.\nQuestion: What is 2+2?\nAnswer:",
                "answer": "4",
                "reasoning_family": "math_word_problem",
                "source_dataset": "gsm8k_train",
            }
        )

        self.assertEqual(row["response"], "4")
        self.assertIn("verified,math_word_problem,answer_only,gsm8k_train", row["condition"])
        self.assertIn("What is 2+2?", row["instruction"])
        self.assertFalse(row["instruction"].endswith("Answer:"))

    def test_build_mix_from_local_verified_source(self) -> None:
        from wgram_lm.data.hrm_text_source_mix import SourceSpec, build_hrm_text_source_mix

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "gsm8k.jsonl"
            source.write_text(
                json.dumps({"question": "What is 3+4?", "answer": "#### 7"}) + "\n",
                encoding="utf-8",
            )

            stats = build_hrm_text_source_mix(
                out_dir=root / "mix",
                verified_sources=[
                    SourceSpec(
                        name="unit_gsm8k",
                        dataset="local",
                        config="local",
                        split="train",
                        adapter="gsm8k",
                        local_jsonl=str(source),
                    )
                ],
                max_verified_rows_per_source=5,
                dolly_rows=0,
                seed=1,
            )

            path = Path(stats["files"]["verified_reasoning"])
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(stats["verified"]["written"], 1)
            self.assertEqual(rows[0]["response"], "7")
            self.assertIn("instruction", rows[0])
            self.assertTrue((root / "mix" / "manifest.json").exists())

    def test_build_mix_respects_verified_offset_for_eval_splits(self) -> None:
        from wgram_lm.data.hrm_text_source_mix import SourceSpec, build_hrm_text_source_mix

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "gsm8k.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps({"question": "First?", "answer": "#### 1"}),
                        json.dumps({"question": "Second?", "answer": "#### 2"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            stats = build_hrm_text_source_mix(
                out_dir=root / "mix_eval",
                verified_sources=[
                    SourceSpec(
                        name="unit_gsm8k",
                        dataset="local",
                        config="local",
                        split="train",
                        adapter="gsm8k",
                        local_jsonl=str(source),
                    )
                ],
                max_verified_rows_per_source=5,
                verified_offset=1,
                dolly_rows=0,
                seed=1,
            )

            path = Path(stats["files"]["verified_reasoning"])
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["response"], "2")
            self.assertEqual(stats["verified_offset"], 1)


if __name__ == "__main__":
    unittest.main()
