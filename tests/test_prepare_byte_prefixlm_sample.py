from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "555_prepare_byte_prefixlm_sample.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_byte_prefixlm_sample", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PrepareBytePrefixLMSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_build_dataset_accepts_jsonl_and_parquet_globs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cleaned"
            out = Path(tmp) / "sampled"
            (root / "data").mkdir(parents=True)
            (root / "data_clustered" / "SYNTH").mkdir(parents=True)
            (root / "data" / "general.jsonl").write_text(
                json.dumps({"instruction": "Say hello", "response": "Hello there."}) + "\n",
                encoding="utf-8",
            )
            table = pa.table(
                {
                    "instruction": ["Compute 2+2", "Translate hello to Korean"],
                    "condition": ["math", "ko"],
                    "response": ["4", "안녕하세요"],
                }
            )
            pq.write_table(table, root / "data_clustered" / "SYNTH" / "part.parquet")
            args = argparse.Namespace(
                cleaned_data_root=str(root),
                source_files="data/general.jsonl",
                source_globs="data_clustered/SYNTH/*.parquet",
                out=str(out),
                epochs=1,
                max_rows=0,
                max_rows_per_file=0,
                max_scan_rows_per_file=0,
                max_inst_bytes=512,
                max_resp_bytes=512,
                shuffle_epochs=False,
                seed=7,
            )

            report = self.module.build_dataset(args)

            self.assertEqual(report["rows"], 3)
            self.assertEqual(
                report["source_files"],
                ["data/general.jsonl", "data_clustered/SYNTH/part.parquet"],
            )
            self.assertEqual(report["accepted_by_file"]["data/general.jsonl"], 1)
            self.assertEqual(report["accepted_by_file"]["data_clustered/SYNTH/part.parquet"], 2)
            self.assertTrue((out / "tokens.npy").is_file())
            self.assertGreater(int(np.load(out / "tokens.npy").shape[0]), 0)

    def test_build_dataset_caps_scanned_rows_per_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cleaned"
            out = Path(tmp) / "sampled"
            (root / "data_clustered" / "large").mkdir(parents=True)
            table = pa.table(
                {
                    "instruction": ["x" * 2000, "Say hello"],
                    "response": ["too long", "Hello."],
                    "unused_blob": ["y" * 10000, "z" * 10000],
                }
            )
            pq.write_table(table, root / "data_clustered" / "large" / "part.parquet")
            args = argparse.Namespace(
                cleaned_data_root=str(root),
                source_files="",
                source_globs="data_clustered/large/*.parquet",
                out=str(out),
                epochs=1,
                max_rows=0,
                max_rows_per_file=0,
                max_scan_rows_per_file=1,
                max_inst_bytes=512,
                max_resp_bytes=512,
                shuffle_epochs=False,
                seed=7,
            )

            with self.assertRaisesRegex(ValueError, "no rows accepted"):
                self.module.build_dataset(args)

            args.max_scan_rows_per_file = 2
            report = self.module.build_dataset(args)

            self.assertEqual(report["rows"], 1)
            self.assertEqual(report["rejected"]["instruction_too_long"], 1)
            self.assertEqual(report["rejected"]["scan_limit"], 0)
            self.assertEqual(
                report["scanned_by_file"],
                {"data_clustered/large/part.parquet": 2},
            )

    def test_build_dataset_honors_curriculum_bucket_quotas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cleaned"
            out = Path(tmp) / "sampled"
            (root / "data").mkdir(parents=True)
            (root / "data_clustered" / "SYNTH").mkdir(parents=True)
            (root / "data" / "no_robots.jsonl").write_text(
                "".join(
                    json.dumps({"instruction": f"Talk naturally {idx}", "response": f"Natural reply {idx}."}) + "\n"
                    for idx in range(5)
                ),
                encoding="utf-8",
            )
            pq.write_table(
                pa.table(
                    {
                        "instruction": [f"Compute {idx}+{idx}" for idx in range(10)],
                        "response": [str(idx + idx) for idx in range(10)],
                    }
                ),
                root / "data_clustered" / "SYNTH" / "part.parquet",
            )
            args = argparse.Namespace(
                cleaned_data_root=str(root),
                source_files="data/no_robots.jsonl",
                source_globs="data_clustered/SYNTH/*.parquet",
                out=str(out),
                epochs=1,
                max_rows=0,
                max_rows_per_file=0,
                max_scan_rows_per_file=0,
                max_inst_bytes=512,
                max_resp_bytes=512,
                bucket_quotas="general_instruction=3 synthetic_math_like=2",
                bucket_max_rows_per_file="general_instruction=3 synthetic_math_like=2",
                shuffle_epochs=False,
                seed=7,
            )

            report = self.module.build_dataset(args)

            self.assertEqual(report["rows"], 5)
            self.assertEqual(report["accepted_by_bucket"]["general_instruction"], 3)
            self.assertEqual(report["accepted_by_bucket"]["synthetic_math_like"], 2)
            self.assertEqual(
                report["source_bucket_contract"]["bucket_quotas"],
                {"general_instruction": 3, "synthetic_math_like": 2},
            )

    def test_utility_selection_prefers_high_scored_rows_before_first_seen_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cleaned"
            out = Path(tmp) / "sampled"
            scores = Path(tmp) / "scores.jsonl"
            (root / "data").mkdir(parents=True)
            (root / "data" / "no_robots.jsonl").write_text(
                "".join(
                    json.dumps({"instruction": f"Prompt {idx}", "response": f"Reply {idx}."}) + "\n"
                    for idx in range(3)
                ),
                encoding="utf-8",
            )
            scores.write_text(
                "\n".join(
                    [
                        json.dumps({"source_file": "data/no_robots.jsonl", "row_index": 0, "utility": -1.0}),
                        json.dumps({"source_file": "data/no_robots.jsonl", "row_index": 1, "utility": 9.0}),
                        json.dumps({"source_file": "data/no_robots.jsonl", "row_index": 2, "utility": 1.0}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                cleaned_data_root=str(root),
                source_files="data/no_robots.jsonl",
                source_globs="",
                out=str(out),
                epochs=1,
                max_rows=1,
                max_rows_per_file=1,
                max_scan_rows_per_file=0,
                max_inst_bytes=512,
                max_resp_bytes=512,
                bucket_quotas="",
                bucket_max_rows_per_file="",
                utility_score_jsonl=str(scores),
                selection_mode="utility",
                utility_temperature=0.0,
                shuffle_epochs=False,
                seed=7,
            )

            report = self.module.build_dataset(args)

            self.assertEqual(report["rows"], 1)
            self.assertEqual(report["accepted_row_indices_by_file"]["data/no_robots.jsonl"], [1])
            self.assertEqual(report["data_selection_contract"]["selection_mode"], "utility")
            self.assertEqual(report["data_selection_contract"]["utility_scores_loaded"], 3)
            self.assertIn("OPUS-compatible", report["data_selection_contract"]["plain_language"])

    def test_utility_selection_does_not_treat_unscored_rows_as_zero_utility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cleaned"
            out = Path(tmp) / "sampled"
            scores = Path(tmp) / "scores.jsonl"
            (root / "data").mkdir(parents=True)
            (root / "data" / "no_robots.jsonl").write_text(
                "".join(
                    json.dumps({"instruction": f"Prompt {idx}", "response": f"Reply {idx}."}) + "\n"
                    for idx in range(3)
                ),
                encoding="utf-8",
            )
            scores.write_text(
                json.dumps({"source_file": "data/no_robots.jsonl", "row_index": 1, "utility": -9.0}) + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                cleaned_data_root=str(root),
                source_files="data/no_robots.jsonl",
                source_globs="",
                out=str(out),
                epochs=1,
                max_rows=1,
                max_rows_per_file=1,
                max_scan_rows_per_file=0,
                max_inst_bytes=512,
                max_resp_bytes=512,
                bucket_quotas="",
                bucket_max_rows_per_file="",
                utility_score_jsonl=str(scores),
                selection_mode="utility",
                utility_temperature=0.0,
                shuffle_epochs=False,
                seed=7,
            )

            report = self.module.build_dataset(args)

            self.assertEqual(report["accepted_row_indices_by_file"]["data/no_robots.jsonl"], [1])
            self.assertEqual(report["data_selection_contract"]["accepted_scored_rows"], 1)
            self.assertEqual(report["data_selection_contract"]["accepted_unscored_rows"], 0)


if __name__ == "__main__":
    unittest.main()
