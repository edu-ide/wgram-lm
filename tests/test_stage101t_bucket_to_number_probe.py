from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "586_build_stage101t_bucket_to_number_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101t_bucket_to_number_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101TBucketToNumberProbeTests(unittest.TestCase):
    def test_rows_include_semantic_bucket_and_numeric_readback_lessons(self) -> None:
        module = load_module()
        rows = module.bucket_to_number_rows()

        self.assertTrue(rows)
        self.assertTrue(all(row["stage101t_bucket_to_number_required"] for row in rows))

        lesson_types = {row["stage101t_lesson_type"] for row in rows}
        self.assertEqual({"semantic_bucket", "numeric_readback"}, lesson_types)

        bucket_axes = {
            row["belief_axis"] for row in rows if row["stage101t_lesson_type"] == "semantic_bucket"
        }
        readback_axes = {
            row["belief_axis"] for row in rows if row["stage101t_lesson_type"] == "numeric_readback"
        }
        self.assertEqual({"support_bucket", "reliability_bucket", "sufficiency_bucket"}, bucket_axes)
        self.assertEqual({"support", "reliability", "sufficiency"}, readback_axes)

    def test_bucket_labels_map_to_numeric_targets(self) -> None:
        module = load_module()
        rows = module.bucket_to_number_rows()

        mapping_rows = [
            row for row in rows if row["stage101t_lesson_type"] == "numeric_readback"
        ]
        mapping = {
            (row["readback_axis"], row["readback_bucket_answer"]): row["intelligence_answer"]
            for row in mapping_rows
        }

        self.assertEqual(" +0.00", mapping[("support", " neutral")])
        self.assertEqual(" +0.80", mapping[("support", " supports")])
        self.assertEqual(" -0.80", mapping[("support", " contradicts")])
        self.assertEqual(" 0.10", mapping[("reliability", " low")])
        self.assertEqual(" 0.50", mapping[("reliability", " unknown")])
        self.assertEqual(" 0.90", mapping[("reliability", " high")])
        self.assertEqual(" 0.10", mapping[("sufficiency", " insufficient")])
        self.assertEqual(" 0.50", mapping[("sufficiency", " partial")])
        self.assertEqual(" 0.90", mapping[("sufficiency", " sufficient")])

    def test_source_cases_have_bucket_and_readback_rows_per_axis(self) -> None:
        module = load_module()
        rows = module.bucket_to_number_rows()

        by_case_axis: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in rows:
            if row["source_case_type"] == "bucket_dictionary_readback":
                continue
            axis = str(row["readback_axis"])
            by_case_axis[(str(row["bucket_case_id"]), axis)].add(str(row["stage101t_lesson_type"]))

        self.assertTrue(by_case_axis)
        self.assertTrue(
            all({"semantic_bucket", "numeric_readback"}.issubset(types) for types in by_case_axis.values())
        )

    def test_build_keeps_anchors_and_writes_heldout(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / "anchor.jsonl"
            train = root / "train.jsonl"
            heldout = root / "heldout.jsonl"
            anchor.write_text(
                json.dumps(
                    {
                        "id": "gd_lite_anchor",
                        "task": "truthy_answer_icl",
                        "prompt": "Q:",
                        "intelligence_answer": " True",
                        "parrot_answer": " False",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = module.build_arg_parser().parse_args(
                [
                    "--anchor-jsonl",
                    str(anchor),
                    "--extra-anchor-jsonl",
                    "",
                    "--train-out",
                    str(train),
                    "--eval-out",
                    str(heldout),
                    "--bucket-replay-factor",
                    "2",
                ]
            )
            report = module.build(args)
            train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
            heldout_rows = [json.loads(line) for line in heldout.read_text(encoding="utf-8").splitlines()]

        self.assertIn("gd_lite_anchor", {row["id"] for row in train_rows})
        self.assertTrue(any(row.get("stage101t_bucket_to_number_required") for row in train_rows))
        self.assertTrue(all(row.get("stage101t_bucket_to_number_required") for row in heldout_rows))
        self.assertEqual(report["train_rows"], len(train_rows))
        self.assertEqual(report["eval_rows"], len(heldout_rows))

    def test_probe_rows_fit_stage101_seq_len_384(self) -> None:
        module = load_module()
        rows = module.bucket_to_number_rows() + module.bucket_to_number_heldout_rows()

        for row in rows:
            shifted_len = len(str(row["prompt"]).encode("utf-8")) + len(
                str(row["intelligence_answer"]).encode("utf-8")
            ) - 1
            self.assertLessEqual(shifted_len, 384, row["id"])


if __name__ == "__main__":
    unittest.main()
