from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "585_build_stage101s_scalar_prior_calibration_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101s_scalar_prior_calibration_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101SScalarPriorCalibrationProbeTests(unittest.TestCase):
    def test_calibration_rows_focus_low_and_neutral_scalar_states(self) -> None:
        module = load_module()
        rows = module.scalar_prior_calibration_rows()

        self.assertTrue(rows)
        self.assertTrue(all(row["stage101s_scalar_prior_calibration_required"] for row in rows))

        answers_by_axis: dict[str, Counter[str]] = {
            "support": Counter(),
            "reliability": Counter(),
            "sufficiency": Counter(),
        }
        for row in rows:
            answers_by_axis[str(row["belief_axis"])][str(row["intelligence_answer"])] += 1

        self.assertGreaterEqual(answers_by_axis["support"][" +0.00"], answers_by_axis["support"][" +0.80"])
        self.assertGreaterEqual(answers_by_axis["support"][" +0.00"], answers_by_axis["support"][" -0.80"])
        self.assertGreaterEqual(answers_by_axis["reliability"][" 0.10"], answers_by_axis["reliability"][" 0.90"])
        self.assertGreaterEqual(answers_by_axis["sufficiency"][" 0.10"], answers_by_axis["sufficiency"][" 0.90"])

    def test_source_free_rows_are_present_before_source_semantics(self) -> None:
        module = load_module()
        rows = module.scalar_prior_calibration_rows()

        source_free = [
            row for row in rows if row["source_case_type"] == "source_free_scalar_prior_calibration"
        ]
        self.assertTrue(source_free)
        self.assertTrue(all("Evidence:" not in str(row["prompt"]) for row in source_free))
        self.assertTrue({row["belief_axis"] for row in source_free}.issuperset({"support", "reliability", "sufficiency"}))

    def test_build_keeps_anchors_base_factorized_rows_and_calibration_rows(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / "anchor.jsonl"
            base = root / "base.jsonl"
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
            base.write_text(
                json.dumps(
                    {
                        "id": "stage101r_factorized_train_00_insufficient_support",
                        "task": "factorized_numeric_belief_support_icl",
                        "prompt": "Q:",
                        "intelligence_answer": " +0.00",
                        "parrot_answer": " -0.80",
                        "negative_answers": [" -0.80", " +0.80"],
                        "factorized_numeric_belief_required": True,
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
                    "--base-factorized-jsonl",
                    str(base),
                    "--train-out",
                    str(train),
                    "--eval-out",
                    str(heldout),
                    "--base-replay-factor",
                    "2",
                    "--calibration-replay-factor",
                    "2",
                ]
            )
            report = module.build(args)
            train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
            heldout_rows = [json.loads(line) for line in heldout.read_text(encoding="utf-8").splitlines()]

        row_ids = {row["id"] for row in train_rows}
        self.assertIn("gd_lite_anchor", row_ids)
        self.assertTrue(any(row_id.endswith("_base_replay01") for row_id in row_ids))
        self.assertTrue(any(row.get("stage101s_scalar_prior_calibration_required") for row in train_rows))
        self.assertTrue(all(row.get("stage101s_scalar_prior_calibration_required") for row in heldout_rows))
        self.assertEqual(report["train_rows"], len(train_rows))
        self.assertEqual(report["eval_rows"], len(heldout_rows))

    def test_probe_rows_fit_stage101_seq_len_384(self) -> None:
        module = load_module()
        rows = module.scalar_prior_calibration_rows() + module.scalar_prior_calibration_heldout_rows()

        for row in rows:
            shifted_len = len(str(row["prompt"]).encode("utf-8")) + len(
                str(row["intelligence_answer"]).encode("utf-8")
            ) - 1
            self.assertLessEqual(shifted_len, 384, row["id"])


if __name__ == "__main__":
    unittest.main()
