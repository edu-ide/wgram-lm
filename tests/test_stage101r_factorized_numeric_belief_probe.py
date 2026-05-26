from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "584_build_stage101r_factorized_numeric_belief_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101r_factorized_numeric_belief_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101RFactorizedNumericBeliefProbeTests(unittest.TestCase):
    def test_rows_cover_all_scalar_axes(self) -> None:
        module = load_module()
        rows = module.factorized_numeric_belief_rows()

        axes = {row["belief_axis"] for row in rows}
        self.assertEqual({"support", "reliability", "sufficiency"}, axes)
        self.assertTrue(all(row["factorized_numeric_belief_required"] for row in rows))
        self.assertTrue(all(len(row["negative_answers"]) == 2 for row in rows))
        self.assertTrue(
            all(row["intelligence_answer"] not in row["negative_answers"] for row in rows)
        )

    def test_same_claim_has_positive_negative_and_neutral_support(self) -> None:
        module = load_module()
        rows = module.factorized_numeric_belief_rows()

        by_claim: dict[str, set[float]] = defaultdict(set)
        for row in rows:
            if row["belief_axis"] == "support":
                by_claim[str(row["source_claim"])].add(float(row["belief_support_score"]))

        self.assertTrue(any({-0.8, 0.0, 0.8}.issubset(values) for values in by_claim.values()))

    def test_untrusted_and_insufficient_are_different_states(self) -> None:
        module = load_module()
        rows = module.factorized_numeric_belief_rows() + module.factorized_numeric_belief_heldout_rows()

        insufficient_rows = [
            row for row in rows if row["source_case_type"] == "insufficient_factorized_belief"
        ]
        untrusted_rows = [
            row for row in rows if row["source_case_type"] == "untrusted_only_factorized_belief"
        ]
        self.assertTrue(insufficient_rows)
        self.assertTrue(untrusted_rows)

        self.assertTrue(all(float(row["source_reliability_score"]) == 0.9 for row in insufficient_rows))
        self.assertTrue(all(float(row["evidence_sufficiency_score"]) == 0.1 for row in insufficient_rows))
        self.assertTrue(all(float(row["source_reliability_score"]) == 0.1 for row in untrusted_rows))
        self.assertTrue(all(float(row["evidence_sufficiency_score"]) == 0.1 for row in untrusted_rows))

    def test_build_writes_anchors_replayed_rows_and_heldout(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / "anchor.jsonl"
            extra_anchor = root / "extra_anchor.jsonl"
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
            extra_anchor.write_text(
                json.dumps(
                    {
                        "id": "stage101b_anchor",
                        "task": "truthy_answer_icl",
                        "prompt": "Q:",
                        "intelligence_answer": " False",
                        "parrot_answer": " True",
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
                    str(extra_anchor),
                    "--train-out",
                    str(train),
                    "--eval-out",
                    str(heldout),
                    "--scalar-replay-factor",
                    "2",
                ]
            )
            report = module.build(args)
            train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
            heldout_rows = [json.loads(line) for line in heldout.read_text(encoding="utf-8").splitlines()]

        row_ids = {row["id"] for row in train_rows}
        self.assertIn("gd_lite_anchor", row_ids)
        self.assertIn("stage101b_anchor", row_ids)
        self.assertTrue(any(row["id"].endswith("_replay01") for row in train_rows))
        self.assertTrue(any(row.get("factorized_numeric_belief_required") for row in train_rows))
        self.assertTrue(all(row.get("factorized_numeric_belief_required") for row in heldout_rows))
        self.assertEqual(report["train_rows"], len(train_rows))
        self.assertEqual(report["eval_rows"], len(heldout_rows))

    def test_probe_rows_fit_stage101_seq_len_384(self) -> None:
        module = load_module()
        rows = module.factorized_numeric_belief_rows() + module.factorized_numeric_belief_heldout_rows()

        for row in rows:
            shifted_len = len(str(row["prompt"]).encode("utf-8")) + len(
                str(row["intelligence_answer"]).encode("utf-8")
            ) - 1
            self.assertLessEqual(shifted_len, 384, row["id"])


if __name__ == "__main__":
    unittest.main()
