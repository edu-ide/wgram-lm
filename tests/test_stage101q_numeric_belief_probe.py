from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "583_build_stage101q_numeric_belief_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101q_numeric_belief_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101QNumericBeliefProbeTests(unittest.TestCase):
    def test_rows_have_numeric_belief_ledger(self) -> None:
        module = load_module()
        rows = module.numeric_belief_rows()

        case_types = {row["source_case_type"] for row in rows}
        self.assertIn("direct_reliable_numeric_belief", case_types)
        self.assertIn("untrusted_override_numeric_belief", case_types)
        self.assertIn("trusted_conflict_numeric_belief", case_types)
        self.assertIn("insufficient_numeric_belief", case_types)

        self.assertTrue(all(row["numeric_belief_required"] for row in rows))
        self.assertTrue(all(len(row["candidate_answers"]) == 3 for row in rows))
        self.assertTrue(all(len(row["negative_answers"]) == 2 for row in rows))
        self.assertTrue(
            all(row["intelligence_answer"] not in row["negative_answers"] for row in rows)
        )

        for row in rows:
            self.assertGreaterEqual(row["belief_support_score"], -1.0)
            self.assertLessEqual(row["belief_support_score"], 1.0)
            self.assertGreaterEqual(row["source_reliability_score"], 0.0)
            self.assertLessEqual(row["source_reliability_score"], 1.0)
            self.assertGreaterEqual(row["evidence_sufficiency_score"], 0.0)
            self.assertLessEqual(row["evidence_sufficiency_score"], 1.0)

    def test_same_claim_can_have_positive_negative_and_unknown_beliefs(self) -> None:
        module = load_module()
        rows = module.numeric_belief_rows()

        by_claim: dict[str, set[str]] = defaultdict(set)
        by_support: dict[str, set[float]] = defaultdict(set)
        for row in rows:
            by_claim[str(row["source_claim"])].add(str(row["final_answer"]))
            by_support[str(row["source_claim"])].add(float(row["belief_support_score"]))

        self.assertTrue(any({"True", "False", "Unknown"}.issubset(values) for values in by_claim.values()))
        self.assertTrue(any({-0.8, 0.0, 0.8}.issubset(values) for values in by_support.values()))

    def test_unknown_rows_have_low_sufficiency_and_neutral_support(self) -> None:
        module = load_module()
        rows = module.numeric_belief_rows() + module.numeric_belief_heldout_rows()
        unknown_rows = [row for row in rows if row["final_answer"] == "Unknown"]

        self.assertTrue(unknown_rows)
        self.assertTrue(all(abs(float(row["belief_support_score"])) <= 0.01 for row in unknown_rows))
        self.assertTrue(all(float(row["evidence_sufficiency_score"]) <= 0.2 for row in unknown_rows))

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
                    "--numeric-replay-factor",
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
        self.assertTrue(any(row.get("numeric_belief_required") for row in train_rows))
        self.assertTrue(all(row.get("numeric_belief_required") for row in heldout_rows))
        self.assertEqual(report["train_rows"], len(train_rows))
        self.assertEqual(report["eval_rows"], len(heldout_rows))

    def test_probe_rows_fit_stage101_seq_len_384(self) -> None:
        module = load_module()
        rows = module.numeric_belief_rows() + module.numeric_belief_heldout_rows()

        for row in rows:
            shifted_len = len(str(row["prompt"]).encode("utf-8")) + len(
                str(row["intelligence_answer"]).encode("utf-8")
            ) - 1
            self.assertLessEqual(shifted_len, 384, row["id"])


if __name__ == "__main__":
    unittest.main()
