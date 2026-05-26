from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "582_build_stage101p_belief_update_source_reliability_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101p_belief_update_source_reliability_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101PBeliefUpdateSourceReliabilityProbeTests(unittest.TestCase):
    def test_rows_cover_belief_update_reliability_conflict_and_unknown(self) -> None:
        module = load_module()
        rows = module.belief_update_rows()

        case_types = {row["source_case_type"] for row in rows}
        self.assertIn("claim_first_belief_revision", case_types)
        self.assertIn("untrusted_source_override", case_types)
        self.assertIn("trusted_source_conflict", case_types)
        self.assertIn("insufficient_source_unknown", case_types)

        answers = {row["intelligence_answer"] for row in rows}
        self.assertIn(" True", answers)
        self.assertIn(" False", answers)
        self.assertIn(" Unknown", answers)
        self.assertTrue(all(row["belief_update_required"] for row in rows))
        self.assertTrue(all(row["candidate_answers"] == [" True", " False", " Unknown"] for row in rows))
        self.assertTrue(all(len(row["negative_answers"]) == 2 for row in rows))
        self.assertTrue(
            all(row["intelligence_answer"] not in row["negative_answers"] for row in rows)
        )

    def test_counterfactual_same_claim_can_require_different_answers(self) -> None:
        module = load_module()
        rows = module.belief_update_rows()

        by_claim: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            by_claim[str(row["source_claim"])].add(str(row["intelligence_answer"]))

        self.assertTrue(any(len(answers) >= 2 for answers in by_claim.values()))

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
                    "--source-replay-factor",
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
        self.assertTrue(any(row.get("belief_update_required") for row in train_rows))
        self.assertTrue(all(row.get("belief_update_required") for row in heldout_rows))
        self.assertEqual(report["train_rows"], len(train_rows))
        self.assertEqual(report["eval_rows"], len(heldout_rows))
        self.assertGreaterEqual(report["belief_update_rows"], 24)

    def test_probe_rows_fit_stage101_seq_len_384(self) -> None:
        module = load_module()
        rows = module.belief_update_rows() + module.belief_update_heldout_rows()

        for row in rows:
            shifted_len = len(str(row["prompt"]).encode("utf-8")) + len(
                str(row["intelligence_answer"]).encode("utf-8")
            ) - 1
            self.assertLessEqual(shifted_len, 384, row["id"])


if __name__ == "__main__":
    unittest.main()
