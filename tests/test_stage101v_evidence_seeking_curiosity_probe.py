from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "588_build_stage101v_evidence_seeking_curiosity_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101v_evidence_seeking_curiosity_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101VEvidenceSeekingCuriosityProbeTests(unittest.TestCase):
    def test_rows_force_answer_or_ask_decision_after_causal_chain(self) -> None:
        module = load_module()
        rows = module.evidence_seeking_curiosity_rows()

        self.assertTrue(rows)
        self.assertTrue(all(row["stage101v_evidence_seeking_curiosity_required"] for row in rows))

        by_case: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            by_case[str(row["curiosity_case_id"])].add(str(row["stage101v_chain_step"]))

        self.assertTrue(by_case)
        expected = {
            "answer_policy",
            "evidence_request",
            "curiosity_reason",
        }
        self.assertTrue(all(expected.issubset(steps) for steps in by_case.values()))

    def test_source_quality_counterfactual_flips_answer_policy(self) -> None:
        module = load_module()
        rows = [
            row
            for row in module.evidence_seeking_curiosity_rows()
            if row["stage101v_chain_step"] == "answer_policy"
        ]
        by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            pair_id = str(row.get("curiosity_pair_id", ""))
            if pair_id.startswith("source_quality"):
                by_pair[pair_id].append(row)

        self.assertTrue(by_pair)
        pair_rows = next(iter(by_pair.values()))
        self.assertEqual(2, len(pair_rows))
        self.assertEqual({row["source_claim"] for row in pair_rows}, {pair_rows[0]["source_claim"]})
        self.assertEqual({row["evidence_payload"] for row in pair_rows}, {pair_rows[0]["evidence_payload"]})
        self.assertEqual({" answer_now", " ask_more"}, {row["intelligence_answer"] for row in pair_rows})

    def test_request_type_names_the_missing_evidence_not_generic_unknown(self) -> None:
        module = load_module()
        rows = [
            row
            for row in module.evidence_seeking_curiosity_rows()
            if row["stage101v_chain_step"] == "evidence_request"
            and row["intelligence_answer"] != " no_more_evidence"
        ]

        self.assertTrue(rows)
        request_answers = {row["intelligence_answer"] for row in rows}
        self.assertIn(" ask_reliable_source", request_answers)
        self.assertIn(" ask_relevant_evidence", request_answers)
        self.assertIn(" ask_exact_detail", request_answers)
        self.assertTrue(all("Causal curiosity chain:" in row["prompt"] for row in rows))

    def test_build_writes_train_heldout_and_report(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            heldout = root / "heldout.jsonl"
            report_out = root / "report.json"
            args = module.build_arg_parser().parse_args(
                [
                    "--train-out",
                    str(train),
                    "--eval-out",
                    str(heldout),
                    "--report-out",
                    str(report_out),
                ]
            )
            report = module.build(args)
            train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
            heldout_rows = [json.loads(line) for line in heldout.read_text(encoding="utf-8").splitlines()]
            saved_report = json.loads(report_out.read_text(encoding="utf-8"))

        self.assertEqual(report, saved_report)
        self.assertEqual(report["train_rows"], len(train_rows))
        self.assertEqual(report["eval_rows"], len(heldout_rows))
        self.assertGreaterEqual(report["curiosity_contract"]["ask_more_rows"], 1)
        self.assertGreaterEqual(report["curiosity_contract"]["answer_now_rows"], 1)

    def test_probe_rows_fit_stage101_seq_len_384(self) -> None:
        module = load_module()
        rows = module.evidence_seeking_curiosity_rows() + module.evidence_seeking_curiosity_heldout_rows()

        for row in rows:
            shifted_len = len(str(row["prompt"]).encode("utf-8")) + len(
                str(row["intelligence_answer"]).encode("utf-8")
            ) - 1
            self.assertLessEqual(shifted_len, 384, row["id"])


if __name__ == "__main__":
    unittest.main()
