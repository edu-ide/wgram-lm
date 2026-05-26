from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "587_build_stage101u_causal_evidence_chain_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101u_causal_evidence_chain_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101UCausalEvidenceChainProbeTests(unittest.TestCase):
    def test_rows_force_stepwise_causal_evidence_chain_per_case(self) -> None:
        module = load_module()
        rows = module.causal_evidence_chain_rows()

        self.assertTrue(rows)
        self.assertTrue(all(row["stage101u_causal_evidence_chain_required"] for row in rows))

        expected_steps = {
            "source_role",
            "source_reliability",
            "evidence_relevance",
            "claim_support",
            "evidence_sufficiency",
            "numeric_belief_support",
            "numeric_belief_reliability",
            "numeric_belief_sufficiency",
        }
        by_case: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            if row["source_case_type"] == "causal_dictionary_readback":
                continue
            by_case[str(row["causal_case_id"])].add(str(row["stage101u_chain_step"]))

        self.assertTrue(by_case)
        self.assertTrue(all(expected_steps.issubset(steps) for steps in by_case.values()))

    def test_counterfactual_pairs_change_one_causal_factor_and_change_only_its_child_answer(self) -> None:
        module = load_module()
        rows = module.causal_evidence_chain_rows()
        by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            pair_id = str(row.get("causal_pair_id", ""))
            if pair_id and row["stage101u_chain_step"] == "source_reliability":
                by_pair[pair_id].append(row)

        self.assertTrue(by_pair)
        source_pair = next(values for pair_id, values in by_pair.items() if "source_quality" in pair_id)
        self.assertEqual(2, len(source_pair))
        self.assertEqual({row["source_claim"] for row in source_pair}, {source_pair[0]["source_claim"]})
        self.assertEqual({row["evidence_payload"] for row in source_pair}, {source_pair[0]["evidence_payload"]})
        self.assertEqual({" 0.90", " 0.10"}, {row["intelligence_answer"] for row in source_pair})

    def test_numeric_belief_rows_depend_on_parent_chain_steps(self) -> None:
        module = load_module()
        rows = module.causal_evidence_chain_rows()
        numeric_rows = [row for row in rows if str(row["stage101u_chain_step"]).startswith("numeric_belief_")]

        self.assertTrue(numeric_rows)
        for row in numeric_rows:
            parents = row["causal_parent_steps"]
            self.assertIn("source_reliability", parents)
            self.assertIn("evidence_relevance", parents)
            self.assertIn("claim_support", parents)
            self.assertIn("evidence_sufficiency", parents)
            self.assertIn("Causal chain:", row["prompt"])

    def test_build_writes_train_heldout_and_report_causal_contract(self) -> None:
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
        self.assertTrue(all(row["stage101u_causal_evidence_chain_required"] for row in train_rows))
        self.assertTrue(all(row["stage101u_causal_evidence_chain_required"] for row in heldout_rows))
        self.assertEqual(report["train_rows"], len(train_rows))
        self.assertEqual(report["eval_rows"], len(heldout_rows))
        self.assertGreaterEqual(report["causal_contract"]["source_quality_counterfactual_pairs"], 1)

    def test_probe_rows_fit_stage101_seq_len_384(self) -> None:
        module = load_module()
        rows = module.causal_evidence_chain_rows() + module.causal_evidence_chain_heldout_rows()

        for row in rows:
            shifted_len = len(str(row["prompt"]).encode("utf-8")) + len(
                str(row["intelligence_answer"]).encode("utf-8")
            ) - 1
            self.assertLessEqual(shifted_len, 384, row["id"])


if __name__ == "__main__":
    unittest.main()
