from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "602_build_stage101y_provenance_ledger_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101y_provenance_ledger_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101YProvenanceLedgerProbeTests(unittest.TestCase):
    def test_rows_use_ledger_and_hide_feature_targets(self) -> None:
        module = load_module()
        rows = module.provenance_ledger_rows()

        self.assertEqual(32, len(rows))
        for row in rows:
            self.assertEqual("source_reliability", row["pair_feature"])
            self.assertIn("Source ledger:", row["original_prompt"])
            self.assertIn("Source ledger:", row["counterfactual_prompt"])
            self.assertNotIn("trusted source", row["original_prompt"].lower())
            self.assertNotIn("untrusted source", row["original_prompt"].lower())
            self.assertNotIn("feature_targets", row)
            self.assertNotEqual(row["original_answer"], row["counterfactual_answer"])

    def test_contract_balances_source_ids_and_answers(self) -> None:
        module = load_module()
        rows = module.provenance_ledger_rows()
        contract = module.provenance_ledger_contract(rows)

        self.assertEqual({" yes": 16, " no": 16}, contract["original_answer_counts"])
        self.assertEqual({" yes": 16, " no": 16}, contract["counterfactual_answer_counts"])
        self.assertEqual({"S1": 16, "S2": 16}, contract["original_source_counts"])
        self.assertEqual({"S1": 16, "S2": 16}, contract["counterfactual_source_counts"])


if __name__ == "__main__":
    unittest.main()
