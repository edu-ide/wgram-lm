from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "603_build_stage101z_source_binding_ledger_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101z_source_binding_ledger_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101ZSourceBindingLedgerProbeTests(unittest.TestCase):
    def test_rows_separate_source_id_from_content(self) -> None:
        module = load_module()
        rows = module.source_binding_ledger_rows()

        self.assertEqual(32, len(rows))
        for row in rows:
            self.assertIn("Evidence source:", row["original_prompt"])
            self.assertIn("Evidence value:", row["original_prompt"])
            self.assertIn("Evidence source:", row["counterfactual_prompt"])
            self.assertIn("Evidence value:", row["counterfactual_prompt"])
            self.assertNotIn("says", row["original_prompt"])
            self.assertNotEqual(row["original_answer"], row["counterfactual_answer"])

    def test_contract_balances_bindings(self) -> None:
        module = load_module()
        contract = module.source_binding_ledger_contract(module.source_binding_ledger_rows())

        self.assertEqual({" yes": 16, " no": 16}, contract["original_answer_counts"])
        self.assertEqual({"S1": 16, "S2": 16}, contract["original_source_counts"])
        self.assertEqual({"source_reliability": 32}, contract["pair_feature_counts"])


if __name__ == "__main__":
    unittest.main()
