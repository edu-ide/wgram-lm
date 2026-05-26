from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "606_build_stage102c_randomized_trust_ledger_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage102c_randomized_trust_ledger_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage102CRandomizedTrustLedgerProbeTests(unittest.TestCase):
    def test_rows_randomize_verified_source_identity(self) -> None:
        module = load_module()
        rows = module.randomized_trust_ledger_rows()

        self.assertEqual(64, len(rows))
        contract = module.randomized_trust_ledger_contract(rows)
        self.assertEqual({"S1": 32, "S2": 32}, contract["verified_source_counts"])
        self.assertEqual({"S1": 32, "S2": 32}, contract["original_source_counts"])
        self.assertEqual({" yes": 32, " no": 32}, contract["original_answer_counts"])

    def test_prompt_ledger_role_matches_answer_not_source_name(self) -> None:
        module = load_module()
        rows = module.randomized_trust_ledger_rows()
        s2_verified_rows = [row for row in rows if row["verified_source"] == "S2"]

        self.assertTrue(s2_verified_rows)
        for row in s2_verified_rows:
            self.assertIn("S2 = verified", row["original_prompt"])
            self.assertIn("S1 = unverified", row["original_prompt"])
            if row["original_source"] == "S2":
                self.assertEqual(" yes", row["original_answer"])
            else:
                self.assertEqual(" no", row["original_answer"])


if __name__ == "__main__":
    unittest.main()
