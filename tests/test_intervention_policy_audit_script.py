from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def _load_module():
    path = Path("scripts/179_audit_intervention_policy.py")
    spec = importlib.util.spec_from_file_location("intervention_policy_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InterventionPolicyAuditScriptTests(unittest.TestCase):
    def test_counts_donor_and_core_off_disagreements(self) -> None:
        module = _load_module()
        records = [
            {
                "id": "a",
                "mode": "donor_only_with_evidence",
                "hit": True,
                "completion": "Answer: opal-river",
            },
            {
                "id": "a",
                "mode": "qtrm_residual_with_evidence",
                "hit": False,
                "completion": "Answer: stone-arch",
            },
            {
                "id": "a",
                "mode": "qtrm_core_off_with_evidence",
                "hit": True,
                "completion": "Answer: opal-river",
            },
            {
                "id": "b",
                "mode": "donor_only_with_evidence",
                "hit": False,
                "completion": "Answer: private",
            },
            {
                "id": "b",
                "mode": "qtrm_residual_with_evidence",
                "hit": True,
                "completion": "Answer: UNKNOWN",
            },
            {
                "id": "b",
                "mode": "qtrm_core_off_with_evidence",
                "hit": False,
                "completion": "Answer: private",
            },
        ]

        summary = module.build_intervention_audit(records)

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["donor_hit_qtrm_miss_count"], 1)
        self.assertEqual(summary["qtrm_hit_donor_miss_count"], 1)
        self.assertEqual(summary["core_off_beats_qtrm_count"], 1)
        self.assertEqual(summary["qtrm_beats_core_off_count"], 1)
        self.assertEqual(summary["qtrm_changed_donor_completion_count"], 2)
        self.assertEqual(summary["donor_hit_qtrm_miss_cases"][0]["id"], "a")

    def test_cli_writes_summary_json(self) -> None:
        module = _load_module()
        rows = [
            {"id": "a", "mode": "donor_only_with_evidence", "hit": True, "completion": "A"},
            {"id": "a", "mode": "qtrm_residual_with_evidence", "hit": False, "completion": "B"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "eval.jsonl"
            out_path = Path(tmp) / "summary.json"
            in_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            module.main(["--eval-jsonl", str(in_path), "--out-json", str(out_path)])

            written = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(written["case_count"], 1)
            self.assertEqual(written["donor_hit_qtrm_miss_count"], 1)


if __name__ == "__main__":
    unittest.main()
