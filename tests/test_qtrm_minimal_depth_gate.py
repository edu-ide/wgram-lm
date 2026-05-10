import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "301_build_qtrm_minimal_depth_gate.py"
    spec = importlib.util.spec_from_file_location("qtrm_minimal_depth_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QTRMMinimalDepthGateTests(unittest.TestCase):
    def test_build_gate_report_accepts_full_over_coreoff_and_donor(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = root / "full.json"
            coreoff = root / "coreoff.json"
            donor = root / "donor.json"
            full.write_text(json.dumps({"summary": {"cases": 4, "answer_accuracy": 1.0}}))
            coreoff.write_text(json.dumps({"summary": {"cases": 4, "answer_accuracy": 0.0}}))
            donor.write_text(
                json.dumps(
                    {
                        "summary": {
                            "by_mode": {
                                "forced_choice": {"accuracy": 0.5},
                                "greedy": {"accuracy": 0.25},
                            }
                        }
                    }
                )
            )

            report = module.build_gate_report(
                full_json=full,
                core_off_json=coreoff,
                donor_json=donor,
                out_dir=root / "out",
                min_full_accuracy=0.95,
                min_core_off_drop=0.5,
                min_donor_gain=0.25,
            )

        self.assertEqual(report["decision"], "accepted_l2")
        self.assertTrue(report["accepted"])
        self.assertEqual(report["metrics"]["full_minus_core_off"], 1.0)
        self.assertEqual(report["metrics"]["full_minus_donor"], 0.5)

    def test_build_gate_report_rejects_weak_core_drop(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = root / "full.json"
            coreoff = root / "coreoff.json"
            full.write_text(json.dumps({"summary": {"cases": 4, "answer_accuracy": 1.0}}))
            coreoff.write_text(json.dumps({"summary": {"cases": 4, "answer_accuracy": 0.9}}))

            report = module.build_gate_report(
                full_json=full,
                core_off_json=coreoff,
                donor_json=None,
                out_dir=root / "out",
                min_full_accuracy=0.95,
                min_core_off_drop=0.5,
                min_donor_gain=0.25,
            )

        self.assertEqual(report["decision"], "rejected")
        self.assertIn("core_off_drop_below_min", report["reject_reasons"])


if __name__ == "__main__":
    unittest.main()
