from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def _load_script():
    path = Path("scripts/402_score_m7b_core_depth_gate.py")
    spec = importlib.util.spec_from_file_location("m7b_core_depth_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_report(path: Path, *, accuracy: float, histogram: dict[str, int]) -> None:
    path.write_text(
        json.dumps(
            {
                "metrics": {
                    "cases": 64,
                    "accuracy": accuracy,
                    "invalid_pred_rate": 0.0,
                    "prompt_echo_rate": 0.0,
                    "pred_answer_histogram": histogram,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )


class M7BCoreDepthGateTests(unittest.TestCase):
    def test_accepts_depth_gain_with_clean_surface(self):
        module = _load_script()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = root / "full.json"
            base = root / "base.json"
            shallow = root / "shallow.json"
            out = root / "gate.json"
            _write_report(full, accuracy=0.15, histogram={"B": 20, "C": 20, "D": 24})
            _write_report(base, accuracy=0.09, histogram={"A": 64})
            _write_report(shallow, accuracy=0.11, histogram={"B": 40, "C": 24})

            report = module.score_gate(
                module.argparse.Namespace(
                    full_report=str(full),
                    baseline_report=str(base),
                    shallow_report=[str(shallow)],
                    out_json=str(out),
                    min_cases=64,
                    min_full_accuracy=0.0,
                    min_gain_vs_baseline=0.03,
                    min_gain_vs_best_shallow=0.03,
                    max_invalid_pred_rate=0.05,
                    max_prompt_echo_rate=0.05,
                    max_pred_fraction=0.60,
                )
            )

        self.assertTrue(report["accepted"])
        self.assertEqual(report["decision"], "accepted_m7b_public_mcq_core_depth_gate")

    def test_rejects_single_label_collapse(self):
        module = _load_script()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = root / "full.json"
            base = root / "base.json"
            out = root / "gate.json"
            _write_report(full, accuracy=0.15, histogram={"D": 64})
            _write_report(base, accuracy=0.09, histogram={"A": 64})

            report = module.score_gate(
                module.argparse.Namespace(
                    full_report=str(full),
                    baseline_report=str(base),
                    shallow_report=[],
                    out_json=str(out),
                    min_cases=64,
                    min_full_accuracy=0.0,
                    min_gain_vs_baseline=0.03,
                    min_gain_vs_best_shallow=0.03,
                    max_invalid_pred_rate=0.05,
                    max_prompt_echo_rate=0.05,
                    max_pred_fraction=0.60,
                )
            )

        self.assertFalse(report["accepted"])
        self.assertIn("max_pred_fraction_le_max", report["reject_reasons"])


if __name__ == "__main__":
    unittest.main()
