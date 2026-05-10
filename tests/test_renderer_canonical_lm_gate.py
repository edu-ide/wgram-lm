import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "302_build_renderer_canonical_lm_gate.py"
    spec = importlib.util.spec_from_file_location("renderer_canonical_lm_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RendererCanonicalLMGateTests(unittest.TestCase):
    def test_summarize_by_mode_counts_hits(self):
        module = _load_module()

        summary = module.summarize_by_mode(
            [
                {"mode": "full", "hit": True},
                {"mode": "full", "hit": False},
                {"mode": "core_off", "hit": False},
            ]
        )

        self.assertEqual(summary["full"]["exact"], "1/2")
        self.assertEqual(summary["core_off"]["accuracy"], 0.0)

    def test_build_gate_report_rejects_zero_generation(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {"mode": "qtrm_core_steps_8_no_evidence", "hit": False},
                {"mode": "qtrm_core_off_no_evidence", "hit": False},
                {"mode": "donor_only_no_evidence", "hit": False},
            ]
            path = root / "generation.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            report = module.build_gate_report(
                generation_jsonl=path,
                out_dir=root / "out",
                full_mode="qtrm_core_steps_8_no_evidence",
                core_off_mode="qtrm_core_off_no_evidence",
                donor_mode="donor_only_no_evidence",
                ablation_mode="qtrm_core_steps_8_answer_next_token_decoder_off_no_evidence",
                min_full_accuracy=0.5,
                min_core_off_drop=0.25,
                min_ablation_drop=0.25,
            )

        self.assertEqual(report["decision"], "rejected")
        self.assertIn("full_generation_accuracy_below_min", report["reject_reasons"])


if __name__ == "__main__":
    unittest.main()
