import json
import tempfile
import unittest
from pathlib import Path


class AsiCausalLoopGateScriptTests(unittest.TestCase):
    def test_script_writes_markdown_and_json_gate(self):
        import importlib.util

        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "154_build_asi_causal_loop_gate.py"
        )
        spec = importlib.util.spec_from_file_location("asi_causal_gate_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.json"
            md_path = Path(tmp) / "gate.md"
            json_path = Path(tmp) / "gate.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "scripted_harness": 0.70,
                        "donor_harness": 0.72,
                        "qtrm_harness": 0.78,
                        "qtrm_latent_core_off": 0.72,
                        "qtrm_world_model_off": 0.73,
                        "qtrm_verifier_off": 0.70,
                    }
                ),
                encoding="utf-8",
            )

            gate = module.write_gate_report(
                str(metrics_path),
                markdown_out=str(md_path),
                json_out=str(json_path),
                min_gain=0.02,
                min_drop=0.03,
            )

            self.assertEqual(gate["status"], "accepted")
            self.assertIn("# ASI Causal Loop Gate", md_path.read_text(encoding="utf-8"))
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8"))["status"],
                "accepted",
            )


if __name__ == "__main__":
    unittest.main()
