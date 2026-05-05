import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def load_gate_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "191_build_raw_intelligence_gate.py"
    spec = importlib.util.spec_from_file_location("raw_intelligence_gate_script", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RawIntelligenceGateScriptTests(unittest.TestCase):
    def test_script_writes_markdown_and_json_for_pure_reasoning(self):
        module = load_gate_script()

        records = [
            {"id": "case-a", "mode": "donor_only_no_evidence", "hit": False},
            {"id": "case-a", "mode": "qtrm_core_off_no_evidence", "hit": False},
            {"id": "case-a", "mode": "qtrm_core_steps_1_no_evidence", "hit": False},
            {"id": "case-a", "mode": "qtrm_core_steps_4_no_evidence", "hit": True},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            eval_path = Path(tmp) / "eval.jsonl"
            md_path = Path(tmp) / "gate.md"
            json_path = Path(tmp) / "gate.json"
            eval_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            gate = module.write_gate_report(
                [str(eval_path)],
                gate_type="pure_recursive_reasoning",
                markdown_out=str(md_path),
                json_out=str(json_path),
            )

            markdown = md_path.read_text(encoding="utf-8")
            summary = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(gate["status"], "accepted")
        self.assertEqual(summary["status"], "accepted")
        self.assertIn("mode_semantics", summary)
        self.assertIn("eval_contract", summary)
        self.assertIn("Raw Intelligence Gate", markdown)
        self.assertIn("Mode Semantics", markdown)
        self.assertIn("donor fallback is not forced", markdown)
        self.assertIn("pure_recursive_reasoning", markdown)

    def test_cli_exposes_gate_type(self):
        module = load_gate_script()

        args = module.build_arg_parser().parse_args(
            ["--gate-type", "temporal_spatial_context"]
        )

        self.assertEqual(args.gate_type, "temporal_spatial_context")


if __name__ == "__main__":
    unittest.main()
