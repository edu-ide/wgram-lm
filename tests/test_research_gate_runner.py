import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "300_research_gate_runner.py"
    spec = importlib.util.spec_from_file_location("research_gate_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ResearchGateRunnerTests(unittest.TestCase):
    def test_list_gates_includes_donorless_depth_gate(self):
        module = _load_module()

        gates = module.list_gates("smoke")

        by_name = {gate["name"]: gate for gate in gates}
        self.assertEqual(by_name["donorless_recurrent_depth"]["target_level"], "L1 scaffold")
        self.assertEqual(by_name["prompt_source_position_binder"]["target_level"], "L1 scaffold")
        self.assertEqual(by_name["prompt_source_position_binder_numeric"]["target_level"], "L1 scaffold")
        self.assertEqual(
            by_name["prompt_source_position_binder_token_plus_numeric"]["target_level"],
            "L1 scaffold",
        )
        self.assertEqual(by_name["qtrm_minimal_depth"]["target_level"], "L2 local gate")
        self.assertEqual(by_name["qtrm_source_pointer_state"]["target_level"], "L2 local gate")
        self.assertEqual(by_name["qtrm_numeric_source_pointer_state"]["target_level"], "L2 local gate")
        self.assertEqual(
            by_name["qtrm_token_numeric_source_pointer_state"]["target_level"],
            "L2 local gate",
        )
        self.assertEqual(by_name["renderer_canonical_lm"]["target_level"], "L3 candidate")
        self.assertEqual(
            by_name["small_general_reasoning"]["target_level"],
            "L2 local gate / L3 candidate",
        )

    def test_gate_command_uses_python_script_and_out_dir(self):
        module = _load_module()
        gate = module.gate_specs("smoke")["donorless_recurrent_depth"]

        command = module.gate_command(gate, "local_eval/example")

        self.assertIn("scripts/260_train_donorless_recurrent_depth_probe.py", command)
        self.assertIn("--out-dir", command)
        self.assertIn("local_eval/example", command)
        self.assertIn("--steps", command)

    def test_source_pointer_state_gate_uses_refresh_script(self):
        module = _load_module()
        gate = module.gate_specs("smoke")["qtrm_source_pointer_state"]

        command = module.gate_command(gate, "local_eval/source_pointer")

        self.assertIn("scripts/319_run_qtrm_source_pointer_state_gate.py", command)
        self.assertIn("--out-dir", command)
        self.assertIn("local_eval/source_pointer", command)
        self.assertIn("--min-value-drop", command)

    def test_numeric_source_pointer_state_gate_enables_numeric_ablation(self):
        module = _load_module()
        gate = module.gate_specs("smoke")["qtrm_numeric_source_pointer_state"]

        command = module.gate_command(gate, "local_eval/numeric_source_pointer")

        self.assertIn("scripts/319_run_qtrm_source_pointer_state_gate.py", command)
        self.assertIn("--numeric-source-features", command)
        self.assertIn("--min-numeric-value-drop", command)
        self.assertIn("local_eval/numeric_source_pointer", command)

    def test_token_numeric_source_pointer_state_gate_enables_token_numeric_ablation(self):
        module = _load_module()
        gate = module.gate_specs("smoke")["qtrm_token_numeric_source_pointer_state"]

        command = module.gate_command(gate, "local_eval/token_numeric_source_pointer")

        self.assertIn("scripts/319_run_qtrm_source_pointer_state_gate.py", command)
        self.assertIn("--token-numeric-value-features", command)
        self.assertIn("--min-token-numeric-value-drop", command)
        self.assertIn("local_eval/token_numeric_source_pointer", command)

    def test_prompt_source_position_binder_gate_uses_binder_script(self):
        module = _load_module()
        gate = module.gate_specs("smoke")["prompt_source_position_binder"]

        command = module.gate_command(gate, "local_eval/prompt_binder")

        self.assertIn("scripts/320_train_prompt_source_position_binder_probe.py", command)
        self.assertIn("--train-jsonl", command)
        self.assertIn("--eval-jsonl", command)
        self.assertIn("local_eval/prompt_binder", command)

    def test_numeric_source_position_binder_gate_uses_numeric_input(self):
        module = _load_module()
        gate = module.gate_specs("smoke")["prompt_source_position_binder_numeric"]

        command = module.gate_command(gate, "local_eval/numeric_binder")

        self.assertIn("scripts/320_train_prompt_source_position_binder_probe.py", command)
        self.assertIn("--input-source", command)
        self.assertIn("numeric_value_embedding", command)
        self.assertIn("local_eval/numeric_binder", command)

    def test_token_plus_numeric_source_position_binder_gate_uses_canonical_token_input(self):
        module = _load_module()
        gate = module.gate_specs("smoke")[
            "prompt_source_position_binder_token_plus_numeric"
        ]

        command = module.gate_command(gate, "local_eval/token_plus_numeric_binder")

        self.assertIn("scripts/320_train_prompt_source_position_binder_probe.py", command)
        self.assertIn("--input-source", command)
        self.assertIn("token_plus_numeric_value", command)
        self.assertIn("local_eval/token_plus_numeric_binder", command)

    def test_normalize_and_accept_decision(self):
        module = _load_module()
        gate = module.gate_specs("smoke")["donorless_recurrent_depth"]

        self.assertEqual(module.normalize_decision({"decision": "accepted_l1"}), "accepted_l1")
        self.assertTrue(module.is_accepted({"decision": "accepted_l1"}, gate))
        self.assertFalse(module.is_accepted({"decision": "rejected"}, gate))

    def test_dry_run_writes_gate_summary(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            summary = module.run_gate(
                gate_name="donorless_recurrent_depth",
                profile="smoke",
                out_root=tmp,
                dry_run=True,
            )
            summary_path = Path(summary["out_dir"]) / "gate_summary.json"
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["decision"], "dry_run")
        self.assertFalse(summary["accepted"])
        self.assertEqual(loaded["gate"], "donorless_recurrent_depth")

    def test_skip_existing_report_reuses_decision(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "existing"
            out_dir.mkdir()
            (out_dir / "report.json").write_text(
                json.dumps(
                    {
                        "decision": "accepted_l1",
                        "eval_metrics": {"depth8_final_exact": 1.0, "depth1_final_exact": 0.0},
                        "ablations": {"state_reset": {"depth8_final_exact": 0.0}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = module.run_gate(
                gate_name="donorless_recurrent_depth",
                profile="smoke",
                out_root=tmp,
                out_dir=out_dir,
                skip_existing=True,
            )

        self.assertTrue(summary["accepted"])
        self.assertEqual(summary["decision"], "accepted_l1")
        self.assertEqual(summary["decisive_metrics"]["eval_metrics.depth8_final_exact"], 1.0)


if __name__ == "__main__":
    unittest.main()
