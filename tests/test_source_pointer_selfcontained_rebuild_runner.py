import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def load_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "350_rebuild_source_pointer_selfcontained_stack.py"
    )
    spec = importlib.util.spec_from_file_location(
        "source_pointer_selfcontained_rebuild",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SourcePointerSelfContainedRebuildRunnerTests(unittest.TestCase):
    def test_dry_run_plans_self_contained_l2_l3_rebuild_chain(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--dry-run",
                "--profile",
                "smoke",
                "--out-dir",
                "local_eval/selfcontained_rebuild_test",
            ]
        )

        report = module.run(args)

        self.assertEqual("dry_run", report["decision"])
        self.assertFalse(report["accepted"])
        stage_names = [stage["name"] for stage in report["stages"]]
        self.assertEqual(
            [
                "base_train",
                "l2_gate",
                "l2_materialize",
                "l3_tune",
                "l3_audit",
                "l3_materialize",
            ],
            stage_names,
        )
        base_checkpoint = "local_eval/selfcontained_rebuild_test/00_base/last.pt"
        l2_materialized = (
            "local_eval/selfcontained_rebuild_test/02_l2_self_contained/"
            "accepted_l2_self_contained.pt"
        )
        l2_command = report["stages"][1]["command"]
        l3_tune_command = report["stages"][3]["command"]
        self.assertIn("--allow-random-init", report["stages"][0]["command"])
        self.assertEqual(base_checkpoint, l2_command[l2_command.index("--init-checkpoint") + 1])
        self.assertEqual(
            l2_materialized,
            l3_tune_command[l3_tune_command.index("--init-checkpoint") + 1],
        )
        self.assertIn("--keep-rejected-checkpoints", l3_tune_command)
        self.assertIn("--batch-integrated-training", l2_command)
        self.assertIn("--token-numeric-source-slots", l2_command)
        self.assertIn("--core-source-position-binder-source-slots-only", l2_command)
        self.assertIn("--core-source-position-binder-raw-source-slots", l2_command)
        self.assertIn("--strict-prompt-binding-ablation", l2_command)
        l2_materialize_command = report["stages"][2]["command"]
        self.assertIn("scripts/329_materialize_qtrm_checkpoint_stack.py", l2_materialize_command)
        self.assertIn("--fail-on-unmatched-keys", l2_materialize_command)
        self.assertIn("--token-numeric-source-slots", l2_materialize_command)
        self.assertIn("--core-source-position-binder", l2_materialize_command)
        self.assertIn("scripts/321_run_source_pointer_l3_hard_gate.py", report["stages"][4]["command"])

    def test_l3_audit_uses_final_candidate_checkpoint_from_l3_tune(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--dry-run",
                "--out-dir",
                "out",
                "--l3-save-every",
                "50",
                "--l3-steps",
                "150",
            ]
        )

        plan = module.build_plan(args)
        l3_audit = next(stage for stage in plan if stage["name"] == "l3_audit")

        self.assertIn("out/03_l3_tune/train/step_000150.pt", l3_audit["command"])

    def test_triage_profile_uses_intermediate_budget_between_smoke_and_standard(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--dry-run",
                "--profile",
                "triage",
                "--out-dir",
                "out",
            ]
        )

        plan = module.build_plan(args)
        base = next(stage for stage in plan if stage["name"] == "base_train")
        l2 = next(stage for stage in plan if stage["name"] == "l2_gate")
        l3 = next(stage for stage in plan if stage["name"] == "l3_tune")

        self.assertEqual("30", base["command"][base["command"].index("--steps") + 1])
        self.assertEqual("15", base["command"][base["command"].index("--save-every") + 1])
        self.assertEqual("60", l2["command"][l2["command"].index("--steps") + 1])
        self.assertEqual("32", l2["command"][l2["command"].index("--max-eval-cases") + 1])
        self.assertEqual("8", l2["command"][l2["command"].index("--row-batch-size") + 1])
        self.assertEqual("80", l3["command"][l3["command"].index("--steps") + 1])

    def test_l3_audit_uses_source_slot_causal_flags(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--dry-run",
                "--out-dir",
                "out",
            ]
        )

        plan = module.build_plan(args)
        l3_audit = next(stage for stage in plan if stage["name"] == "l3_audit")
        command = l3_audit["command"]

        self.assertIn("--token-numeric-source-slots", command)
        self.assertIn("--core-source-position-binder-source-slots-only", command)
        self.assertIn("--core-source-position-binder-raw-source-slots", command)
        self.assertEqual("0.25", command[command.index("--min-token-numeric-value-drop") + 1])
        self.assertEqual("0.25", command[command.index("--min-source-binder-value-drop") + 1])

    def test_l2_reject_appends_autoresearch_style_operation_ledger(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_dir = tmp_path / "run"
            ledger = tmp_path / "results.tsv"
            args = module.build_arg_parser().parse_args(
                [
                    "--profile",
                    "smoke",
                    "--out-dir",
                    str(out_dir),
                    "--operation-ledger",
                    str(ledger),
                ]
            )
            original_run_command = module.run_command
            original_load_report = module._load_report

            def fake_run_command(command, *, cwd, env, out_dir):
                return 0

            def fake_load_report(path):
                if str(path).endswith("01_l2_gate/report.json"):
                    return {
                        "decision": "rejected",
                        "accepted": False,
                        "full_trace_exact_accuracy": 0.0,
                        "full_value_accuracy": 0.3,
                        "value_drop": 0.3,
                    }
                return {}

            try:
                module.run_command = fake_run_command
                module._load_report = fake_load_report
                report = module.run(args)
            finally:
                module.run_command = original_run_command
                module._load_report = original_load_report

            self.assertEqual("l2_rejected", report["decision"])
            self.assertEqual(
                {
                    "full_trace_exact_accuracy": 0.0,
                    "full_value_accuracy": 0.3,
                    "value_drop": 0.3,
                },
                report["decisive_metrics"],
            )
            rows = ledger.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(2, len(rows))
            self.assertIn("timestamp\tgate\tprofile\tdecision\tstatus", rows[0])
            self.assertIn("source_pointer_selfcontained_stack", rows[1])
            self.assertIn("\tl2_rejected\tdiscard\tfull_trace_exact_accuracy\t0.0\t", rows[1])
            saved_report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual("stop L4 work; recover L2 source-pointer gate before rebuilding L3", saved_report["next_action"])


if __name__ == "__main__":
    unittest.main()
