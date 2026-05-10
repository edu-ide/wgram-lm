import importlib.util
from pathlib import Path
import sys
import unittest


def load_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "319_run_qtrm_source_pointer_state_gate.py"
    )
    spec = importlib.util.spec_from_file_location("qtrm_source_pointer_state_gate", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QTRMSourcePointerStateGateTests(unittest.TestCase):
    def test_training_command_uses_source_position_and_prompt_initialization(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            ["--out-dir", "out", "--steps", "9", "--save-every", "9"]
        )

        command = module.training_command(args, train_out_dir=Path("out/train"))
        text = " ".join(command)

        self.assertIn("--role-value-list-class-mode source_position", text)
        self.assertIn("--core-role-value-prompt-ce-weight 1.0", text)
        self.assertIn("--core-role-value-prompt-target-mode initial", text)
        self.assertIn("--core-primitive-role-value-state-ce-weight 1.0", text)
        self.assertIn("--save-trainable-only", command)

    def test_training_command_can_enable_numeric_source_features(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--out-dir",
                "out",
                "--numeric-source-features",
                "--numeric-source-max-list-len",
                "7",
                "--numeric-source-value-vocab-size",
                "64",
            ]
        )

        command = module.training_command(args, train_out_dir=Path("out/train"))
        text = " ".join(command)

        self.assertIn("--numeric-source-features", command)
        self.assertIn("--numeric-source-max-list-len 7", text)
        self.assertIn("--numeric-source-value-vocab-size 64", text)
        self.assertIn("--trainable-param-policy", command)
        self.assertIn("numeric_projector_primitive_role_value_state_machine", command)

    def test_training_command_can_enable_token_numeric_value_features(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--out-dir",
                "out",
                "--token-numeric-value-features",
                "--token-numeric-value-vocab-size",
                "64",
            ]
        )

        command = module.training_command(args, train_out_dir=Path("out/train"))
        text = " ".join(command)

        self.assertIn("--token-numeric-value-features", command)
        self.assertIn("--token-numeric-value-vocab-size 64", text)
        self.assertIn("--trainable-param-policy", command)
        self.assertIn("token_numeric_context_primitive_role_value_state_machine", command)

    def test_training_command_can_enable_token_numeric_internal_binder(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--out-dir",
                "out",
                "--token-numeric-value-features",
                "--core-source-position-binder",
                "--core-source-position-binder-gate-min",
                "1.0",
                "--core-source-position-binder-state-gate-min",
                "0.25",
                "--core-source-position-binder-state-st",
                "--core-source-position-binder-query-state",
                "--core-source-position-binder-query-state-gate-min",
                "0.5",
                "--core-source-value-binder",
                "--core-source-value-binder-state-gate-min",
                "0.75",
                "--core-source-value-binder-state-st",
                "--core-source-value-prompt-ce-weight",
                "0.8",
                "--core-primitive-role-value-source-value-conditioning",
                "--core-primitive-role-value-source-value-gate-min",
                "0.6",
            ]
        )

        command = module.training_command(args, train_out_dir=Path("out/train"))
        text = " ".join(command)

        self.assertIn("--core-source-position-binder", command)
        self.assertIn("--core-source-position-binder-gate-min 1.0", text)
        self.assertIn("--core-source-position-binder-state-gate-min 0.25", text)
        self.assertIn("--core-source-position-binder-state-st", command)
        self.assertIn("--core-source-position-binder-query-state", command)
        self.assertIn(
            "--core-source-position-binder-query-state-gate-min 0.5",
            text,
        )
        self.assertIn("--core-source-value-binder", command)
        self.assertIn("--core-source-value-binder-state-gate-min 0.75", text)
        self.assertIn("--core-source-value-binder-state-st", command)
        self.assertIn("--core-source-value-prompt-ce-weight 0.8", text)
        self.assertIn(
            "--core-primitive-role-value-source-value-conditioning",
            command,
        )
        self.assertIn(
            "--core-primitive-role-value-source-value-gate-min 0.6",
            text,
        )
        self.assertIn("--token-numeric-value-features", command)
        self.assertIn("--trainable-param-policy", command)
        self.assertIn(
            "token_numeric_context_binder_primitive_role_value_state_machine",
            text,
        )

    def test_training_command_can_enable_prompt_only_internal_binder(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--out-dir",
                "out",
                "--core-source-position-binder",
                "--core-source-position-binder-gate-min",
                "1.0",
                "--core-source-position-binder-state-gate-min",
                "0.25",
                "--core-source-position-binder-state-st",
                "--core-source-position-binder-query-state",
                "--core-source-position-binder-query-state-gate-min",
                "0.5",
                "--core-source-value-binder",
                "--core-source-value-binder-state-gate-min",
                "0.75",
                "--core-source-value-binder-state-st",
                "--core-source-value-prompt-ce-weight",
                "0.8",
                "--core-primitive-role-value-source-value-conditioning",
                "--core-primitive-role-value-source-value-gate-min",
                "0.6",
            ]
        )

        command = module.training_command(args, train_out_dir=Path("out/train"))
        text = " ".join(command)

        self.assertIn("--core-source-position-binder", command)
        self.assertIn("--core-source-position-binder-gate-min 1.0", text)
        self.assertIn("--core-source-position-binder-state-gate-min 0.25", text)
        self.assertIn("--core-source-position-binder-state-st", command)
        self.assertIn("--core-source-position-binder-query-state", command)
        self.assertIn(
            "--core-source-position-binder-query-state-gate-min 0.5",
            text,
        )
        self.assertIn("--core-source-value-binder", command)
        self.assertIn("--core-source-value-binder-state-gate-min 0.75", text)
        self.assertIn("--core-source-value-binder-state-st", command)
        self.assertIn("--core-source-value-prompt-ce-weight 0.8", text)
        self.assertIn(
            "--core-primitive-role-value-source-value-conditioning",
            command,
        )
        self.assertIn(
            "--core-primitive-role-value-source-value-gate-min 0.6",
            text,
        )
        self.assertIn("--trainable-param-policy", command)
        self.assertIn(
            "prompt_context_binder_primitive_role_value_state_machine",
            text,
        )

    def test_training_command_can_enable_token_numeric_source_slots(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--out-dir",
                "out",
                "--token-numeric-source-slots",
                "--token-numeric-source-slot-vocab-size",
                "64",
                "--token-numeric-source-slot-max-slots",
                "7",
                "--token-numeric-source-slot-id-mode",
                "relative_parity",
                "--token-numeric-source-slot-gate-min",
                "1.0",
                "--token-numeric-source-slot-parity-ce-weight",
                "0.7",
                "--token-numeric-source-slot-predicate-feedback",
                "--token-numeric-source-slot-predicate-ce-weight",
                "0.9",
                "--core-primitive-role-value-pair-trace-contrast-weight",
                "1.2",
                "--core-primitive-role-value-pair-trace-contrast-margin",
                "0.4",
                "--core-source-position-binder",
                "--core-source-position-binder-source-slots-only",
                "--core-source-position-binder-raw-source-slots",
                "--core-source-position-binder-state-gate-min",
                "0.25",
                "--core-source-position-binder-state-st",
            ]
        )

        command = module.training_command(args, train_out_dir=Path("out/train"))
        text = " ".join(command)

        self.assertIn("--token-numeric-source-slots", command)
        self.assertIn("--token-numeric-source-slot-vocab-size 64", text)
        self.assertIn("--token-numeric-source-slot-max-slots 7", text)
        self.assertIn("--token-numeric-source-slot-id-mode relative_parity", text)
        self.assertIn("--token-numeric-source-slot-gate-min 1.0", text)
        self.assertIn("--token-numeric-source-slot-parity-ce-weight 0.7", text)
        self.assertIn("--token-numeric-source-slot-predicate-feedback", command)
        self.assertIn("--token-numeric-source-slot-predicate-ce-weight 0.9", text)
        self.assertIn(
            "--core-primitive-role-value-pair-trace-contrast-weight 1.2",
            text,
        )
        self.assertIn(
            "--core-primitive-role-value-pair-trace-contrast-margin 0.4",
            text,
        )
        self.assertIn("--core-source-position-binder-source-slots-only", command)
        self.assertIn("--core-source-position-binder-raw-source-slots", command)
        self.assertIn("--core-source-position-binder-state-gate-min 0.25", text)
        self.assertIn("--core-source-position-binder-state-st", command)
        self.assertIn("--trainable-param-policy", command)
        self.assertIn(
            "token_numeric_source_slot_context_binder_primitive_role_value_state_machine",
            text,
        )

    def test_training_command_can_use_batch_integrated_trainer(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--out-dir",
                "out",
                "--batch-integrated-training",
                "--row-batch-size",
                "16",
                "--token-numeric-source-slots",
                "--token-numeric-source-slot-predicate-feedback",
                "--core-source-position-binder",
                "--core-source-position-binder-source-slots-only",
                "--core-source-position-binder-raw-source-slots",
            ]
        )

        command = module.training_command(args, train_out_dir=Path("out/train"))
        text = " ".join(command)

        self.assertIn("scripts/324_train_qtrm_source_pointer_batch.py", command)
        self.assertIn("--row-batch-size 16", text)
        self.assertIn("--token-numeric-source-slots", command)
        self.assertIn("--token-numeric-source-slot-predicate-feedback", command)
        self.assertIn("--core-source-position-binder", command)
        self.assertIn("--core-source-position-binder-source-slots-only", command)
        self.assertIn("--core-source-position-binder-raw-source-slots", command)

    def test_eval_command_uses_source_position_and_primitive_off_ablation(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--out-dir",
                "out",
                "--core-source-position-binder",
                "--core-source-position-binder-source-slots-only",
                "--core-source-position-binder-query-state",
                "--core-source-position-binder-query-state-gate-min",
                "0.5",
                "--core-source-value-binder",
                "--core-source-value-binder-state-gate-min",
                "0.75",
                "--core-source-value-binder-state-st",
                "--core-primitive-role-value-source-value-conditioning",
                "--core-primitive-role-value-source-value-gate-min",
                "0.6",
            ]
        )

        full = module.eval_command(
            args,
            checkpoint=Path("out/train/last.pt"),
            out_json=Path("out/full.json"),
            primitive_off=False,
            numeric_off=False,
            token_numeric_off=False,
            source_slot_off=False,
            source_binder_off=False,
        )
        off = module.eval_command(
            args,
            checkpoint=Path("out/train/last.pt"),
            out_json=Path("out/off.json"),
            primitive_off=True,
            numeric_off=False,
            token_numeric_off=False,
            source_slot_off=False,
            source_binder_off=False,
        )
        numeric_off = module.eval_command(
            args,
            checkpoint=Path("out/train/last.pt"),
            out_json=Path("out/numeric_off.json"),
            primitive_off=False,
            numeric_off=True,
            token_numeric_off=False,
            source_slot_off=False,
            source_binder_off=False,
        )
        token_numeric_off = module.eval_command(
            args,
            checkpoint=Path("out/train/last.pt"),
            out_json=Path("out/token_numeric_off.json"),
            primitive_off=False,
            numeric_off=False,
            token_numeric_off=True,
            source_slot_off=False,
            source_binder_off=False,
        )
        source_binder_off = module.eval_command(
            args,
            checkpoint=Path("out/train/last.pt"),
            out_json=Path("out/source_binder_off.json"),
            primitive_off=False,
            numeric_off=False,
            token_numeric_off=False,
            source_slot_off=False,
            source_binder_off=True,
        )
        strict_prompt_binding_off = module.eval_command(
            args,
            checkpoint=Path("out/train/last.pt"),
            out_json=Path("out/strict_prompt_binding_off.json"),
            primitive_off=False,
            numeric_off=False,
            token_numeric_off=False,
            source_slot_off=False,
            source_binder_off=False,
            strict_prompt_binding_off=True,
        )

        self.assertIn("--role-value-list-class-mode", full)
        self.assertIn("source_position", full)
        self.assertIn("--core-source-position-binder-source-slots-only", full)
        self.assertIn("--core-source-position-binder-query-state", full)
        self.assertIn(
            "--core-source-position-binder-query-state-gate-min 0.5",
            " ".join(full),
        )
        self.assertIn("--core-source-value-binder", full)
        self.assertIn("--core-source-value-binder-state-gate-min 0.75", " ".join(full))
        self.assertIn("--core-source-value-binder-state-st", full)
        self.assertIn(
            "--core-primitive-role-value-source-value-conditioning",
            full,
        )
        self.assertIn(
            "--core-primitive-role-value-source-value-gate-min 0.6",
            " ".join(full),
        )
        self.assertNotIn("--disable-core-primitive-role-value-executor", full)
        self.assertIn("--disable-core-primitive-role-value-executor", off)
        self.assertIn("--disable-numeric-source-features", numeric_off)
        self.assertIn("--disable-token-numeric-value-features", token_numeric_off)
        self.assertIn("--disable-core-source-position-binder", source_binder_off)
        self.assertIn(
            "--disable-core-source-position-binder",
            strict_prompt_binding_off,
        )
        self.assertIn(
            "--disable-core-role-value-prompt-extract",
            strict_prompt_binding_off,
        )
        self.assertIn(
            "--disable-core-primitive-prompt-context",
            strict_prompt_binding_off,
        )

    def test_eval_command_can_ablate_token_numeric_source_slots(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--out-dir",
                "out",
                "--token-numeric-source-slots",
                "--token-numeric-source-slot-vocab-size",
                "64",
                "--token-numeric-source-slot-max-slots",
                "7",
                "--token-numeric-source-slot-id-mode",
                "relative_parity",
                "--token-numeric-source-slot-gate-min",
                "1.0",
            ]
        )

        full = module.eval_command(
            args,
            checkpoint=Path("out/train/last.pt"),
            out_json=Path("out/full.json"),
            primitive_off=False,
            numeric_off=False,
            token_numeric_off=False,
            source_slot_off=False,
            source_binder_off=False,
        )
        off = module.eval_command(
            args,
            checkpoint=Path("out/train/last.pt"),
            out_json=Path("out/source_slot_off.json"),
            primitive_off=False,
            numeric_off=False,
            token_numeric_off=False,
            source_slot_off=True,
            source_binder_off=False,
        )
        text = " ".join(full)

        self.assertIn("--token-numeric-source-slots", full)
        self.assertIn("--token-numeric-source-slot-vocab-size 64", text)
        self.assertIn("--token-numeric-source-slot-max-slots 7", text)
        self.assertIn("--token-numeric-source-slot-id-mode relative_parity", text)
        self.assertIn("--token-numeric-source-slot-gate-min 1.0", text)
        self.assertIn("--disable-token-numeric-source-slots", off)

    def test_decision_requires_trace_value_and_drop(self):
        module = load_module()

        accepted = module.summarize_gate(
            full_summary={
                "trace_exact_accuracy": 0.30,
                "value_accuracy": 0.60,
                "step_exact_accuracy": 0.50,
            },
            ablation_summary={
                "trace_exact_accuracy": 0.0,
                "value_accuracy": 0.10,
                "step_exact_accuracy": 0.05,
            },
            min_trace_exact=0.25,
            min_value_accuracy=0.50,
            min_value_drop=0.25,
            numeric_ablation_summary={"value_accuracy": 0.20},
            min_numeric_value_drop=0.25,
            source_slot_ablation_summary={"value_accuracy": 0.20},
            min_source_slot_value_drop=0.25,
            source_binder_ablation_summary={"value_accuracy": 0.20},
            min_source_binder_value_drop=0.25,
            strict_prompt_binding_ablation_summary={"value_accuracy": 0.20},
            min_strict_prompt_binding_value_drop=0.25,
        )
        rejected = module.summarize_gate(
            full_summary={
                "trace_exact_accuracy": 0.30,
                "value_accuracy": 0.60,
                "step_exact_accuracy": 0.50,
            },
            ablation_summary={
                "trace_exact_accuracy": 0.20,
                "value_accuracy": 0.55,
                "step_exact_accuracy": 0.45,
            },
            min_trace_exact=0.25,
            min_value_accuracy=0.50,
            min_value_drop=0.25,
            numeric_ablation_summary={"value_accuracy": 0.20},
            min_numeric_value_drop=0.25,
            source_slot_ablation_summary={"value_accuracy": 0.20},
            min_source_slot_value_drop=0.25,
            source_binder_ablation_summary={"value_accuracy": 0.20},
            min_source_binder_value_drop=0.25,
            strict_prompt_binding_ablation_summary={"value_accuracy": 0.55},
            min_strict_prompt_binding_value_drop=0.25,
        )

        self.assertEqual(accepted["decision"], "accepted_l2")
        self.assertTrue(accepted["accepted"])
        self.assertEqual(rejected["decision"], "rejected")
        self.assertFalse(rejected["accepted"])

    def test_candidate_checkpoints_are_sorted_with_last_at_end(self):
        module = load_module()

        paths = [
            Path("run/train/step_000200.pt"),
            Path("run/train/last.pt"),
            Path("run/train/step_000100.pt"),
        ]

        ordered = module.sort_candidate_checkpoints(paths)

        self.assertEqual(
            [path.name for path in ordered],
            ["step_000100.pt", "step_000200.pt", "last.pt"],
        )

    def test_best_candidate_prefers_trace_then_value_then_step_accuracy(self):
        module = load_module()

        best = module.select_best_full_candidate(
            [
                {
                    "checkpoint": "step_000100.pt",
                    "summary": {
                        "trace_exact_accuracy": 0.0,
                        "value_accuracy": 0.40,
                        "step_exact_accuracy": 0.20,
                    },
                },
                {
                    "checkpoint": "step_000200.pt",
                    "summary": {
                        "trace_exact_accuracy": 0.1,
                        "value_accuracy": 0.10,
                        "step_exact_accuracy": 0.10,
                    },
                },
            ]
        )

        self.assertEqual(best["checkpoint"], "step_000200.pt")


if __name__ == "__main__":
    unittest.main()
