import importlib.util
from pathlib import Path
import sys
import unittest


def load_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "316_run_qtrm_absolute_ordered_state_gate.py"
    )
    spec = importlib.util.spec_from_file_location("qtrm_absolute_ordered_state_gate", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QTRMAbsoluteOrderedStateGateTests(unittest.TestCase):
    def test_default_data_paths_use_absolute_coverage_split(self):
        module = load_module()

        self.assertIn("qtrm_absolute_ordered_state_train", module.DEFAULT_TRAIN_DATA)
        self.assertIn("qtrm_absolute_ordered_state_eval", module.DEFAULT_EVAL_DATA)

    def test_training_command_is_state_only_and_uses_absolute_list_classes(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--out-dir",
                "out",
                "--steps",
                "7",
                "--save-every",
                "7",
            ]
        )

        command = module.training_command(args, train_out_dir=Path("out/train"))
        text = " ".join(command)

        self.assertIn("--role-value-list-class-mode absolute", text)
        self.assertIn("--final-logit-ce-weight 0.0", text)
        self.assertIn("--depth-final-ce-weight 0.0", text)
        self.assertIn("--all-depth-ce-weight 0.0", text)
        self.assertIn("--progress-margin-weight 0.0", text)
        self.assertIn("--core-role-value-prompt-ce-weight 1.0", text)
        self.assertIn("--core-role-value-prompt-target-mode initial", text)
        self.assertIn("--core-primitive-role-value-state-ce-weight 1.0", text)
        self.assertIn("--save-trainable-only", command)

    def test_eval_command_uses_role_value_mode_and_ablation_flag(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(["--out-dir", "out"])

        full = module.eval_command(
            args,
            checkpoint=Path("out/train/last.pt"),
            out_json=Path("out/full.json"),
            primitive_off=False,
        )
        ablated = module.eval_command(
            args,
            checkpoint=Path("out/train/last.pt"),
            out_json=Path("out/off.json"),
            primitive_off=True,
        )

        self.assertIn("--use-role-value-state", full)
        self.assertIn("--use-core-primitive-role-value-state", full)
        self.assertIn("--role-value-list-class-mode", full)
        self.assertIn("absolute", full)
        self.assertNotIn("--disable-core-primitive-role-value-executor", full)
        self.assertIn("--disable-core-primitive-role-value-executor", ablated)

    def test_decision_requires_full_score_and_ablation_drop(self):
        module = load_module()
        accepted = module.summarize_gate(
            full_summary={
                "trace_exact_accuracy": 0.25,
                "value_accuracy": 0.60,
                "step_exact_accuracy": 0.50,
            },
            ablation_summary={
                "trace_exact_accuracy": 0.0,
                "value_accuracy": 0.10,
                "step_exact_accuracy": 0.05,
            },
            min_trace_exact=0.10,
            min_value_accuracy=0.40,
            min_value_drop=0.20,
        )
        rejected = module.summarize_gate(
            full_summary={
                "trace_exact_accuracy": 0.25,
                "value_accuracy": 0.60,
                "step_exact_accuracy": 0.50,
            },
            ablation_summary={
                "trace_exact_accuracy": 0.20,
                "value_accuracy": 0.55,
                "step_exact_accuracy": 0.45,
            },
            min_trace_exact=0.10,
            min_value_accuracy=0.40,
            min_value_drop=0.20,
        )

        self.assertEqual(accepted["decision"], "accepted_l2")
        self.assertTrue(accepted["accepted"])
        self.assertEqual(rejected["decision"], "rejected")
        self.assertFalse(rejected["accepted"])


if __name__ == "__main__":
    unittest.main()
