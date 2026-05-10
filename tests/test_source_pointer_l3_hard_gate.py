import importlib.util
from pathlib import Path
import unittest


def load_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "321_run_source_pointer_l3_hard_gate.py"
    )
    spec = importlib.util.spec_from_file_location("source_pointer_l3_hard_gate", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SourcePointerL3HardGateTests(unittest.TestCase):
    def test_build_hard_rows_cover_expected_perturbations(self):
        module = load_module()

        rows = module.build_l3_hard_rows(count_per_variant=2, seed=7)

        self.assertEqual(len(rows), 8)
        self.assertEqual(
            {row["hard_variant"] for row in rows},
            {
                "range_shift_v32to63",
                "fifth_position_single_even",
                "duplicate_even_binding",
                "surface_paraphrase",
            },
        )
        self.assertTrue(all(row["task_family"] == "list_transform" for row in rows))
        self.assertTrue(all(row["role_value_list_class_mode"] == "source_position" for row in rows))
        self.assertTrue(all(row["role_value_supervise_null_slots"] for row in rows))
        self.assertTrue(all("[" in row["prompt"] and "]" in row["prompt"] for row in rows))
        self.assertTrue(all(len(row["input_list"]) == 5 for row in rows))

    def test_fifth_position_variant_has_only_last_even(self):
        module = load_module()

        rows = module.build_l3_hard_rows(count_per_variant=4, seed=11)
        fifth_rows = [
            row for row in rows if row["hard_variant"] == "fifth_position_single_even"
        ]

        self.assertTrue(fifth_rows)
        for row in fifth_rows:
            values = row["input_list"]
            self.assertTrue(all(value % 2 == 1 for value in values[:4]))
            self.assertEqual(values[4] % 2, 0)
            self.assertEqual(row["depth_targets"]["1"], str(values[4]))

    def test_eval_command_uses_canonical_source_pointer_path_and_ablations(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--out-dir",
                "out",
                "--checkpoint",
                "ckpt.pt",
                "--data-jsonl",
                "eval.jsonl",
            ]
        )

        full = module.eval_command(
            args,
            checkpoint=Path("ckpt.pt"),
            out_json=Path("full.json"),
        )
        primitive_off = module.eval_command(
            args,
            checkpoint=Path("ckpt.pt"),
            out_json=Path("primitive_off.json"),
            primitive_off=True,
        )
        source_off = module.eval_command(
            args,
            checkpoint=Path("ckpt.pt"),
            out_json=Path("source_off.json"),
            source_binder_off=True,
        )

        self.assertIn("--use-role-value-state", full)
        self.assertIn("--use-core-primitive-role-value-state", full)
        self.assertIn("--role-value-list-class-mode", full)
        self.assertIn("source_position", full)
        self.assertIn("--token-numeric-value-features", full)
        self.assertIn("--core-source-position-binder", full)
        self.assertIn("--disable-core-primitive-role-value-executor", primitive_off)
        self.assertIn("--disable-core-source-position-binder", source_off)

    def test_eval_command_can_use_token_numeric_source_slots(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--out-dir",
                "out",
                "--checkpoint",
                "ckpt.pt",
                "--data-jsonl",
                "eval.jsonl",
                "--token-numeric-source-slots",
                "--token-numeric-source-slot-id-mode",
                "relative_parity",
                "--token-numeric-source-slot-vocab-size",
                "3",
                "--token-numeric-source-slot-gate-min",
                "1.0",
                "--token-numeric-source-slot-predicate-feedback",
                "--token-numeric-source-slot-predicate-gate-min",
                "1.0",
                "--core-source-position-binder-gate-min",
                "1.0",
                "--core-source-position-binder-state-gate-min",
                "0.25",
                "--core-source-position-binder-state-st",
                "--core-source-position-binder-source-slots-only",
                "--core-source-position-binder-raw-source-slots",
            ]
        )

        full = module.eval_command(
            args,
            checkpoint=Path("ckpt.pt"),
            out_json=Path("full.json"),
        )
        token_off = module.eval_command(
            args,
            checkpoint=Path("ckpt.pt"),
            out_json=Path("token_off.json"),
            token_numeric_off=True,
        )
        text = " ".join(full)

        self.assertIn("--token-numeric-source-slots", full)
        self.assertIn("--token-numeric-source-slot-id-mode relative_parity", text)
        self.assertIn("--token-numeric-source-slot-vocab-size 3", text)
        self.assertIn("--token-numeric-source-slot-gate-min 1.0", text)
        self.assertIn("--token-numeric-source-slot-predicate-feedback", full)
        self.assertIn("--token-numeric-source-slot-predicate-gate-min 1.0", text)
        self.assertIn("--core-source-position-binder-gate-min 1.0", text)
        self.assertIn("--core-source-position-binder-state-gate-min 0.25", text)
        self.assertIn("--core-source-position-binder-state-st", full)
        self.assertIn("--core-source-position-binder-source-slots-only", full)
        self.assertIn("--core-source-position-binder-raw-source-slots", full)
        self.assertIn("--disable-token-numeric-source-slots", token_off)

    def test_l3_decision_requires_value_trace_variant_and_ablation_drop(self):
        module = load_module()
        full = {"trace_exact_accuracy": 0.12, "value_accuracy": 0.45}
        primitive_off = {"value_accuracy": 0.10}
        token_off = {"value_accuracy": 0.45}
        source_off = {"value_accuracy": 0.11}
        by_variant = {
            "a": {"value_accuracy": 0.31},
            "b": {"value_accuracy": 0.41},
        }

        accepted = module.summarize_l3_decision(
            full_summary=full,
            primitive_off_summary=primitive_off,
            token_numeric_off_summary=token_off,
            source_binder_off_summary=source_off,
            full_by_variant=by_variant,
            min_trace_exact=0.10,
            min_value_accuracy=0.40,
            min_primitive_value_drop=0.20,
            min_token_numeric_value_drop=0.0,
            min_source_binder_value_drop=0.20,
            min_variant_value_accuracy=0.30,
        )
        rejected = module.summarize_l3_decision(
            full_summary=full,
            primitive_off_summary={"value_accuracy": 0.35},
            token_numeric_off_summary=token_off,
            source_binder_off_summary=source_off,
            full_by_variant=by_variant,
            min_trace_exact=0.10,
            min_value_accuracy=0.40,
            min_primitive_value_drop=0.20,
            min_token_numeric_value_drop=0.0,
            min_source_binder_value_drop=0.20,
            min_variant_value_accuracy=0.30,
        )

        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["decision"], "accepted_l3")
        self.assertFalse(rejected["accepted"])
        self.assertIn("primitive-off", " ".join(rejected["reject_reasons"]))

    def test_l3_decision_can_require_token_numeric_drop(self):
        module = load_module()

        rejected = module.summarize_l3_decision(
            full_summary={"trace_exact_accuracy": 0.50, "value_accuracy": 0.80},
            primitive_off_summary={"value_accuracy": 0.10},
            token_numeric_off_summary={"value_accuracy": 0.75},
            source_binder_off_summary={"value_accuracy": 0.10},
            full_by_variant={"a": {"value_accuracy": 0.80}},
            min_trace_exact=0.10,
            min_value_accuracy=0.40,
            min_primitive_value_drop=0.20,
            min_token_numeric_value_drop=0.20,
            min_source_binder_value_drop=0.20,
            min_variant_value_accuracy=0.30,
        )

        self.assertFalse(rejected["accepted"])
        self.assertIn("token-numeric-off", " ".join(rejected["reject_reasons"]))


if __name__ == "__main__":
    unittest.main()
