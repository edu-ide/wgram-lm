import importlib.util
from pathlib import Path
import tempfile
import unittest


def load_script():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "235_build_mixed_family_composition_gate.py"
    )
    spec = importlib.util.spec_from_file_location("mixed_family_gate", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MixedFamilyCompositionGateBuilderTests(unittest.TestCase):
    def test_builds_mixed_list_arithmetic_with_dynamic_finality(self):
        module = load_script()

        bundle = module.build_mixed_family_composition_gate(
            cases_per_family=2,
            train_start_index=41000,
            eval_start_index=51000,
            train_list_variants=(0, 1),
            eval_list_variants=(6,),
            eval_list_lengths=(7, 9),
            mixed_repeat=3,
            dense=True,
        )

        train_families = {row["task_family"] for row in bundle.train_rows}
        eval_families = {row["task_family"] for row in bundle.eval_rows}
        eval_lengths = {row["list_length"] for row in bundle.eval_rows}

        self.assertIn("mixed_list_arithmetic", train_families)
        self.assertEqual(eval_families, {"mixed_list_arithmetic"})
        self.assertEqual(eval_lengths, {7, 9})
        self.assertEqual(bundle.summary["finality_mode"], "answer_match")
        self.assertEqual(bundle.summary["codebook_version"], "dynamic_halt_v3")
        self.assertEqual(bundle.summary["mixed_repeat"], 3)
        self.assertEqual(bundle.summary["mixed_train_rows"], 12)
        self.assertEqual(bundle.summary["eval_rows"], 4)

        row = bundle.eval_rows[0]
        self.assertEqual(
            [row["transition_state_codes"][str(depth)] for depth in range(1, 9)],
            [0, 1, 2, 3, 4, 4, 4, 4],
        )
        self.assertEqual(row["transition_finality_targets"]["2"], 0)
        self.assertEqual(row["transition_finality_targets"]["4"], 1)
        self.assertEqual(row["transition_finality_targets"]["8"], 1)
        self.assertNotIn("terminal", row["latent_action_trace"][1]["action_name"])

    def test_write_mixed_family_gate_writes_train_eval_and_summary(self):
        module = load_script()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_out = tmp_path / "train.jsonl"
            eval_out = tmp_path / "eval.jsonl"
            summary_out = tmp_path / "summary.json"

            bundle = module.write_mixed_family_composition_gate(
                train_out=train_out,
                eval_out=eval_out,
                summary_out=summary_out,
                cases_per_family=1,
                train_start_index=41000,
                eval_start_index=51000,
                train_list_variants=(0,),
                eval_list_variants=(7,),
                eval_list_lengths=(7,),
                mixed_repeat=2,
                dense=True,
            )

            self.assertTrue(train_out.exists())
            self.assertTrue(eval_out.exists())
            self.assertTrue(summary_out.exists())
            self.assertEqual(bundle.summary["mixed_train_rows"], 2)
            self.assertEqual(bundle.summary["eval_rows"], 1)

    def test_train_rows_interleave_mixed_cases_for_deterministic_schedule(self):
        module = load_script()

        bundle = module.build_mixed_family_composition_gate(
            cases_per_family=2,
            train_start_index=41000,
            eval_start_index=51000,
            train_list_variants=(0, 1),
            eval_list_variants=(6,),
            eval_list_lengths=(7,),
            mixed_repeat=2,
            dense=False,
        )

        first_eight_families = [row["task_family"] for row in bundle.train_rows[:8]]

        self.assertIn("mixed_list_arithmetic", first_eight_families)
        self.assertNotEqual(
            [row["task_family"] for row in bundle.train_rows[-4:]],
            ["mixed_list_arithmetic"] * 4,
        )

    def test_can_build_train_mixed_cases_with_multiple_list_lengths(self):
        module = load_script()

        bundle = module.build_mixed_family_composition_gate(
            cases_per_family=2,
            train_start_index=41000,
            eval_start_index=51000,
            train_list_variants=(0,),
            eval_list_variants=(6,),
            train_list_lengths=(5, 7, 9),
            eval_list_lengths=(7,),
            mixed_repeat=1,
            dense=False,
        )

        mixed_train_lengths = {
            row["list_length"]
            for row in bundle.train_rows
            if row["task_family"] == "mixed_list_arithmetic"
        }

        self.assertEqual(mixed_train_lengths, {5, 7, 9})
        self.assertEqual(bundle.summary["train_list_lengths"], [5, 7, 9])
        self.assertEqual(bundle.summary["mixed_train_rows"], 6)

    def test_can_build_reverse_arithmetic_to_list_composition(self):
        module = load_script()

        bundle = module.build_mixed_family_composition_gate(
            cases_per_family=2,
            train_start_index=41000,
            eval_start_index=51000,
            train_list_variants=(0,),
            eval_list_variants=(6,),
            train_list_lengths=(5,),
            eval_list_lengths=(7,),
            mixed_repeat=1,
            composition_orders=("arithmetic_to_list",),
            dense=True,
        )

        train_families = {row["task_family"] for row in bundle.train_rows}
        eval_families = {row["task_family"] for row in bundle.eval_rows}

        self.assertIn("mixed_arithmetic_list", train_families)
        self.assertEqual(eval_families, {"mixed_arithmetic_list"})
        self.assertEqual(bundle.summary["composition_orders"], ["arithmetic_to_list"])

        row = bundle.eval_rows[0]
        self.assertEqual(
            [row["transition_state_codes"][str(depth)] for depth in range(1, 9)],
            [0, 2, 3, 1, 1, 4, 4, 4],
        )
        self.assertEqual(row["transition_finality_targets"]["4"], 0)
        self.assertEqual(row["transition_finality_targets"]["5"], 1)
        self.assertEqual(row["transition_finality_targets"]["8"], 1)
        self.assertEqual(row["composition_order"], "arithmetic_to_list")

    def test_reverse_composition_supports_extra_train_paraphrases(self):
        module = load_script()

        bundle = module.build_mixed_family_composition_gate(
            cases_per_family=1,
            train_start_index=41000,
            eval_start_index=51000,
            train_list_variants=(22, 23),
            eval_list_variants=(6,),
            train_list_lengths=(5,),
            eval_list_lengths=(7,),
            mixed_repeat=1,
            composition_orders=("arithmetic_to_list",),
            dense=False,
        )

        train_variants = {
            row["surface_variant_index"]
            for row in bundle.train_rows
            if row["task_family"] == "mixed_arithmetic_list"
        }

        self.assertEqual(train_variants, {22, 23})
        self.assertEqual(bundle.summary["train_list_variants"], [22, 23])

    def test_list_to_arithmetic_extra_variants_do_not_wrap_to_eval_variants(self):
        module = load_script()

        bundle = module.build_mixed_family_composition_gate(
            cases_per_family=1,
            train_start_index=41000,
            eval_start_index=51000,
            train_list_variants=(8, 15, 23),
            eval_list_variants=(6, 7),
            train_list_lengths=(5,),
            eval_list_lengths=(7,),
            mixed_repeat=1,
            composition_orders=("list_to_arithmetic",),
            dense=False,
        )

        train_variants = {
            row["surface_variant_index"]
            for row in bundle.train_rows
            if row["task_family"] == "mixed_list_arithmetic"
        }
        eval_variants = {
            row["surface_variant_index"]
            for row in bundle.eval_rows
            if row["task_family"] == "mixed_list_arithmetic"
        }

        self.assertEqual(train_variants, {8, 15, 23})
        self.assertEqual(eval_variants, {6, 7})
        self.assertFalse(train_variants & eval_variants)

    def test_variant_overlap_requires_explicit_diagnostic_flag(self):
        module = load_script()

        with self.assertRaises(ValueError):
            module.build_mixed_family_composition_gate(
                cases_per_family=1,
                train_start_index=41000,
                eval_start_index=51000,
                train_list_variants=(6,),
                eval_list_variants=(6,),
                train_list_lengths=(5,),
                eval_list_lengths=(7,),
                dense=False,
            )

        bundle = module.build_mixed_family_composition_gate(
            cases_per_family=1,
            train_start_index=41000,
            eval_start_index=51000,
            train_list_variants=(6,),
            eval_list_variants=(6,),
            train_list_lengths=(5,),
            eval_list_lengths=(7,),
            dense=False,
            allow_variant_overlap=True,
        )

        self.assertEqual(bundle.summary["variant_overlap"], [6])
        self.assertTrue(bundle.summary["allow_variant_overlap"])

    def test_composition_orders_are_interleaved_for_short_training_budgets(self):
        module = load_script()

        bundle = module.build_mixed_family_composition_gate(
            cases_per_family=2,
            train_start_index=41000,
            eval_start_index=51000,
            train_list_variants=(0,),
            eval_list_variants=(6,),
            train_list_lengths=(5,),
            eval_list_lengths=(7,),
            mixed_repeat=1,
            composition_orders=("list_to_arithmetic", "arithmetic_to_list"),
            dense=False,
        )

        first_mixed_orders = [
            row.get("composition_order", "list_to_arithmetic")
            for row in bundle.train_rows
            if row["task_family"] in {"mixed_list_arithmetic", "mixed_arithmetic_list"}
        ][:4]

        self.assertEqual(
            first_mixed_orders,
            [
                "list_to_arithmetic",
                "arithmetic_to_list",
                "list_to_arithmetic",
                "arithmetic_to_list",
            ],
        )


if __name__ == "__main__":
    unittest.main()
