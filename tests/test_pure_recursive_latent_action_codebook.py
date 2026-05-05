import importlib.util
from pathlib import Path
import tempfile
import unittest


def load_builder_script():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "229_build_pure_recursive_latent_action_codebook_cases.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pure_recursive_latent_action_codebook", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PureRecursiveLatentActionCodebookTests(unittest.TestCase):
    def test_latent_action_codes_remove_unseen_family_operation_labels(self):
        module = load_builder_script()

        bundle = module.build_latent_action_codebook_holdout(
            holdout_family="list_transform",
            cases_per_family=2,
            start_index=0,
            stress_variants_per_case=2,
            drop_solver_operation_names=True,
        )

        self.assertEqual(
            set(bundle.summary["unseen_eval_operations"]),
            {"filter_even", "double_filtered"},
        )
        self.assertEqual(bundle.summary["unseen_eval_latent_action_codes"], [])
        self.assertEqual(
            set(bundle.summary["eval_latent_action_names"]),
            {"extract_or_unary_transform", "compose_from_previous", "hold_final"},
        )
        self.assertEqual(bundle.summary["interpretation"], "latent_action_transfer_feasible")

    def test_remapped_rows_use_transition_state_codes_without_operation_names(self):
        module = load_builder_script()

        bundle = module.build_latent_action_codebook_holdout(
            holdout_family="list_transform",
            cases_per_family=1,
            start_index=0,
            stress_variants_per_case=1,
            drop_solver_operation_names=True,
        )
        row = bundle.eval_rows[0]

        self.assertEqual(row["transition_state_codes"], {"1": 0, "2": 1, "4": 3, "8": 3})
        self.assertEqual(row["transition_finality_targets"], {"1": 0, "2": 1, "4": 1, "8": 1})
        self.assertEqual(
            [step["action_name"] for step in row["latent_action_trace"]],
            [
                "extract_or_unary_transform",
                "compose_from_previous",
                "hold_final",
                "hold_final",
            ],
        )
        self.assertNotIn("operation", row["solver_trace"][0])
        self.assertTrue(row["latent_action_codebook_applied"])

    def test_writer_emits_codebook_train_eval_summary(self):
        module = load_builder_script()

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            bundle = module.write_latent_action_codebook_holdout(
                train_out=out_dir / "train.jsonl",
                eval_out=out_dir / "eval.jsonl",
                summary_out=out_dir / "summary.json",
                holdout_family="list_transform",
                cases_per_family=1,
                start_index=20,
                stress_variants_per_case=1,
            )

            self.assertTrue((out_dir / "train.jsonl").exists())
            self.assertTrue((out_dir / "eval.jsonl").exists())
            self.assertTrue((out_dir / "summary.json").exists())

        self.assertEqual(bundle.summary["train_count"], 3)
        self.assertEqual(bundle.summary["eval_count"], 1)

    def test_terminal_v2_splits_terminal_and_nonterminal_compose(self):
        module = load_builder_script()

        bundle = module.build_latent_action_codebook_holdout(
            holdout_family="list_transform",
            cases_per_family=1,
            start_index=0,
            stress_variants_per_case=1,
            codebook_version="terminal_v2",
        )
        row = bundle.eval_rows[0]

        self.assertEqual(row["transition_state_codes"], {"1": 0, "2": 1, "4": 4, "8": 4})
        self.assertEqual(bundle.summary["latent_action_codebook_size"], 5)
        self.assertEqual(bundle.summary["unseen_eval_latent_action_codes"], [])
        self.assertIn(
            "compose_from_previous_terminal",
            bundle.summary["train_latent_action_names"],
        )
        self.assertIn(
            "compose_from_previous_terminal",
            bundle.summary["eval_latent_action_names"],
        )


if __name__ == "__main__":
    unittest.main()
