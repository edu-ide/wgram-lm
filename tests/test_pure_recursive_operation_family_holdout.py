import importlib.util
from pathlib import Path
import tempfile
import unittest


def load_builder_script():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "228_build_pure_recursive_operation_family_holdout.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pure_recursive_operation_family_holdout", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PureRecursiveOperationFamilyHoldoutTests(unittest.TestCase):
    def test_list_family_holdout_reports_unseen_operations(self):
        module = load_builder_script()

        bundle = module.build_operation_family_holdout(
            holdout_family="list_transform",
            cases_per_family=2,
            start_index=0,
            stress_variants_per_case=2,
        )

        self.assertEqual(
            {row["task_family"] for row in bundle.train_rows},
            {"arithmetic_chain", "symbolic_binding", "boolean_logic"},
        )
        self.assertEqual(
            {row["task_family"] for row in bundle.eval_rows},
            {"list_transform"},
        )
        self.assertEqual(
            set(bundle.summary["unseen_eval_operations"]),
            {"filter_even", "double_filtered"},
        )
        self.assertIn("hold_final", bundle.summary["shared_operations"])
        self.assertEqual(bundle.summary["interpretation"], "reject_fixed_label_transfer")

    def test_writer_emits_train_eval_and_summary_files(self):
        module = load_builder_script()

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            bundle = module.write_operation_family_holdout(
                train_out=out_dir / "train.jsonl",
                eval_out=out_dir / "eval.jsonl",
                summary_out=out_dir / "summary.json",
                holdout_family="list_transform",
                cases_per_family=1,
                start_index=10,
                stress_variants_per_case=1,
            )

            self.assertTrue((out_dir / "train.jsonl").exists())
            self.assertTrue((out_dir / "eval.jsonl").exists())
            self.assertTrue((out_dir / "summary.json").exists())

        self.assertEqual(bundle.summary["train_count"], 3)
        self.assertEqual(bundle.summary["eval_count"], 1)

    def test_cli_accepts_holdout_family_paths(self):
        module = load_builder_script()

        args = module.build_arg_parser().parse_args(
            [
                "--holdout-family",
                "list_transform",
                "--train-out",
                "/tmp/train.jsonl",
                "--eval-out",
                "/tmp/eval.jsonl",
                "--summary-out",
                "/tmp/summary.json",
            ]
        )

        self.assertEqual(args.holdout_family, "list_transform")
        self.assertEqual(args.train_out, "/tmp/train.jsonl")
        self.assertEqual(args.eval_out, "/tmp/eval.jsonl")
        self.assertEqual(args.summary_out, "/tmp/summary.json")


if __name__ == "__main__":
    unittest.main()
