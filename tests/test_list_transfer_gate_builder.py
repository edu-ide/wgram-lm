import importlib.util
from pathlib import Path
import tempfile
import unittest


def load_script():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "234_build_list_transfer_gate.py"
    )
    spec = importlib.util.spec_from_file_location("list_transfer_gate", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ListTransferGateBuilderTests(unittest.TestCase):
    def test_builds_list_paraphrase_holdout_without_family_holdout(self):
        module = load_script()

        bundle = module.build_list_transfer_gate(
            cases_per_family=2,
            train_start_index=21000,
            eval_start_index=22000,
            train_list_variants=(0, 1, 2, 3, 4, 5),
            eval_list_variants=(6, 7),
            dense=False,
        )

        train_families = {row["task_family"] for row in bundle.train_rows}
        eval_families = {row["task_family"] for row in bundle.eval_rows}
        train_list_variants = {
            row["surface_variant_index"]
            for row in bundle.train_rows
            if row["task_family"] == "list_transform"
        }
        eval_list_variants = {
            row["surface_variant_index"]
            for row in bundle.eval_rows
            if row["task_family"] == "list_transform"
        }

        self.assertEqual(
            train_families,
            {"arithmetic_chain", "boolean_logic", "list_transform", "symbolic_binding"},
        )
        self.assertEqual(eval_families, {"list_transform"})
        self.assertEqual(train_list_variants, {0, 1, 2, 3, 4, 5})
        self.assertEqual(eval_list_variants, {6, 7})
        self.assertTrue(train_list_variants.isdisjoint(eval_list_variants))
        self.assertEqual(bundle.summary["split_type"], "list_paraphrase_cluster_holdout")

    def test_dense_action_terminal_has_no_nonterminal_finality(self):
        module = load_script()

        bundle = module.build_list_transfer_gate(
            cases_per_family=2,
            train_start_index=21000,
            eval_start_index=22000,
            train_list_variants=(0, 1),
            eval_list_variants=(6,),
            dense=True,
        )

        for row in [*bundle.train_rows, *bundle.eval_rows]:
            for depth, code in row["transition_state_codes"].items():
                finality = int(row["transition_finality_targets"][depth])
                if int(code) in {0, 2}:
                    self.assertEqual(finality, 0)
                if int(code) in {1, 3, 4}:
                    self.assertEqual(finality, 1)

    def test_builds_long_list_value_range_holdout(self):
        module = load_script()

        bundle = module.build_list_transfer_gate(
            cases_per_family=2,
            train_start_index=21000,
            eval_start_index=31000,
            train_list_variants=(0, 1, 2, 3, 4, 5),
            eval_list_variants=(6,),
            eval_list_lengths=(7, 9),
            dense=False,
        )

        train_list_lengths = {
            row.get("list_length")
            for row in bundle.train_rows
            if row["task_family"] == "list_transform"
        }
        eval_list_lengths = {
            row.get("list_length")
            for row in bundle.eval_rows
            if row["task_family"] == "list_transform"
        }
        eval_value_starts = {
            row.get("list_value_start")
            for row in bundle.eval_rows
            if row["task_family"] == "list_transform"
        }

        self.assertEqual(train_list_lengths, {5})
        self.assertEqual(eval_list_lengths, {7, 9})
        self.assertEqual(bundle.summary["eval_list_lengths"], [7, 9])
        self.assertTrue(all(int(value) >= 31000 for value in eval_value_starts))
        self.assertEqual(bundle.summary["eval_rows"], 4)

    def test_write_list_transfer_gate_writes_train_eval_and_summary(self):
        module = load_script()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_out = tmp_path / "train.jsonl"
            eval_out = tmp_path / "eval.jsonl"
            summary_out = tmp_path / "summary.json"

            bundle = module.write_list_transfer_gate(
                train_out=train_out,
                eval_out=eval_out,
                summary_out=summary_out,
                cases_per_family=1,
                train_start_index=21000,
                eval_start_index=22000,
                train_list_variants=(0, 1),
                eval_list_variants=(7,),
                dense=True,
            )

            self.assertTrue(train_out.exists())
            self.assertTrue(eval_out.exists())
            self.assertTrue(summary_out.exists())
            self.assertEqual(bundle.summary["train_rows"], 26)
            self.assertEqual(bundle.summary["eval_rows"], 1)


if __name__ == "__main__":
    unittest.main()
