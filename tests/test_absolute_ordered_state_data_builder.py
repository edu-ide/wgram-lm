import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def load_builder():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "317_build_absolute_ordered_state_gate_data.py"
    )
    spec = importlib.util.spec_from_file_location("absolute_ordered_state_data", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AbsoluteOrderedStateDataBuilderTests(unittest.TestCase):
    def test_build_rows_have_solver_trace_and_absolute_targets(self):
        module = load_builder()

        rows = module.build_absolute_ordered_state_rows(
            count=8,
            seed=17,
            value_modulus=16,
            list_len=5,
            include_coverage=True,
        )

        self.assertGreaterEqual(len(rows), 8)
        self.assertTrue(all(row["task_family"] == "list_transform" for row in rows))
        self.assertTrue(all(row["role_value_list_class_mode"] == "absolute" for row in rows))
        self.assertTrue(all(row.get("solver_trace") for row in rows))
        self.assertTrue(all("filter_even" in row["solver_trace"][0]["operation"] for row in rows))

    def test_train_coverage_contains_eval_absolute_target_classes(self):
        module = load_builder()
        value_state = module.load_algorithmic_value_state_module()
        train_rows, eval_rows, summary = module.build_absolute_ordered_state_split(
            train_count=64,
            eval_count=16,
            train_seed=123,
            eval_seed=456,
            value_modulus=24,
            list_len=5,
            value_vocab_size=128,
        )

        train_classes = module.target_classes_for_rows(
            train_rows,
            value_state=value_state,
            value_vocab_size=128,
        )
        eval_classes = module.target_classes_for_rows(
            eval_rows,
            value_state=value_state,
            value_vocab_size=128,
        )

        self.assertTrue(eval_classes)
        self.assertTrue(eval_classes.issubset(train_classes))
        self.assertEqual(summary["split_type"], "absolute_value_coverage_combo_holdout")
        self.assertTrue(set(summary["eval_target_classes"]).issubset(summary["train_target_classes"]))

    def test_write_split_writes_jsonl_and_summary(self):
        module = load_builder()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_out = tmp_path / "train.jsonl"
            eval_out = tmp_path / "eval.jsonl"
            summary_out = tmp_path / "summary.json"

            summary = module.write_absolute_ordered_state_split(
                train_out=train_out,
                eval_out=eval_out,
                summary_out=summary_out,
                train_count=12,
                eval_count=4,
                value_modulus=16,
                list_len=5,
                value_vocab_size=128,
            )

            self.assertTrue(train_out.exists())
            self.assertTrue(eval_out.exists())
            self.assertTrue(summary_out.exists())
            train_lines = train_out.read_text(encoding="utf-8").strip().splitlines()
            eval_lines = eval_out.read_text(encoding="utf-8").strip().splitlines()
            loaded_summary = json.loads(summary_out.read_text(encoding="utf-8"))

        self.assertEqual(len(train_lines), summary["train_rows"])
        self.assertEqual(len(eval_lines), summary["eval_rows"])
        self.assertEqual(loaded_summary["eval_rows"], summary["eval_rows"])


if __name__ == "__main__":
    unittest.main()
