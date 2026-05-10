import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from qtrm_mm.algorithmic_value_state import role_value_targets_from_row


def load_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "326_build_source_copy_lexicalization_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "source_copy_lexicalization_builder",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _csv(values: list[int]) -> str:
    return ",".join(str(value) for value in values) if values else "EMPTY"


class SourceCopyLexicalizationBuilderTests(unittest.TestCase):
    def test_rows_copy_even_values_without_doubling_but_keep_source_targets(self):
        module = load_module()

        bundle = module.build_source_copy_lexicalization_split(
            train_groups=2,
            eval_groups=1,
            permutations_per_group=3,
            seed=41,
        )

        self.assertEqual(bundle.summary["split_type"], "source_copy_lexicalization")
        self.assertEqual(len(bundle.train_rows), 6)
        self.assertEqual(len(bundle.eval_rows), 3)

        for row in bundle.train_rows + bundle.eval_rows:
            values = [int(value) for value in row["input_list"]]
            evens = [value for value in values if value % 2 == 0]
            self.assertEqual(row["answer_aliases"], [_csv(evens)])
            self.assertNotIn("double", row["question"].lower())
            self.assertEqual(row["role_value_list_class_mode"], "source_position")
            self.assertTrue(row["role_value_source_copy_no_doubled"])
            targets = role_value_targets_from_row(
                row,
                num_steps=4,
                num_roles=12,
                value_vocab_size=128,
                list_class_mode="source_position",
                supervise_nulls=True,
            )
            flat_targets = [int(value) for step in targets for value in step]
            self.assertGreaterEqual(max(flat_targets), 0)
            source_positions = [
                index + 1 for index, value in enumerate(values) if value % 2 == 0
            ]
            self.assertEqual(targets[0][: len(source_positions)], source_positions)
            self.assertEqual(targets[1][: len(source_positions)], source_positions)

    def test_write_split_persists_outputs(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            train_out = Path(tmp) / "train.jsonl"
            eval_out = Path(tmp) / "eval.jsonl"
            summary_out = Path(tmp) / "summary.json"

            summary = module.write_source_copy_lexicalization_split(
                train_out=train_out,
                eval_out=eval_out,
                summary_out=summary_out,
                train_groups=1,
                eval_groups=1,
                permutations_per_group=2,
                seed=43,
            )

            train_rows = [
                json.loads(line)
                for line in train_out.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            eval_rows = [
                json.loads(line)
                for line in eval_out.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            persisted_summary = json.loads(summary_out.read_text(encoding="utf-8"))

            self.assertEqual(len(train_rows), 2)
            self.assertEqual(len(eval_rows), 2)
            self.assertEqual(summary["rows"], 4)
            self.assertEqual(persisted_summary["rows"], 4)


if __name__ == "__main__":
    unittest.main()
