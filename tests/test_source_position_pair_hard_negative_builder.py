import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from wgram_lm.algorithmic_value_state import role_value_targets_from_row


def load_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "323_build_source_position_pair_hard_negatives.py"
    )
    spec = importlib.util.spec_from_file_location(
        "source_position_pair_hard_negatives",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def source_position_signature(row: dict) -> tuple[tuple[int, ...], ...]:
    targets = role_value_targets_from_row(
        row,
        num_steps=4,
        num_roles=12,
        value_vocab_size=128,
        list_class_mode="source_position",
        supervise_nulls=True,
    )
    return tuple(tuple(step) for step in targets)


class SourcePositionPairHardNegativeBuilderTests(unittest.TestCase):
    def test_pair_groups_keep_same_values_but_force_different_source_targets(self):
        module = load_module()

        bundle = module.build_source_position_pair_hard_negative_split(
            train_groups=3,
            eval_groups=2,
            permutations_per_group=4,
            seed=23,
        )

        self.assertEqual(len(bundle.train_rows), 12)
        self.assertEqual(len(bundle.eval_rows), 8)
        self.assertEqual(
            bundle.summary["split_type"],
            "source_position_pair_hard_negative",
        )
        self.assertEqual(
            bundle.summary["major_bottleneck"],
            "source-position anti-shortcut binding",
        )

        for rows in (bundle.train_rows, bundle.eval_rows):
            by_group: dict[str, list[dict]] = {}
            for row in rows:
                by_group.setdefault(row["pair_group_id"], []).append(row)
                self.assertEqual(row["hard_variant"], "paired_permutation_source_positions")
                self.assertEqual(row["role_value_list_class_mode"], "source_position")
                self.assertTrue(row["role_value_supervise_null_slots"])
                self.assertEqual(len(row["input_list"]), 5)

            for group_rows in by_group.values():
                self.assertEqual(len(group_rows), 4)
                self.assertEqual(
                    len({row["value_multiset_signature"] for row in group_rows}),
                    1,
                )
                self.assertGreater(
                    len({tuple(row["input_list"]) for row in group_rows}),
                    1,
                )
                self.assertGreater(
                    len({source_position_signature(row) for row in group_rows}),
                    1,
                )

        self.assertTrue(
            set(bundle.summary["train_pair_group_ids"]).isdisjoint(
                bundle.summary["eval_pair_group_ids"]
            )
        )
        self.assertTrue(
            set(bundle.summary["train_value_multiset_signatures"]).isdisjoint(
                bundle.summary["eval_value_multiset_signatures"]
            )
        )

    def test_write_split_persists_jsonl_and_summary(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            train_out = Path(tmp) / "train.jsonl"
            eval_out = Path(tmp) / "eval.jsonl"
            summary_out = Path(tmp) / "summary.json"

            summary = module.write_source_position_pair_hard_negative_split(
                train_out=train_out,
                eval_out=eval_out,
                summary_out=summary_out,
                train_groups=2,
                eval_groups=1,
                permutations_per_group=3,
                seed=29,
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

            self.assertEqual(len(train_rows), 6)
            self.assertEqual(len(eval_rows), 3)
            self.assertEqual(summary["rows"], 9)
            self.assertEqual(persisted_summary["rows"], 9)


if __name__ == "__main__":
    unittest.main()
