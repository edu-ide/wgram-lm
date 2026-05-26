import importlib.util
from pathlib import Path
import sys
import unittest

import torch


def load_module():
    path = Path("scripts/625_train_bpe_gd_preference.py")
    spec = importlib.util.spec_from_file_location("bpe_gd_preference_train", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BPEGDPreferenceTrainTests(unittest.TestCase):
    def test_pairwise_preference_loss_rewards_larger_positive_margin(self):
        module = load_module()
        bad = module.pairwise_preference_loss(
            torch.tensor([-0.5, -0.1]),
            beta=4.0,
            target_margin=0.05,
        )
        good = module.pairwise_preference_loss(
            torch.tensor([0.5, 0.8]),
            beta=4.0,
            target_margin=0.05,
        )
        self.assertLess(float(good), float(bad))

    def test_select_train_rows_excludes_heldout_and_balances_tasks(self):
        module = load_module()
        rows = []
        for task in ("a", "b"):
            for index in range(5):
                rows.append(
                    {
                        "id": f"{task}{index}",
                        "task": task,
                        "prompt": f"p {task} {index}",
                        "intelligence_answer": " yes",
                        "parrot_answer": " no",
                    }
                )
        selected = module.select_train_rows(
            rows,
            exclude_ids={"a0", "b0"},
            max_rows=4,
            seed=7,
            balance_by_task=True,
        )
        self.assertEqual(4, len(selected))
        self.assertNotIn("a0", {row["id"] for row in selected})
        self.assertNotIn("b0", {row["id"] for row in selected})
        self.assertEqual({"a": 2, "b": 2}, {
            task: sum(1 for row in selected if row["task"] == task)
            for task in ("a", "b")
        })

    def test_apply_focus_replay_duplicates_matching_tasks_only(self):
        module = load_module()
        rows = [
            {"id": "a", "task": "repetitive_answer/algebra/original"},
            {"id": "b", "task": "truthy_answer/surprising_truth"},
        ]
        replayed = module.apply_focus_replay(
            rows,
            focus_tasks=["algebra"],
            replay_factor=4,
        )
        ids = [row["id"] for row in replayed]
        self.assertEqual(4, ids.count("a"))
        self.assertEqual(1, ids.count("b"))


if __name__ == "__main__":
    unittest.main()
