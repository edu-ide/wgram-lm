import unittest

import torch

from qtrm_mm.training_optimizers import (
    build_memory_efficient_optimizer,
    resolve_optimizer_name,
)


class TrainingOptimizerTests(unittest.TestCase):
    def test_adamw_report_counts_trainable_parameters(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.LayerNorm(8))

        optimizer, report = build_memory_efficient_optimizer(
            model,
            optimizer_name="adamw",
            lr=1e-3,
            weight_decay=0.01,
            device=torch.device("cpu"),
        )

        self.assertEqual(report["resolved"], "adamw")
        self.assertEqual(
            report["trainable_parameter_count"],
            sum(param.numel() for param in model.parameters()),
        )
        self.assertEqual(len(optimizer.param_groups), 1)

    def test_auto_resolves_to_plain_adamw_on_cpu(self):
        self.assertEqual(resolve_optimizer_name("auto", device=torch.device("cpu")), "adamw")

    def test_galore_splits_large_2d_tensors(self):
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.LayerNorm(8))

        _optimizer, report = build_memory_efficient_optimizer(
            model,
            optimizer_name="galore_adamw",
            lr=1e-3,
            weight_decay=0.01,
            device=torch.device("cpu"),
            galore_rank=4,
            galore_min_dim=8,
        )

        self.assertEqual(report["resolved"], "galore_adamw")
        self.assertGreater(report["galore_tensor_count"], 0)
        self.assertGreater(report["regular_tensor_count"], 0)


if __name__ == "__main__":
    unittest.main()
