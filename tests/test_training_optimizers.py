import unittest
from unittest import mock

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
            beta1=0.8,
            beta2=0.95,
        )

        self.assertEqual(report["resolved"], "adamw")
        self.assertEqual(report["beta1"], 0.8)
        self.assertEqual(report["beta2"], 0.95)
        self.assertEqual(optimizer.param_groups[0]["betas"], (0.8, 0.95))
        self.assertEqual(
            report["trainable_parameter_count"],
            sum(param.numel() for param in model.parameters()),
        )
        self.assertEqual(len(optimizer.param_groups), 1)

    def test_auto_resolves_to_plain_adamw_on_cpu(self):
        self.assertEqual(resolve_optimizer_name("auto", device=torch.device("cpu")), "adamw")

    def test_galore_splits_large_2d_tensors(self):
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.LayerNorm(8))

        class FakeGaLoreAdamW(torch.optim.AdamW):
            pass

        with mock.patch(
            "qtrm_mm.training_optimizers._import_galore",
            return_value=(FakeGaLoreAdamW, FakeGaLoreAdamW),
        ):
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

    def test_adamw8bit_uses_optional_bitsandbytes_builder(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.LayerNorm(8))

        class FakeAdamW8bit(torch.optim.AdamW):
            pass

        class FakePagedAdamW8bit(torch.optim.AdamW):
            pass

        with mock.patch(
            "qtrm_mm.training_optimizers._import_bitsandbytes_adamw8bit",
            return_value=(FakeAdamW8bit, FakePagedAdamW8bit),
        ):
            optimizer, report = build_memory_efficient_optimizer(
                model,
                optimizer_name="paged_adamw8bit",
                lr=1e-3,
                weight_decay=0.01,
                device=torch.device("cpu"),
            )

        self.assertIsInstance(optimizer, FakePagedAdamW8bit)
        self.assertEqual(report["resolved"], "paged_adamw8bit")

    def test_ademamix8bit_uses_optional_bitsandbytes_builder(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.LayerNorm(8))

        class FakeAdEMAMix8bit(torch.optim.AdamW):
            def __init__(self, params, *, betas=(0.9, 0.999, 0.9999), **kwargs):
                super().__init__(params, betas=betas[:2], **kwargs)

        class FakePagedAdEMAMix8bit(torch.optim.AdamW):
            def __init__(self, params, *, betas=(0.9, 0.999, 0.9999), **kwargs):
                super().__init__(params, betas=betas[:2], **kwargs)

        with mock.patch(
            "qtrm_mm.training_optimizers._import_bitsandbytes_ademamix8bit",
            return_value=(FakeAdEMAMix8bit, FakePagedAdEMAMix8bit),
        ):
            optimizer, report = build_memory_efficient_optimizer(
                model,
                optimizer_name="paged_ademamix8bit",
                lr=1e-3,
                weight_decay=0.01,
                beta1=0.8,
                beta2=0.95,
                device=torch.device("cpu"),
            )

        self.assertIsInstance(optimizer, FakePagedAdEMAMix8bit)
        self.assertEqual(report["resolved"], "paged_ademamix8bit")
        self.assertEqual(report["beta3"], 0.9999)

    def test_extra_named_parameters_are_included(self):
        model = torch.nn.Linear(4, 8)
        extra = torch.nn.Parameter(torch.ones(3))

        _optimizer, report = build_memory_efficient_optimizer(
            model,
            optimizer_name="adamw",
            lr=1e-3,
            weight_decay=0.01,
            device=torch.device("cpu"),
            extra_named_parameters=[("extra", extra)],
        )

        self.assertEqual(
            report["trainable_parameter_count"],
            sum(param.numel() for param in model.parameters()) + extra.numel(),
        )


if __name__ == "__main__":
    unittest.main()
