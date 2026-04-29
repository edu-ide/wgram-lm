import unittest

import torch
from torch import nn


class LossTests(unittest.TestCase):
    def test_next_token_loss_ignores_padding_targets(self):
        from qtrm_mm.losses import next_token_lm_loss

        logits = torch.zeros(1, 4, 5)
        logits[0, 0, 2] = 10.0
        logits[0, 1, 3] = 10.0
        logits[0, 2, 4] = -10.0
        input_ids = torch.tensor([[1, 2, 3, 4]])
        attention_mask = torch.tensor([[1, 1, 1, 0]])

        masked = next_token_lm_loss(logits, input_ids, attention_mask=attention_mask)
        unmasked = next_token_lm_loss(logits, input_ids)

        self.assertLess(float(masked), 0.01)
        self.assertGreater(float(unmasked), 3.0)

    def test_next_token_loss_can_train_only_unmasked_label_positions(self):
        from qtrm_mm.losses import next_token_lm_loss

        logits = torch.zeros(1, 4, 5)
        logits[0, 0, 2] = -10.0
        logits[0, 1, 3] = 10.0
        logits[0, 2, 4] = 10.0
        input_ids = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[-100, -100, 3, 4]])

        masked = next_token_lm_loss(logits, input_ids, labels=labels)
        unmasked = next_token_lm_loss(logits, input_ids)

        self.assertLess(float(masked), 0.01)
        self.assertGreater(float(unmasked), 3.0)

    def test_qtrm_smoke_loss_respects_component_weights(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                logits = torch.zeros(1, 3, 5)
                logits[0, 0, 2] = 8.0
                logits[0, 1, 3] = 8.0
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(1, 2, 4),
                    "jepa_target": torch.zeros(1, 2, 4),
                    "jepa_mask": torch.ones(1, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(1, 3, 4),
                    "jepa_latent_mask": torch.ones(1, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(1, 1),
                    "action_logits": torch.zeros(1, 3),
                }

        input_ids = torch.tensor([[1, 2, 3]])
        lm_only, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            jepa_weight=0.0,
            aux_weight=0.0,
        )
        weighted, _, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            jepa_weight=0.5,
            aux_weight=1.0,
        )

        self.assertAlmostEqual(float(lm_only), float(metrics["lm"]), places=5)
        self.assertGreater(float(weighted), float(lm_only))

    def test_core_halt_loss_trains_q_halt_logits_against_targets(self):
        from qtrm_mm.losses import core_halt_loss

        good = {
            "core_q_halt_logits": torch.tensor([[3.0, 3.0], [-3.0, -3.0]]),
            "core_q_continue_logits": torch.empty(2, 0),
        }
        bad = {
            "core_q_halt_logits": torch.tensor([[-3.0, -3.0], [3.0, 3.0]]),
            "core_q_continue_logits": torch.empty(2, 0),
        }
        targets = torch.tensor([1.0, 0.0])

        self.assertLess(float(core_halt_loss(good, targets)), 0.1)
        self.assertGreater(float(core_halt_loss(bad, targets)), 2.0)

    def test_qtrm_smoke_loss_adds_core_halt_loss_without_forwarding_targets(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                if "core_halt_targets" in kwargs:
                    raise AssertionError("core_halt_targets must not be forwarded to the model")
                logits = torch.zeros(2, 3, 5)
                logits[:, 0, 2] = 8.0
                logits[:, 1, 3] = 8.0
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(2, 2, 4),
                    "jepa_target": torch.zeros(2, 2, 4),
                    "jepa_mask": torch.ones(2, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(2, 3, 4),
                    "jepa_latent_mask": torch.ones(2, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(2, 1),
                    "action_logits": torch.zeros(2, 3),
                    "core_q_halt_logits": torch.tensor([[3.0, 3.0], [-3.0, -3.0]]),
                    "core_q_continue_logits": torch.empty(2, 0),
                }

        input_ids = torch.tensor([[1, 2, 3], [1, 2, 3]])
        targets = torch.tensor([1.0, 0.0])
        lm_only, _, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            jepa_weight=0.0,
            aux_weight=0.0,
            core_halt_weight=0.0,
            core_halt_targets=targets,
        )
        weighted, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            jepa_weight=0.0,
            aux_weight=0.0,
            core_halt_weight=1.0,
            core_halt_targets=targets,
        )

        self.assertIn("core_halt", metrics)
        self.assertGreater(float(weighted), float(lm_only))

    def test_infer_core_halt_targets_requires_exact_correctness_and_verifier_pass(self):
        from qtrm_mm.losses import infer_core_halt_targets

        logits = torch.zeros(2, 3, 5)
        logits[0, 0, 2] = 8.0
        logits[0, 1, 3] = 8.0
        logits[1, 0, 2] = 8.0
        logits[1, 1, 4] = 8.0
        labels = torch.tensor([[-100, 2, 3], [-100, 2, 3]])
        outputs = {"logits": logits}

        targets = infer_core_halt_targets(
            outputs,
            input_ids=torch.tensor([[0, 2, 3], [0, 2, 3]]),
            labels=labels,
            verifier_passed=torch.tensor([True, True]),
        )

        self.assertTrue(torch.equal(targets, torch.tensor([1.0, 0.0])))

        targets = infer_core_halt_targets(
            outputs,
            input_ids=torch.tensor([[0, 2, 3], [0, 2, 3]]),
            labels=labels,
            verifier_passed=torch.tensor([False, True]),
        )

        self.assertTrue(torch.equal(targets, torch.tensor([0.0, 0.0])))

    def test_infer_core_halt_targets_applies_donor_kl_stability_gate(self):
        from qtrm_mm.losses import infer_core_halt_targets

        logits = torch.zeros(2, 3, 5)
        logits[:, 0, 2] = 8.0
        logits[:, 1, 3] = 8.0
        labels = torch.tensor([[-100, 2, 3], [-100, 2, 3]])
        donor_logits = logits.clone()
        donor_logits[1, 0] = 0.0
        donor_logits[1, 0, 4] = 8.0
        donor_logits[1, 1] = 0.0
        donor_logits[1, 1, 4] = 8.0
        outputs = {"logits": logits}

        targets = infer_core_halt_targets(
            outputs,
            input_ids=torch.tensor([[0, 2, 3], [0, 2, 3]]),
            labels=labels,
            donor_logits=donor_logits,
            donor_kl_threshold=0.1,
        )

        self.assertTrue(torch.equal(targets, torch.tensor([1.0, 0.0])))

    def test_infer_core_halt_targets_can_report_availability_diagnostics(self):
        from qtrm_mm.losses import infer_core_halt_targets

        logits = torch.zeros(3, 3, 5)
        logits[:, 0, 2] = 8.0
        logits[:, 1, 3] = 8.0
        logits[1, 1] = 0.0
        logits[1, 1, 4] = 8.0
        labels = torch.tensor([[-100, 2, 3], [-100, 2, 3], [-100, 2, 3]])
        donor_logits = logits.clone()
        donor_logits[2, 0] = 0.0
        donor_logits[2, 0, 4] = 8.0
        donor_logits[2, 1] = 0.0
        donor_logits[2, 1, 4] = 8.0
        outputs = {"logits": logits}

        targets, diag = infer_core_halt_targets(
            outputs,
            input_ids=torch.tensor([[0, 2, 3], [0, 2, 3], [0, 2, 3]]),
            labels=labels,
            donor_logits=donor_logits,
            donor_kl_threshold=0.1,
            return_diagnostics=True,
        )

        self.assertTrue(torch.equal(targets, torch.tensor([1.0, 0.0, 0.0])))
        self.assertAlmostEqual(float(diag["exact_next_token_pass_rate"]), 2.0 / 3.0, places=5)
        self.assertAlmostEqual(float(diag["donor_kl_pass_rate"]), 2.0 / 3.0, places=5)
        self.assertAlmostEqual(float(diag["halt_target_pos_rate"]), 1.0 / 3.0, places=5)
        self.assertAlmostEqual(float(diag["halt_target_neg_rate"]), 2.0 / 3.0, places=5)

    def test_qtrm_smoke_loss_can_infer_core_halt_targets_from_batch(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                if "verifier_passed" in kwargs:
                    raise AssertionError("verifier_passed must not be forwarded to the model")
                logits = torch.zeros(2, 3, 5)
                logits[:, 0, 2] = 8.0
                logits[:, 1, 3] = 8.0
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(2, 2, 4),
                    "jepa_target": torch.zeros(2, 2, 4),
                    "jepa_mask": torch.ones(2, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(2, 3, 4),
                    "jepa_latent_mask": torch.ones(2, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(2, 1),
                    "action_logits": torch.zeros(2, 3),
                    "core_q_halt_logits": torch.tensor([[3.0, 3.0], [-3.0, -3.0]]),
                    "core_q_continue_logits": torch.empty(2, 0),
                }

        input_ids = torch.tensor([[0, 2, 3], [0, 2, 3]])
        labels = torch.tensor([[-100, 2, 3], [-100, 2, 3]])
        loss, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            labels=labels,
            verifier_passed=torch.tensor([True, False]),
            jepa_weight=0.0,
            aux_weight=0.0,
            core_halt_weight=1.0,
            core_halt_auto_targets=True,
        )

        self.assertIn("core_halt", metrics)
        self.assertLess(float(metrics["core_halt"]), 0.1)
        self.assertGreater(float(loss), 0.0)

    def test_qtrm_smoke_loss_reports_auto_core_halt_target_rates(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                logits = torch.zeros(2, 3, 5)
                logits[:, 0, 2] = 8.0
                logits[:, 1, 3] = 8.0
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(2, 2, 4),
                    "jepa_target": torch.zeros(2, 2, 4),
                    "jepa_mask": torch.ones(2, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(2, 3, 4),
                    "jepa_latent_mask": torch.ones(2, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(2, 1),
                    "action_logits": torch.zeros(2, 3),
                    "core_q_halt_logits": torch.tensor([[3.0, 3.0], [-3.0, -3.0]]),
                    "core_q_continue_logits": torch.empty(2, 0),
                }

        _, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            torch.tensor([[0, 2, 3], [0, 2, 3]]),
            labels=torch.tensor([[-100, 2, 3], [-100, 2, 3]]),
            verifier_passed=torch.tensor([True, False]),
            jepa_weight=0.0,
            aux_weight=0.0,
            core_halt_weight=1.0,
            core_halt_auto_targets=True,
        )

        self.assertIn("halt_target_pos_rate", metrics)
        self.assertIn("halt_target_neg_rate", metrics)
        self.assertIn("exact_next_token_pass_rate", metrics)
        self.assertIn("verifier_pass_rate", metrics)
        self.assertAlmostEqual(float(metrics["halt_target_pos_rate"]), 0.5, places=5)
        self.assertAlmostEqual(float(metrics["halt_target_neg_rate"]), 0.5, places=5)
        self.assertAlmostEqual(float(metrics["exact_next_token_pass_rate"]), 1.0, places=5)
        self.assertAlmostEqual(float(metrics["verifier_pass_rate"]), 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
