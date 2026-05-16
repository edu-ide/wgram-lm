from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


def _load_script():
    path = Path("scripts/400_train_qtrm_native_public_mcq_final_token.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_public_mcq_final_token", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QTRMNativePublicMCQFinalTokenTrainerTests(unittest.TestCase):
    def test_single_token_option_ids_accepts_letter_variants(self):
        module = _load_script()

        class FakeTokenizer:
            def encode(self, text):
                return {"A": [1], " A": [2], "\nA": [3], "AA": [1, 1]}[text]

        self.assertEqual(module.single_token_option_ids(FakeTokenizer(), "A"), (1, 2, 3))

    def test_option_score_uses_probability_mass_over_renderings(self):
        module = _load_script()
        log_probs = torch.log_softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0)

        score = module.option_score(log_probs, (1, 2))

        expected = torch.logsumexp(log_probs[torch.tensor([1, 2])], dim=0)
        self.assertTrue(torch.allclose(score, expected))

    def test_rendered_option_ids_selects_canonical_space_token(self):
        module = _load_script()

        class FakeTokenizer:
            def encode(self, text):
                return {"A": [1], " A": [2], "\nA": [3]}[text]

        self.assertEqual(module.rendered_option_ids(FakeTokenizer(), "A", "space"), (2,))
        self.assertEqual(module.rendered_option_ids(FakeTokenizer(), "A", "plain"), (1,))

    def test_score_rows_reports_pred_histogram(self):
        module = _load_script()

        class FakeTokenizer:
            def encode(self, text):
                return {"prompt": [0], "A": [1], " A": [1], "\nA": [1], "B": [2], " B": [2], "\nB": [2]}[text]

        class FakeModel(torch.nn.Module):
            def forward(self, x, *, think_steps=0, **kwargs):
                logits = torch.zeros((1, x.shape[1], 3), device=x.device)
                logits[:, -1, 2] = 3.0
                return logits

        rows = [{"qtrm_prompt": "prompt", "answer": "B", "options": ["a", "b"]}]

        metrics = module.score_rows(
            FakeModel(),
            FakeTokenizer(),
            rows,
            seq_len=8,
            think_steps=4,
            device=torch.device("cpu"),
        )

        self.assertEqual(metrics["hits"], 1)
        self.assertEqual(metrics["pred_answer_histogram"], {"B": 1})

    def test_option_distribution_log_probs_normalizes_over_options(self):
        module = _load_script()

        class FakeTokenizer:
            def encode(self, text):
                return {"prompt": [0], "A": [1], " A": [1], "\nA": [1], "B": [2], " B": [2], "\nB": [2]}[text]

        class FakeModel(torch.nn.Module):
            def forward(self, x, *, think_steps=0, **kwargs):
                logits = torch.zeros((1, x.shape[1], 3), device=x.device)
                logits[:, -1, 1] = 1.0
                logits[:, -1, 2] = 3.0
                return logits

        row = {"qtrm_prompt": "prompt", "answer": "B", "options": ["a", "b"]}
        log_probs = module.option_distribution_log_probs(
            FakeModel(),
            FakeTokenizer(),
            row,
            seq_len=8,
            think_steps=4,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.allclose(log_probs.exp().sum(), torch.tensor(1.0)))

    def test_preserve_option_kl_loss_is_zero_for_identical_models(self):
        module = _load_script()

        class FakeTokenizer:
            def encode(self, text):
                return {"prompt": [0], "A": [1], " A": [1], "\nA": [1], "B": [2], " B": [2], "\nB": [2]}[text]

        class FakeModel(torch.nn.Module):
            def forward(self, x, *, think_steps=0, **kwargs):
                logits = torch.zeros((1, x.shape[1], 3), device=x.device)
                logits[:, -1, 1] = 1.0
                logits[:, -1, 2] = 3.0
                return logits

        row = {"qtrm_prompt": "prompt", "answer": "B", "options": ["a", "b"]}
        loss = module.preserve_option_kl_loss(
            FakeModel(),
            FakeModel(),
            FakeTokenizer(),
            row,
            seq_len=8,
            think_steps=4,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.allclose(loss, torch.tensor(0.0)))

    def test_parse_depths_handles_csv(self):
        module = _load_script()

        self.assertEqual(module.parse_depths("0, 1,2"), (0, 1, 2))

    def test_depth_gain_loss_only_penalizes_when_full_depth_is_not_better(self):
        module = _load_script()

        class FakeTokenizer:
            def encode(self, text):
                return {"prompt": [0], "A": [1], " A": [1], "\nA": [1], "B": [2], " B": [2], "\nB": [2]}[text]

        class FakeModel(torch.nn.Module):
            def forward(self, x, *, think_steps=0, **kwargs):
                logits = torch.zeros((1, x.shape[1], 3), device=x.device)
                logits[:, -1, 2] = 4.0 if int(think_steps) >= 4 else 1.0
                return logits

        row = {"qtrm_prompt": "prompt", "answer": "B", "options": ["a", "b"]}
        loss = module.depth_gain_loss_for_row(
            FakeModel(),
            FakeTokenizer(),
            row,
            seq_len=8,
            full_think_steps=4,
            shallow_depths=(0, 1),
            device=torch.device("cpu"),
            target_rendering="space",
            margin=0.25,
        )

        self.assertTrue(torch.allclose(loss, torch.tensor(0.0)))


if __name__ == "__main__":
    unittest.main()
