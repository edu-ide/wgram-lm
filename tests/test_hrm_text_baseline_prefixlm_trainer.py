import importlib.util
import sys
import unittest
from pathlib import Path

import torch


def load_module():
    path = Path("scripts/536_train_hrm_text_baseline_prefixlm.py")
    spec = importlib.util.spec_from_file_location("hrm_text_baseline_prefixlm", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class HRMTextBaselinePrefixLMTrainerTests(unittest.TestCase):
    def test_official_hrm_baseline_forward_speaks_with_lm_vocab(self):
        module = load_module()
        model = module.build_official_model(
            vocab_size=32,
            max_seq_len=8,
            hidden_size=16,
            num_heads=4,
            n_layers=1,
            expansion=1.0,
            h_cycles=2,
            l_cycles=2,
        )
        batch = module.make_single_sequence_batch(
            input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.long),
            labels=torch.tensor([-100, -100, 3, 4], dtype=torch.long),
            prefix_len=2,
        )

        _, loss, metrics = model(carry=None, batch=batch, bp_steps=2)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(metrics["accuracy"][1]), 2.0)

    def test_report_marks_baseline_as_official_hrm_text_contract(self):
        module = load_module()

        report = module.build_report(
            dataset_summary={
                "contract": "hrm_text_data_io_prefixlm",
                "vocab_size": 64,
                "seq_len": 8,
            },
            eval_dataset_summary=None,
            args=module.build_arg_parser().parse_args(
                ["--sampled-data", "/tmp/sample", "--out-dir", "/tmp/out"]
            ),
            losses=[{"step": 1, "loss": 4.0, "tokens_seen": 8, "target_tokens_seen": 4}],
            eval_losses=[],
            tokens_seen=8,
            target_tokens_seen=4,
        )

        self.assertEqual(report["model"]["baseline_family"], "official_hrm_text")
        self.assertEqual(report["dataset"]["contract"], "hrm_text_data_io_prefixlm")
        self.assertFalse(report["accepted"])
        self.assertIn("same textbook", report["plain_language_read"].lower())


if __name__ == "__main__":
    unittest.main()
