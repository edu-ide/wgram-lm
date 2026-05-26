from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "611_train_stage102h_learned_provenance_reader.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage102h_learned_provenance_reader", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage102HLearnedProvenanceReaderTests(unittest.TestCase):
    def test_examples_are_built_from_text_and_card_targets(self) -> None:
        module = load_module()
        examples = module.build_reader_examples(max_rows=1)

        self.assertEqual(12, len(examples))
        first = examples[0]
        self.assertIn("text", first)
        self.assertIn("target", first)
        self.assertIn("source_index", first["target"])
        self.assertIn("source_verified", first["target"])
        self.assertIn("claim_supported", first["target"])
        self.assertNotIn("original_source", first["text"])
        self.assertNotIn("verified_source", first["text"])

    def test_reader_forward_outputs_card_logits(self) -> None:
        module = load_module()
        reader = module.LearnedProvenanceReader(vocab_size=128, hidden_dim=16, max_sources=2)
        batch = module.encode_text_batch(["S1 is verified. Evidence came from S1."], seq_len=64)

        output = reader(batch)

        self.assertEqual((1, 2), tuple(output["source_index_logits"].shape))
        self.assertEqual((1,), tuple(output["source_verified_logit"].shape))
        self.assertEqual((1,), tuple(output["claim_supported_logit"].shape))

    def test_tiny_training_gate_accepts_prompt_card_prediction(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            report = module.run_train(
                Namespace(
                    out_dir=tmp,
                    train_rows=4,
                    eval_rows=4,
                    steps=80,
                    batch_size=8,
                    lr=3e-3,
                    weight_decay=0.0,
                    hidden_dim=64,
                    seq_len=192,
                    max_sources=2,
                    seed=7,
                    log_every=80,
                    device="cpu",
                )
            )

        self.assertGreaterEqual(report["heldout"]["card_accuracy"], 0.95)
        self.assertTrue(report["accepted"])


if __name__ == "__main__":
    unittest.main()
