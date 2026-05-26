from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "607_train_stage102d_provenance_data_world_model.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage102d_provenance_data_world_model", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage102DProvenanceDataWorldModelTests(unittest.TestCase):
    def test_world_examples_do_not_need_answer_labels(self) -> None:
        module = load_module()
        row = {
            "id": "row0",
            "claim": "badge valid.",
            "verified_source": "S2",
            "original_source": "S1",
            "counterfactual_source": "S2",
            "original_prompt": (
                "Source ledger:\n"
                "S1 = unverified\n"
                "S2 = verified\n"
                "Claim: badge valid.\n"
                "Real world:\n"
                "Evidence source: S1\n"
                "Evidence value: valid\n"
                "Q: Can answer now? yes or no.\n"
                "A:"
            ),
            "counterfactual_prompt": (
                "Source ledger:\n"
                "S1 = unverified\n"
                "S2 = verified\n"
                "Claim: badge valid.\n"
                "Imagined change:\n"
                "Evidence source: S2\n"
                "Evidence value: valid\n"
                "Q: Can answer now? yes or no.\n"
                "A:"
            ),
        }

        examples = module.build_world_model_examples(row, side="original")

        self.assertEqual(4, len(examples))
        self.assertEqual("clean", examples[0]["corruption"])
        self.assertEqual(1.0, examples[0]["is_clean"])
        self.assertIn("context_source_index", examples[0])
        self.assertIn("context_verified_source_index", examples[0])
        self.assertIn("expected_source_verified", examples[0])
        self.assertIn("expected_claim_supported", examples[0])
        self.assertNotIn("original_answer", examples[0])
        self.assertNotIn("counterfactual_answer", examples[0])
        self.assertTrue(any(item["corruption"] == "trust_edge_conflict" for item in examples))
        self.assertTrue(any(item["corruption"] == "source_id_conflict" for item in examples))
        self.assertTrue(any(item["corruption"] == "support_conflict" for item in examples))

    def test_corruptions_break_internal_provenance_consistency(self) -> None:
        module = load_module()
        row = {
            "id": "row1",
            "claim": "door locked.",
            "verified_source": "S1",
            "original_source": "S1",
            "counterfactual_source": "S2",
            "original_prompt": (
                "Source ledger:\n"
                "S1 = verified\n"
                "S2 = unverified\n"
                "Claim: door locked.\n"
                "Real world:\n"
                "Evidence source: S1\n"
                "Evidence value: locked\n"
                "Q: Can answer now? yes or no.\n"
                "A:"
            ),
            "counterfactual_prompt": "",
        }

        by_kind = {item["corruption"]: item for item in module.build_world_model_examples(row, "original")}

        self.assertEqual(1.0, by_kind["clean"]["observed_source_verified"])
        self.assertEqual(
            by_kind["clean"]["context_source_index"],
            by_kind["source_id_conflict"]["context_source_index"],
        )
        self.assertEqual(0.0, by_kind["trust_edge_conflict"]["observed_source_verified"])
        self.assertNotEqual(by_kind["clean"]["source_index"], by_kind["source_id_conflict"]["source_index"])
        self.assertEqual(0.0, by_kind["support_conflict"]["claim_supported"])

    def test_world_model_outputs_energy_and_latent(self) -> None:
        module = load_module()
        model = module.ProvenanceDataWorldModel(d_model=8, max_sources=4)
        examples = [
            {
                "source_index": 0,
                "verified_source_index": 0,
                "context_source_index": 0,
                "context_verified_source_index": 0,
                "expected_source_verified": 1.0,
                "expected_claim_supported": 1.0,
                "observed_source_verified": 1.0,
                "claim_supported": 1.0,
            },
            {
                "source_index": 1,
                "verified_source_index": 0,
                "context_source_index": 0,
                "context_verified_source_index": 0,
                "expected_source_verified": 1.0,
                "expected_claim_supported": 1.0,
                "observed_source_verified": 1.0,
                "claim_supported": 1.0,
            },
        ]

        energy, latent = model(examples, device=torch.device("cpu"))

        self.assertEqual((2,), tuple(energy.shape))
        self.assertEqual((2, 8), tuple(latent.shape))
        self.assertTrue(torch.isfinite(energy).all())
        self.assertTrue(torch.isfinite(latent).all())


if __name__ == "__main__":
    unittest.main()
