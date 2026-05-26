from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "605_train_stage102b_provenance_graph_reasoner.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage102b_provenance_graph_reasoner_train", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage102BProvenanceGraphReasonerTrainTests(unittest.TestCase):
    def test_build_graph_features_separates_ledger_source_and_support(self) -> None:
        module = load_module()
        row = {
            "original_source": "S1",
            "counterfactual_source": "S2",
            "original_answer": " yes",
            "counterfactual_answer": " no",
        }

        original = module.build_graph_features(row, "original")
        counterfactual = module.build_graph_features(row, "counterfactual")

        self.assertEqual(0, original["source_index"])
        self.assertEqual(1, counterfactual["source_index"])
        self.assertEqual(1.0, original["source_verified"])
        self.assertEqual(0.0, counterfactual["source_verified"])
        self.assertEqual(1.0, original["claim_supported"])

    def test_reasoner_outputs_authority_vector_and_gate(self) -> None:
        module = load_module()
        reasoner = module.ProvenanceGraphReasoner(d_model=8)
        features = [
            {"source_index": 0, "source_verified": 1.0, "claim_supported": 1.0},
            {"source_index": 1, "source_verified": 0.0, "claim_supported": 1.0},
        ]

        register, metrics = reasoner(features, device=torch.device("cpu"))

        self.assertEqual((2, 8), tuple(register.shape))
        self.assertEqual(2, metrics["rows"])
        self.assertGreater(metrics["mean_authority"], 0.0)
        self.assertLess(metrics["mean_authority"], 1.0)
        self.assertFalse(torch.allclose(register[0], register[1]))

    def test_shuffle_source_ids_changes_features(self) -> None:
        module = load_module()
        row = {"original_source": "S1", "counterfactual_source": "S2"}

        normal = module.build_graph_features(row, "original", source_id_shuffle=False)
        shuffled = module.build_graph_features(row, "original", source_id_shuffle=True)

        self.assertNotEqual(normal["source_index"], shuffled["source_index"])
        self.assertNotEqual(normal["source_verified"], shuffled["source_verified"])


if __name__ == "__main__":
    unittest.main()
