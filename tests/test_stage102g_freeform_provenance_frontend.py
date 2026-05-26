from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "610_eval_stage102g_freeform_provenance_frontend.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage102g_freeform_provenance_frontend", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage102GFreeformProvenanceFrontendTests(unittest.TestCase):
    def test_freeform_prompt_maps_to_same_graph_card(self) -> None:
        module = load_module()
        prompt = (
            "Audit note: S2 is verified for this file; S1 is unverified.\n"
            "The claim under review is: badge valid.\n"
            "Observed evidence came from S2.\n"
            "The evidence says valid.\n"
            "Question: may the answer be given now? yes or no.\n"
            "Answer:"
        )

        card = module.freeform_prompt_to_graph_features(prompt)

        self.assertEqual("S2", card["source_id"])
        self.assertEqual(1, card["source_index"])
        self.assertEqual(1.0, card["source_verified"])
        self.assertEqual(1.0, card["claim_supported"])

    def test_freeform_prompt_support_conflict_is_detected(self) -> None:
        module = load_module()
        prompt = (
            "For provenance, S1 is verified and S2 is unverified.\n"
            "Claim being checked: door locked.\n"
            "Evidence source is S1.\n"
            "Evidence value is open.\n"
            "Can answer now? yes/no.\n"
            "A:"
        )

        card = module.freeform_prompt_to_world_card(prompt)

        self.assertEqual(0, card["source_index"])
        self.assertEqual(0, card["verified_source_index"])
        self.assertEqual(1.0, card["observed_source_verified"])
        self.assertEqual(0.0, card["claim_supported"])
        self.assertEqual(0.0, card["expected_claim_supported"])

    def test_paraphrase_eval_has_no_row_field_dependency(self) -> None:
        module = load_module()
        rows = [
            {
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
        ]

        report = module.evaluate_freeform_frontend(rows)

        self.assertEqual(6, report["cards"])
        self.assertEqual(1.0, report["graph_feature_accuracy"])
        self.assertEqual(1.0, report["world_card_accuracy"])
        self.assertTrue(report["accepted"])


if __name__ == "__main__":
    unittest.main()
