from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "609_eval_stage102f_prompt_provenance_frontend.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage102f_prompt_provenance_frontend", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage102FPromptProvenanceFrontendTests(unittest.TestCase):
    def test_prompt_only_cards_match_compiled_graph_features(self) -> None:
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

        original = module.prompt_to_graph_features(row["original_prompt"])
        counterfactual = module.prompt_to_graph_features(row["counterfactual_prompt"])

        self.assertEqual(0, original["source_index"])
        self.assertEqual(0.0, original["source_verified"])
        self.assertEqual(1.0, original["claim_supported"])
        self.assertEqual(1, counterfactual["source_index"])
        self.assertEqual(1.0, counterfactual["source_verified"])
        self.assertEqual(1.0, counterfactual["claim_supported"])
        self.assertNotIn("verified_source", original)
        self.assertNotIn("original_source", original)

    def test_prompt_only_world_cards_include_context_and_observation(self) -> None:
        module = load_module()
        prompt = (
            "Source ledger:\n"
            "S1 = verified\n"
            "S2 = unverified\n"
            "Claim: door locked.\n"
            "Real world:\n"
            "Evidence source: S1\n"
            "Evidence value: locked\n"
            "Q: Can answer now? yes or no.\n"
            "A:"
        )

        card = module.prompt_to_world_card(prompt)

        self.assertEqual(0, card["source_index"])
        self.assertEqual(0, card["verified_source_index"])
        self.assertEqual(0, card["context_source_index"])
        self.assertEqual(0, card["context_verified_source_index"])
        self.assertEqual(1.0, card["expected_source_verified"])
        self.assertEqual(1.0, card["observed_source_verified"])
        self.assertEqual(1.0, card["expected_claim_supported"])
        self.assertEqual(1.0, card["claim_supported"])

    def test_eval_reports_template_frontend_coverage_without_answer_fields(self) -> None:
        module = load_module()
        rows = [
            {
                "id": "row0",
                "claim": "route changed.",
                "verified_source": "S1",
                "original_source": "S1",
                "counterfactual_source": "S2",
                "original_prompt": (
                    "Source ledger:\n"
                    "S1 = verified\n"
                    "S2 = unverified\n"
                    "Claim: route changed.\n"
                    "Real world:\n"
                    "Evidence source: S1\n"
                    "Evidence value: changed\n"
                    "Q: Can answer now? yes or no.\n"
                    "A:"
                ),
                "counterfactual_prompt": (
                    "Source ledger:\n"
                    "S1 = verified\n"
                    "S2 = unverified\n"
                    "Claim: route changed.\n"
                    "Imagined change:\n"
                    "Evidence source: S2\n"
                    "Evidence value: changed\n"
                    "Q: Can answer now? yes or no.\n"
                    "A:"
                ),
            }
        ]

        report = module.evaluate_prompt_frontend(rows)

        self.assertEqual(2, report["cards"])
        self.assertEqual(1.0, report["graph_feature_accuracy"])
        self.assertEqual(1.0, report["world_card_accuracy"])
        self.assertTrue(report["accepted"])


if __name__ == "__main__":
    unittest.main()
