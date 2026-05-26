from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "612_train_stage102z_final_freeform_answer_path.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage102z_final_freeform_answer_path", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage102ZFinalFreeformAnswerPathTests(unittest.TestCase):
    def test_final_prompt_parser_separates_context_and_observation(self) -> None:
        module = load_module()
        prompt = (
            "Provenance context:\n"
            "Trusted source for this claim is S2. Other source S1 is unverified.\n"
            "Claim under review: badge valid.\n"
            "Expected support value: valid.\n"
            "Observation:\n"
            "Observed evidence came from S1.\n"
            "Observed source status: unverified.\n"
            "Observed evidence says valid.\n"
            "Can answer now? yes or no.\n"
            "A:"
        )

        graph, world = module.prompt_to_final_cards(prompt)

        self.assertEqual(1, graph["source_index"])
        self.assertEqual(1.0, graph["source_verified"])
        self.assertEqual(1.0, graph["claim_supported"])
        self.assertEqual(0, world["source_index"])
        self.assertEqual(1, world["context_source_index"])
        self.assertEqual(1, world["verified_source_index"])
        self.assertEqual(0.0, world["observed_source_verified"])
        self.assertEqual(1.0, world["claim_supported"])

    def test_final_cases_use_freeform_prompt_and_same_answer_contract(self) -> None:
        module = load_module()
        row = {
            "id": "row0",
            "claim": "badge valid.",
            "verified_source": "S2",
            "original_source": "S1",
            "counterfactual_source": "S2",
            "original_answer": " no",
            "counterfactual_answer": " yes",
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

        cases = module.build_final_cases([row])

        self.assertEqual(["clean", "source_id_conflict", "trust_edge_conflict", "support_conflict"], [case["corruption"] for case in cases])
        self.assertEqual(" yes", cases[0]["answer"])
        self.assertTrue(all(case["answer"] == " no" for case in cases[1:]))
        self.assertTrue(all("original_source" not in case["prompt"] for case in cases))
        self.assertTrue(all("verified_source" not in case["prompt"] for case in cases))

    def test_support_matching_handles_non_contiguous_human_equivalent_values(self) -> None:
        module = load_module()

        self.assertEqual(1.0, module._support("bay C", "bus bay is C."))
        self.assertEqual(1.0, module._support("platform 4", "platform is 4."))
        self.assertEqual(0.0, module._support("mismatch", "bus bay is C."))


if __name__ == "__main__":
    unittest.main()
