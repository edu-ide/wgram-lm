from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "608_train_stage102e_world_model_gated_answer_register.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage102e_world_model_gated_answer_register", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage102EWorldModelGatedAnswerRegisterTests(unittest.TestCase):
    def test_cases_use_answerable_side_and_make_corruptions_no(self) -> None:
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

        cases = module.build_world_gated_answer_cases(row)

        self.assertEqual(4, len(cases))
        self.assertEqual("counterfactual", cases[0]["side"])
        self.assertEqual("clean", cases[0]["corruption"])
        self.assertEqual(" yes", cases[0]["answer"])
        self.assertTrue(all(case["answer"] == " no" for case in cases[1:]))
        self.assertTrue(all(case["negative_answer"] == " yes" for case in cases[1:]))

    def test_world_model_gate_changes_register_and_can_be_disabled(self) -> None:
        module = load_module()
        graph_reasoner = module.STAGE102B.ProvenanceGraphReasoner(d_model=8)
        world_model = module.STAGE102D.ProvenanceDataWorldModel(d_model=4, max_sources=4)
        gated = module.WorldModelGatedAnswerRegister(
            d_model=8,
            graph_reasoner=graph_reasoner,
            world_model=world_model,
            world_d_model=4,
        )
        graph_features = {"source_index": 0, "source_verified": 1.0, "claim_supported": 1.0}
        clean_world = {
            "source_index": 0,
            "verified_source_index": 0,
            "context_source_index": 0,
            "context_verified_source_index": 0,
            "expected_source_verified": 1.0,
            "expected_claim_supported": 1.0,
            "observed_source_verified": 1.0,
            "claim_supported": 1.0,
        }
        corrupt_world = dict(clean_world)
        corrupt_world["source_index"] = 1

        clean_register, clean_metrics = gated(
            graph_features,
            clean_world,
            device=torch.device("cpu"),
        )
        corrupt_register, corrupt_metrics = gated(
            graph_features,
            corrupt_world,
            device=torch.device("cpu"),
        )
        clean_off, _ = gated(
            graph_features,
            clean_world,
            device=torch.device("cpu"),
            world_off=True,
        )
        corrupt_off, _ = gated(
            graph_features,
            corrupt_world,
            device=torch.device("cpu"),
            world_off=True,
        )

        self.assertEqual((1, 8), tuple(clean_register.shape))
        self.assertIn("world_energy", clean_metrics)
        self.assertIn("world_gate", corrupt_metrics)
        self.assertFalse(torch.allclose(clean_register, corrupt_register))
        self.assertTrue(torch.allclose(clean_off, corrupt_off))

    def test_parser_supports_fast_full_gate_options(self) -> None:
        module = load_module()
        parser = module.build_arg_parser()

        args = parser.parse_args(
            [
                "--answer-checkpoint",
                "answer.pt",
                "--world-model-checkpoint",
                "world.pt",
                "--out-dir",
                "out",
                "--depths",
                "2",
                "4",
                "8",
                "16",
                "--eval-depths",
                "16",
                "--skip-eval-before",
            ]
        )

        self.assertEqual([2, 4, 8, 16], args.depths)
        self.assertEqual([16], args.eval_depths)
        self.assertTrue(args.skip_eval_before)


if __name__ == "__main__":
    unittest.main()
