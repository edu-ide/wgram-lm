from pathlib import Path
import importlib.util
import unittest

import torch


def _load_module():
    path = Path("scripts/221_eval_qtrm_primitive_transition_rollout.py")
    spec = importlib.util.spec_from_file_location("qtrm_primitive_rollout_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QTRMPrimitiveTransitionRolloutEvalTests(unittest.TestCase):
    def test_operation_names_from_logits_decodes_argmax_sequence(self):
        module = _load_module()

        names = module.operation_names_from_logits(
            torch.tensor([[0.1, 2.0], [3.0, 0.2]]),
            {0: "hold_final", 1: "add_operands"},
        )

        self.assertEqual(names, ["add_operands", "hold_final"])

    def test_summarize_rollouts_counts_operation_state_and_final(self):
        module = _load_module()

        summary = module.summarize_rollouts(
            [
                {
                    "task_family": "arithmetic_chain",
                    "rollout": {
                        "operation_exact_count": 4,
                        "state_exact_count": 4,
                        "total_steps": 4,
                        "final_exact_match": True,
                    },
                },
                {
                    "task_family": "arithmetic_chain",
                    "rollout": {
                        "operation_exact_count": 3,
                        "state_exact_count": 2,
                        "total_steps": 4,
                        "final_exact_match": False,
                    },
                },
            ]
        )

        self.assertEqual(summary["operation_exact"], "7/8")
        self.assertEqual(summary["state_exact"], "6/8")
        self.assertEqual(summary["final_exact"], "1/2")
        self.assertEqual(summary["by_family"]["arithmetic_chain"]["final_exact"], "1/2")


if __name__ == "__main__":
    unittest.main()
