from pathlib import Path
import importlib.util
import json
import tempfile
import unittest

import torch


def _load_module():
    path = Path("scripts/200_eval_core_world_model_transition.py")
    spec = importlib.util.spec_from_file_location("core_world_model_transition_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CoreWorldModelTransitionEvalTests(unittest.TestCase):
    def test_transition_mse_masks_invalid_transitions(self):
        module = _load_module()
        pred = torch.tensor([[[1.0, 1.0], [10.0, 10.0]]])
        target = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
        mask = torch.tensor([[True, False]])

        metrics = module.transition_metrics_from_tensors(pred, target, mask)

        self.assertEqual(metrics["transition_count"], 1)
        self.assertAlmostEqual(metrics["transition_mse"], 1.0)

    def test_load_answer_hits_keys_by_case_id_and_mode(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answers.jsonl"
            path.write_text(
                json.dumps({"id": "case-1", "mode": "qtrm_core_steps_2_no_evidence", "hit": True})
                + "\n",
                encoding="utf-8",
            )

            hits = module.load_answer_hits(path)

        self.assertEqual(hits[("case-1", "qtrm_core_steps_2_no_evidence")], True)

    def test_pearson_returns_zero_for_constant_input(self):
        module = _load_module()

        self.assertEqual(module.pearson_correlation([1.0, 1.0], [0.0, 1.0]), 0.0)

    def test_summarize_records_reports_mean_mse_and_hit_correlation(self):
        module = _load_module()
        records = [
            {
                "mode": "qtrm_core_steps_2_no_evidence",
                "transition_mse": 1.0,
                "transition_count": 1,
                "hit": True,
            },
            {
                "mode": "qtrm_core_steps_2_no_evidence",
                "transition_mse": 3.0,
                "transition_count": 1,
                "hit": False,
            },
        ]

        summary = module.summarize_records(records)

        self.assertEqual(summary["total_records"], 2)
        self.assertEqual(summary["by_mode"]["qtrm_core_steps_2_no_evidence"]["count"], 2)
        self.assertAlmostEqual(
            summary["by_mode"]["qtrm_core_steps_2_no_evidence"]["mean_transition_mse"],
            2.0,
        )
        self.assertLess(summary["transition_mse_hit_pearson"], 0.0)


if __name__ == "__main__":
    unittest.main()
