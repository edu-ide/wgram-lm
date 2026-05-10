from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "311_summarize_list_order_lm_gate.py"
    spec = importlib.util.spec_from_file_location("list_order_lm_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def row(mode: str, case_id: str, completion: str, answer: str, *, hit: bool):
    return {
        "mode": mode,
        "id": case_id,
        "task_family": "list_transform",
        "completion": completion,
        "answer_aliases": [answer],
        "hit": hit,
        "choice_scores": [
            {"choice": completion, "logprob": -1.0},
            {"choice": answer, "logprob": -2.0 if not hit else -1.0},
        ],
    }


class ListOrderLmGateSummaryTests(unittest.TestCase):
    def test_rejects_when_ablation_ties_core(self) -> None:
        module = load_module()
        cases = {
            "list-0": {
                "id": "list-0",
                "answer_aliases": ["8,4"],
                "depth_targets": {"1": "4,2"},
            }
        }
        rows = [
            row("donor_only_no_evidence", "list-0", "4,2", "8,4", hit=False),
            row("qtrm_core_off_no_evidence", "list-0", "__FORCED_CHOICE_TIE__", "8,4", hit=False),
            row("qtrm_core_steps_8_no_evidence", "list-0", "8,4", "8,4", hit=True),
            row(
                "qtrm_core_steps_8_transition_state_off_no_evidence",
                "list-0",
                "8,4",
                "8,4",
                hit=True,
            ),
        ]

        report = module.summarize_gate(
            rows,
            cases,
            core_mode="qtrm_core_steps_8_no_evidence",
            ablation_mode="qtrm_core_steps_8_transition_state_off_no_evidence",
            baseline_modes=["donor_only_no_evidence", "qtrm_core_off_no_evidence"],
            min_overall_hits=1,
            require_ablation_drop=True,
        )

        self.assertFalse(report["accepted"])
        self.assertIn("ablation ties or beats core list hits", report["reject_reasons"])

    def test_accepts_when_core_beats_baselines_and_ablation_drops(self) -> None:
        module = load_module()
        cases = {
            "list-0": {
                "id": "list-0",
                "answer_aliases": ["8,4"],
                "depth_targets": {"1": "4,2"},
            }
        }
        rows = [
            row("donor_only_no_evidence", "list-0", "4,2", "8,4", hit=False),
            row("qtrm_core_off_no_evidence", "list-0", "__FORCED_CHOICE_TIE__", "8,4", hit=False),
            row("qtrm_core_steps_8_no_evidence", "list-0", "8,4", "8,4", hit=True),
            row(
                "qtrm_core_steps_8_transition_state_off_no_evidence",
                "list-0",
                "4,2",
                "8,4",
                hit=False,
            ),
        ]

        report = module.summarize_gate(
            rows,
            cases,
            core_mode="qtrm_core_steps_8_no_evidence",
            ablation_mode="qtrm_core_steps_8_transition_state_off_no_evidence",
            baseline_modes=["donor_only_no_evidence", "qtrm_core_off_no_evidence"],
            min_overall_hits=1,
            require_ablation_drop=True,
        )

        self.assertTrue(report["accepted"])
        self.assertEqual(report["decision"], "accepted_l2")


if __name__ == "__main__":
    unittest.main()
