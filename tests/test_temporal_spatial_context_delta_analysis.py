import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_delta_script():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "209_analyze_temporal_spatial_context_delta.py"
    )
    spec = importlib.util.spec_from_file_location("temporal_spatial_context_delta", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TemporalSpatialContextDeltaAnalysisTests(unittest.TestCase):
    def test_analyze_pairs_reports_hit_and_logprob_deltas(self):
        module = load_delta_script()

        records = [
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_8_no_evidence",
                "task_family": "temporal_freshness",
                "hit": True,
                "completion": "green",
                "answer_aliases": ["green"],
                "choice_scores": [
                    {"choice": "green", "logprob": -1.0},
                    {"choice": "red", "logprob": -3.0},
                ],
            },
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_8_temporal_spatial_off_no_evidence",
                "task_family": "temporal_freshness",
                "hit": False,
                "completion": "red",
                "answer_aliases": ["green"],
                "choice_scores": [
                    {"choice": "green", "logprob": -2.0},
                    {"choice": "red", "logprob": -1.0},
                ],
            },
            {
                "id": "case-b",
                "mode": "qtrm_core_steps_8_no_evidence",
                "task_family": "spatial_relation",
                "hit": False,
                "completion": "UNKNOWN",
                "answer_aliases": ["red key"],
                "choice_scores": [{"choice": "red key", "logprob": -4.0}],
            },
            {
                "id": "case-b",
                "mode": "qtrm_core_steps_8_temporal_spatial_off_no_evidence",
                "task_family": "spatial_relation",
                "hit": False,
                "completion": "UNKNOWN",
                "answer_aliases": ["red key"],
                "choice_scores": [{"choice": "red key", "logprob": -5.0}],
            },
        ]

        summary = module.analyze_records(records)

        self.assertEqual(summary["paired_count"], 2)
        self.assertEqual(summary["context_on_only_correct_count"], 1)
        self.assertEqual(summary["context_off_only_correct_count"], 0)
        self.assertEqual(summary["changed_completion_count"], 1)
        self.assertEqual(summary["chosen_logprob_delta_mean"], 1.0)
        self.assertEqual(
            summary["by_task_family"]["temporal_freshness"]["context_on_only_correct_count"],
            1,
        )

    def test_write_summary_outputs_json(self):
        module = load_delta_script()
        records = [
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_8_no_evidence",
                "hit": False,
                "completion": "red",
                "answer_aliases": ["green"],
                "choice_scores": [{"choice": "green", "logprob": -1.0}],
            },
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_8_temporal_spatial_off_no_evidence",
                "hit": False,
                "completion": "red",
                "answer_aliases": ["green"],
                "choice_scores": [{"choice": "green", "logprob": -1.5}],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "summary.json"
            module.write_summary(records, out)
            summary = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(summary["paired_count"], 1)
        self.assertEqual(summary["chosen_logprob_delta_mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
