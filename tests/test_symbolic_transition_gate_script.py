from pathlib import Path
import importlib.util
import unittest


def _load_module():
    path = Path("scripts/201_eval_symbolic_transition_gate.py")
    spec = importlib.util.spec_from_file_location("symbolic_transition_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SymbolicTransitionGateScriptTests(unittest.TestCase):
    def test_transition_target_uses_depth_target_before_final_answer(self):
        module = _load_module()
        row = {
            "answer_aliases": ["final"],
            "depth_targets": {"1": "stage-one", "2": "stage-two"},
        }

        self.assertEqual(module.transition_target_for_step(row, 1), "stage-one")
        self.assertEqual(module.transition_target_for_step(row, 2), "stage-two")
        self.assertEqual(module.transition_target_for_step(row, 4), "final")

    def test_transition_choices_include_target_depth_targets_and_final_choices_once(self):
        module = _load_module()
        row = {
            "answer_aliases": ["final"],
            "choices": ["final", "wrong"],
            "depth_targets": {"1": "stage-one", "2": "stage-two", "4": "final"},
        }

        choices = module.transition_choices_for_step(row, 1)

        self.assertEqual(choices[0], "stage-one")
        self.assertIn("stage-two", choices)
        self.assertIn("final", choices)
        self.assertIn("wrong", choices)
        self.assertEqual(len(choices), len(set(choices)))

    def test_summarize_records_reports_step_and_overall_accuracy(self):
        module = _load_module()
        records = [
            {"core_steps": 1, "hit": True},
            {"core_steps": 1, "hit": False},
            {"core_steps": 2, "hit": True},
        ]

        summary = module.summarize_records(records)

        self.assertEqual(summary["total_records"], 3)
        self.assertAlmostEqual(summary["accuracy"], 2 / 3)
        self.assertEqual(summary["by_core_steps"]["1"]["hit_count"], 1)
        self.assertEqual(summary["by_core_steps"]["1"]["total"], 2)
        self.assertAlmostEqual(summary["by_core_steps"]["2"]["accuracy"], 1.0)

    def test_cli_accepts_transition_scoring_defaults(self):
        module = _load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "config.yaml",
                "--checkpoint",
                "last.pt",
                "--cases",
                "cases.jsonl",
                "--out",
                "out.jsonl",
            ]
        )

        self.assertEqual(args.core_steps, "1,2,4,8")
        self.assertEqual(args.scoring, "causal_forced_choice")


if __name__ == "__main__":
    unittest.main()
