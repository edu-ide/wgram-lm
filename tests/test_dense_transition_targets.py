import importlib.util
from pathlib import Path
import tempfile
import unittest


def load_dense_script():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "232_build_dense_transition_targets.py"
    )
    spec = importlib.util.spec_from_file_location("dense_transition_targets", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DenseTransitionTargetTests(unittest.TestCase):
    def test_densify_list_transform_fills_hold_after_final(self):
        module = load_dense_script()
        row = {
            "task_family": "list_transform",
            "answer_aliases": ["8,4"],
            "depth_targets": {"1": "4,2", "2": "8,4", "4": "8,4", "8": "8,4"},
            "transition_state_codes": {"1": 0, "2": 1, "4": 3, "8": 3},
            "transition_finality_targets": {"1": 0, "2": 1, "4": 1, "8": 1},
            "solver_trace": [
                {"depth": 1, "operation": "filter_even", "state_text": "4,2"},
                {"depth": 2, "operation": "double_filtered", "state_text": "8,4"},
                {"depth": 4, "state_text": "8,4"},
                {"depth": 8, "state_text": "8,4"},
            ],
            "latent_action_trace": [
                {"depth": 1, "action_code": 0, "action_name": "extract_or_unary_transform"},
                {"depth": 2, "action_code": 1, "action_name": "compose_from_previous"},
                {"depth": 4, "action_code": 3, "action_name": "hold_final"},
                {"depth": 8, "action_code": 3, "action_name": "hold_final"},
            ],
            "latent_action_codebook_version": "role_v1",
        }

        dense = module.densify_transition_targets(row, max_depth=8)

        self.assertEqual(dense["transition_state_codes"]["1"], 0)
        self.assertEqual(dense["transition_state_codes"]["2"], 1)
        self.assertEqual(dense["transition_state_codes"]["3"], 3)
        self.assertEqual(dense["transition_state_codes"]["8"], 3)
        self.assertEqual(dense["transition_finality_targets"]["1"], 0)
        self.assertEqual(dense["transition_finality_targets"]["2"], 1)
        self.assertEqual(dense["transition_finality_targets"]["3"], 1)
        self.assertEqual(dense["depth_targets"]["3"], "8,4")
        self.assertEqual(dense["solver_trace"][0]["operation"], "filter_even")
        self.assertEqual(dense["solver_trace"][1]["operation"], "double_filtered")
        self.assertEqual(dense["solver_trace"][2]["operation"], "hold_final")
        self.assertTrue(dense["dense_transition_targets_applied"])

    def test_densify_arithmetic_moves_final_step_to_depth_three(self):
        module = load_dense_script()
        row = {
            "task_family": "arithmetic_chain",
            "answer_aliases": ["17"],
            "solver_trace": [
                {"depth": 1, "state_text": "10"},
                {"depth": 2, "state_text": "20"},
                {"depth": 4, "state_text": "17"},
                {"depth": 8, "state_text": "17"},
            ],
            "latent_action_trace": [
                {"depth": 1, "action_code": 0, "action_name": "extract_or_unary_transform"},
                {"depth": 2, "action_code": 1, "action_name": "compose_from_previous"},
                {"depth": 4, "action_code": 2, "action_name": "final_compose_from_previous"},
                {"depth": 8, "action_code": 3, "action_name": "hold_final"},
            ],
            "latent_action_codebook_version": "role_v1",
        }

        dense = module.densify_transition_targets(row, max_depth=8)

        self.assertEqual(dense["transition_state_codes"]["3"], 2)
        self.assertEqual(dense["transition_finality_targets"]["3"], 1)
        self.assertEqual(dense["transition_state_codes"]["4"], 3)
        self.assertEqual(dense["transition_finality_targets"]["4"], 1)

    def test_action_terminal_finality_ignores_accidental_answer_match(self):
        module = load_dense_script()
        row = {
            "task_family": "boolean_logic",
            "answer_aliases": ["FALSE"],
            "solver_trace": [
                {"depth": 1, "state_text": "TRUE"},
                {"depth": 2, "state_text": "FALSE"},
                {"depth": 4, "state_text": "FALSE"},
                {"depth": 8, "state_text": "FALSE"},
            ],
            "latent_action_trace": [
                {"depth": 1, "action_code": 0, "action_name": "extract_or_unary_nonterminal"},
                {"depth": 2, "action_code": 2, "action_name": "compose_from_previous_nonterminal"},
                {"depth": 4, "action_code": 3, "action_name": "final_compose_from_previous_terminal"},
                {"depth": 8, "action_code": 4, "action_name": "hold_final"},
            ],
            "latent_action_codebook_version": "terminal_v2",
        }

        dense = module.densify_transition_targets(
            row,
            max_depth=8,
            finality_mode="action_terminal",
        )

        self.assertEqual(dense["depth_targets"]["2"], "FALSE")
        self.assertEqual(dense["transition_state_codes"]["2"], 2)
        self.assertEqual(dense["transition_finality_targets"]["2"], 0)
        self.assertEqual(dense["transition_finality_targets"]["3"], 1)

    def test_build_dense_file_writes_rows_and_summary(self):
        module = load_dense_script()
        row = {
            "answer_aliases": ["ok"],
            "solver_trace": [{"depth": 1, "state_text": "ok"}],
            "latent_action_trace": [{"depth": 1, "action_code": 3}],
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "in.jsonl"
            output_path = tmp_path / "out.jsonl"
            input_path.write_text(__import__("json").dumps(row) + "\n", encoding="utf-8")

            summary = module.build_dense_file(
                input_jsonl=input_path,
                output_jsonl=output_path,
                max_depth=2,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["max_depth"], 2)


if __name__ == "__main__":
    unittest.main()
