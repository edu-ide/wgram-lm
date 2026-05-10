import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


def load_probe_script():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "315_train_ordered_list_state_probe.py"
    )
    spec = importlib.util.spec_from_file_location("ordered_list_state_probe", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OrderedListStateProbeTests(unittest.TestCase):
    def test_case_targets_preserve_order_and_distinguish_filtered_from_final(self):
        module = load_probe_script()

        case = module.case_from_values(
            "demo",
            (1, 4, 2, 7, 3),
            max_output_len=4,
        )

        self.assertEqual(case.depth_targets[0], (1, 4, 2, 7))
        self.assertEqual(case.depth_targets[1], (4, 2, -1, -1))
        self.assertEqual(case.depth_targets[2], (8, 4, -1, -1))
        self.assertEqual(case.depth_targets[4], (8, 4, -1, -1))

    def test_batch_targets_use_zero_as_padding_and_value_plus_one_classes(self):
        module = load_probe_script()
        case = module.case_from_values(
            "demo",
            (1, 4, 2, 7, 3),
            max_output_len=4,
        )

        values, targets = module.cases_to_batch(
            [case],
            max_input_len=5,
            max_output_len=4,
            max_depth=4,
            device="cpu",
        )

        self.assertEqual(values.tolist(), [[2, 5, 3, 8, 4]])
        self.assertEqual(
            targets.tolist(),
            [
                [
                    [2, 5, 3, 8],
                    [5, 3, 0, 0],
                    [9, 5, 0, 0],
                    [9, 5, 0, 0],
                    [9, 5, 0, 0],
                ]
            ],
        )

    def test_probe_smoke_profile_writes_report(self):
        module = load_probe_script()
        with tempfile.TemporaryDirectory() as tmp:
            args = module.build_arg_parser().parse_args(
                [
                    "--out-dir",
                    tmp,
                    "--steps",
                    "2",
                    "--train-cases",
                    "8",
                    "--eval-cases",
                    "4",
                    "--batch-size",
                    "4",
                    "--device",
                    "cpu",
                    "--log-every",
                    "0",
                ]
            )

            report = module.train_probe(args)

        self.assertIn(report["decision"], {"accepted_l1", "rejected"})
        self.assertEqual(report["target_level"], "L1 scaffold")
        self.assertIn("eval_metrics", report)

    def test_gate_decision_ignores_op_zero_by_default_for_fixed_operation_probe(self):
        module = load_probe_script()
        args = module.build_arg_parser().parse_args(
            [
                "--out-dir",
                "unused",
                "--accept-min-final-exact",
                "0.90",
                "--accept-min-depth-gain",
                "0.25",
                "--accept-min-ablation-drop",
                "0.20",
            ]
        )
        eval_metrics = {
            "depth1_final_exact": 0.01,
            "depth2_final_exact": 0.94,
            "depth4_final_exact": 0.95,
        }
        ablations = {
            "state_reset": {"depth4_final_exact": 0.02},
            "order_shuffle": {"depth4_final_exact": 0.05},
            "op_zero": {"depth4_final_exact": 0.94},
        }

        self.assertEqual(
            module.decide_gate(eval_metrics, ablations, args),
            "accepted_l1",
        )

        args.require_op_ablation = True
        self.assertEqual(module.decide_gate(eval_metrics, ablations, args), "rejected")


if __name__ == "__main__":
    unittest.main()
