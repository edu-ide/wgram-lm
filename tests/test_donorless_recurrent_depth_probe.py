import importlib.util
from pathlib import Path
import sys
import unittest

import torch


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "260_train_donorless_recurrent_depth_probe.py"
    )
    spec = importlib.util.spec_from_file_location("donorless_depth_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DonorlessRecurrentDepthProbeTests(unittest.TestCase):
    def test_apply_op_matches_modular_targets(self):
        module = _load_module()

        self.assertEqual(module.apply_op(10, 3, 97), 11)
        self.assertEqual(module.apply_op(10, 0, 97), 5)
        self.assertEqual(module.apply_op(10, module.NOOP_ID, 97), 10)

    def test_build_cases_emits_prefix_targets(self):
        module = _load_module()

        cases = module.build_cases(count=3, seed=7, max_program_len=4, modulus=31)

        self.assertEqual(len(cases), 3)
        for case in cases:
            self.assertEqual(len(case.op_ids), 4)
            self.assertEqual(len(case.targets), 5)
            value = case.start
            self.assertEqual(case.targets[0], value)
            for index, op_id in enumerate(case.op_ids, start=1):
                value = module.apply_op(value, op_id, 31)
                self.assertEqual(case.targets[index], value)

    def test_model_outputs_one_logit_row_per_depth(self):
        module = _load_module()

        model = module.DonorlessRecurrentDepthProbe(
            modulus=17,
            num_ops=module.NOOP_ID + 1,
            d_model=16,
            hidden_dim=24,
        )
        start = torch.tensor([1, 2], dtype=torch.long)
        op_ids = torch.tensor(
            [
                [0, 1, module.NOOP_ID],
                [2, 3, module.NOOP_ID],
            ],
            dtype=torch.long,
        )

        logits = model(start_ids=start, op_ids=op_ids)

        self.assertEqual(tuple(logits.shape), (2, 4, 17))

    def test_transition_table_model_outputs_one_logit_row_per_depth(self):
        module = _load_module()

        model = module.DonorlessTransitionTableDepthProbe(
            modulus=17,
            num_ops=module.NOOP_ID + 1,
        )
        start = torch.tensor([1, 2], dtype=torch.long)
        op_ids = torch.tensor(
            [
                [0, 1, module.NOOP_ID],
                [2, 3, module.NOOP_ID],
            ],
            dtype=torch.long,
        )

        logits = model(start_ids=start, op_ids=op_ids)

        self.assertEqual(tuple(logits.shape), (2, 4, 17))

    def test_direct_transition_loss_is_available_for_transition_table(self):
        module = _load_module()

        model = module.DonorlessTransitionTableDepthProbe(
            modulus=17,
            num_ops=module.NOOP_ID + 1,
        )
        op_ids = torch.tensor([[0, 1]], dtype=torch.long)
        targets = torch.tensor([[3, 4, 5]], dtype=torch.long)

        loss = module.direct_transition_loss_for_batch(model, op_ids, targets)

        self.assertEqual(tuple(loss.shape), ())
        self.assertGreater(float(loss.item()), 0.0)

    def test_parser_accepts_gate_args(self):
        module = _load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--out-dir",
                "local_eval/donorless-depth",
                "--steps",
                "10",
                "--depths",
                "0,1,2",
                "--accept-min-final-exact",
                "0.9",
            ]
        )

        self.assertEqual(args.out_dir, "local_eval/donorless-depth")
        self.assertEqual(args.steps, 10)
        self.assertEqual(args.model_kind, "transition_table")
        self.assertEqual(args.direct_transition_ce_weight, 1.0)
        self.assertEqual(module.parse_depths(args.depths), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
