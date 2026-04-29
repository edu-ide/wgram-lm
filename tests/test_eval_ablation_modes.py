import importlib.util
import io
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


def load_eval_module():
    path = Path("scripts/92_eval_qtrm_logits.py")
    spec = importlib.util.spec_from_file_location("eval_qtrm_logits", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EvalAblationModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_eval_module()

    def test_cli_exposes_ablation_modes(self):
        parser = self.module.build_arg_parser()

        for mode in ("residual", "donor_only", "workspace_off", "core_off"):
            args = parser.parse_args(["--ablation-mode", mode])
            self.assertEqual(args.ablation_mode, mode)

    def test_forward_ablation_kwargs_are_explicit(self):
        self.assertEqual(
            self.module.forward_ablation_kwargs("residual"),
            {"disable_workspace": False, "disable_core": False},
        )
        self.assertEqual(
            self.module.forward_ablation_kwargs("workspace_off"),
            {"disable_workspace": True, "disable_core": False},
        )
        self.assertEqual(
            self.module.forward_ablation_kwargs("core_off"),
            {"disable_workspace": False, "disable_core": True},
        )

    def test_donor_only_forces_donor_logits_as_base_policy(self):
        model = types.SimpleNamespace(
            cfg=types.SimpleNamespace(qtrm_logits_scale=0.1, donor_logits_scale=0.25)
        )

        self.module.apply_ablation_mode(model, "donor_only")

        self.assertEqual(model.cfg.qtrm_logits_scale, 0.0)
        self.assertEqual(model.cfg.donor_logits_scale, 1.0)

    def test_eval_script_records_ablation_mode(self):
        text = Path("scripts/92_eval_qtrm_logits.py").read_text(encoding="utf-8")

        self.assertIn('"ablation_mode"', text)
        self.assertIn("disable_workspace", text)
        self.assertIn("disable_core", text)

    def test_cli_can_enable_core_halt_for_eval(self):
        parser = self.module.build_arg_parser()

        args = parser.parse_args(["--enable-core-halt"])

        self.assertTrue(args.enable_core_halt)

    def test_eval_script_records_core_halt_telemetry(self):
        text = Path("scripts/92_eval_qtrm_logits.py").read_text(encoding="utf-8")

        self.assertIn("enable_core_halt", text)
        self.assertIn('"core_halt"', text)
        self.assertIn('"core_steps"', text)
        self.assertIn('"core_halted"', text)

    def test_core_halt_telemetry_serializes_tensor_outputs(self):
        import torch

        record = self.module.core_halt_telemetry(
            {
                "core_q_halt_logits": torch.tensor([[0.1, 0.7], [0.2, 0.8]]),
                "core_q_continue_logits": torch.tensor([[0.9, 0.3], [0.8, 0.2]]),
                "core_steps": torch.tensor([2, 1]),
                "core_halted": torch.tensor([True, False]),
            },
            enabled=True,
        )

        self.assertTrue(record["enabled"])
        self.assertEqual(record["core_steps"], [2, 1])
        self.assertEqual(record["core_halted"], [True, False])
        self.assertEqual(record["q_halt_steps"], 2)
        self.assertAlmostEqual(record["q_halt_last_mean"], 0.75, places=6)
        self.assertEqual(record["q_continue_steps"], 2)

    def test_json_mode_redirects_donor_init_stdout_to_stderr(self):
        original = self.module.QwenDonorAdapter

        class FakeDonor:
            def __init__(self, cfg):
                print(f"donor log for {cfg}")

        self.module.QwenDonorAdapter = FakeDonor
        try:
            stdout = io.StringIO()
            stderr = io.StringIO()
            cfg = types.SimpleNamespace(donor="fake-donor-cfg")

            with redirect_stdout(stdout), redirect_stderr(stderr):
                donor = self.module.build_donor(cfg, no_donor=False, json_mode=True)

            self.assertIsInstance(donor, FakeDonor)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("donor log for fake-donor-cfg", stderr.getvalue())
        finally:
            self.module.QwenDonorAdapter = original


if __name__ == "__main__":
    unittest.main()
