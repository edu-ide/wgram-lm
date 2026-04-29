import importlib.util
import types
import unittest
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


if __name__ == "__main__":
    unittest.main()
