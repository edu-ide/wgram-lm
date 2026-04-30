import importlib.util
import unittest
from pathlib import Path


def load_memory_eval_module():
    path = Path("scripts/95_eval_memory_retrieval.py")
    spec = importlib.util.spec_from_file_location("eval_memory_retrieval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MemoryAblationModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_memory_eval_module()

    def test_mode_settings_accept_workspace_and_core_ablations(self):
        for mode in (
            "qtrm_workspace_off_with_evidence",
            "qtrm_coda_off_with_evidence",
            "qtrm_residual_head_off_with_evidence",
            "qtrm_donor_hidden_off_with_evidence",
            "qtrm_workspace_only_with_evidence",
        ):
            include_evidence, qtrm_scale, donor_scale = self.module.mode_settings(
                mode,
                qtrm_scale=0.1,
                donor_scale=1.0,
            )

            self.assertTrue(include_evidence)
            self.assertEqual(qtrm_scale, 0.1)
            self.assertEqual(donor_scale, 1.0)

        include_evidence, qtrm_scale, donor_scale = self.module.mode_settings(
            "qtrm_core_off_no_evidence",
            qtrm_scale=0.1,
            donor_scale=1.0,
        )

        self.assertFalse(include_evidence)
        self.assertEqual(qtrm_scale, 0.1)
        self.assertEqual(donor_scale, 1.0)

    def test_mode_forward_kwargs_match_ablation_names(self):
        self.assertEqual(
            self.module.mode_forward_kwargs("qtrm_residual_with_evidence"),
            {"disable_workspace": False, "disable_core": False},
        )
        self.assertEqual(
            self.module.mode_forward_kwargs("qtrm_workspace_off_with_evidence"),
            {"disable_workspace": True, "disable_core": False},
        )
        self.assertEqual(
            self.module.mode_forward_kwargs("qtrm_core_off_with_evidence"),
            {"disable_workspace": False, "disable_core": True},
        )
        self.assertEqual(
            self.module.mode_forward_kwargs("qtrm_coda_off_with_evidence"),
            {
                "disable_workspace": False,
                "disable_core": False,
                "disable_coda": True,
                "disable_qtrm_residual": False,
                "disable_donor_context": False,
                "workspace_only_context": False,
            },
        )
        self.assertEqual(
            self.module.mode_forward_kwargs("qtrm_residual_head_off_with_evidence"),
            {
                "disable_workspace": False,
                "disable_core": False,
                "disable_coda": False,
                "disable_qtrm_residual": True,
                "disable_donor_context": False,
                "workspace_only_context": False,
            },
        )
        self.assertEqual(
            self.module.mode_forward_kwargs("qtrm_donor_hidden_off_with_evidence"),
            {
                "disable_workspace": False,
                "disable_core": False,
                "disable_coda": False,
                "disable_qtrm_residual": False,
                "disable_donor_context": True,
                "workspace_only_context": False,
            },
        )
        self.assertEqual(
            self.module.mode_forward_kwargs("qtrm_workspace_only_with_evidence"),
            {
                "disable_workspace": False,
                "disable_core": False,
                "disable_coda": False,
                "disable_qtrm_residual": False,
                "disable_donor_context": False,
                "workspace_only_context": True,
            },
        )

    def test_mode_forward_kwargs_can_force_core_halt_mode(self):
        self.assertEqual(
            self.module.mode_forward_kwargs(
                "qtrm_residual_with_evidence",
                core_halt_mode="enabled",
            ),
            {
                "disable_workspace": False,
                "disable_core": False,
                "enable_core_halt": True,
            },
        )
        self.assertEqual(
            self.module.mode_forward_kwargs(
                "qtrm_residual_with_evidence",
                core_halt_mode="disabled",
            ),
            {
                "disable_workspace": False,
                "disable_core": False,
                "enable_core_halt": False,
            },
        )

    def test_cli_exposes_core_halt_mode(self):
        parser = self.module.build_arg_parser()

        args = parser.parse_args(["--core-halt-mode", "disabled"])

        self.assertEqual(args.core_halt_mode, "disabled")

    def test_core_halt_telemetry_serializes_prompt_forward_outputs(self):
        import torch

        record = self.module.core_halt_telemetry(
            {
                "core_q_halt_logits": torch.tensor([[0.1, 0.7]]),
                "core_q_continue_logits": torch.tensor([[0.9, 0.3]]),
                "core_steps": torch.tensor([2]),
                "core_halted": torch.tensor([False]),
            },
            core_halt_mode="disabled",
        )

        self.assertEqual(record["mode"], "disabled")
        self.assertEqual(record["core_steps"], [2])
        self.assertEqual(record["core_halted"], [False])
        self.assertEqual(record["q_halt_steps"], 2)
        self.assertAlmostEqual(record["q_halt_last_mean"], 0.7, places=6)

    def test_default_modes_include_causal_component_ablations(self):
        self.assertIn("qtrm_workspace_off_with_evidence", self.module.DEFAULT_MODES)
        self.assertIn("qtrm_core_off_with_evidence", self.module.DEFAULT_MODES)
        self.assertIn("qtrm_coda_off_with_evidence", self.module.DEFAULT_MODES)
        self.assertIn("qtrm_residual_head_off_with_evidence", self.module.DEFAULT_MODES)
        self.assertIn("qtrm_donor_hidden_off_with_evidence", self.module.DEFAULT_MODES)
        self.assertIn("qtrm_workspace_only_with_evidence", self.module.DEFAULT_MODES)


if __name__ == "__main__":
    unittest.main()
