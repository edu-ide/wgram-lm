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
        include_evidence, qtrm_scale, donor_scale = self.module.mode_settings(
            "qtrm_workspace_off_with_evidence",
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

    def test_default_modes_include_causal_component_ablations(self):
        self.assertIn("qtrm_workspace_off_with_evidence", self.module.DEFAULT_MODES)
        self.assertIn("qtrm_core_off_with_evidence", self.module.DEFAULT_MODES)


if __name__ == "__main__":
    unittest.main()
