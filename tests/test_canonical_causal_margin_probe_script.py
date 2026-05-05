import importlib.util
import unittest
from pathlib import Path


def load_probe_module():
    path = Path("scripts/169_probe_canonical_causal_margin.py")
    spec = importlib.util.spec_from_file_location("canonical_causal_margin_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CanonicalCausalMarginProbeScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_probe_module()

    def test_default_ablation_modes_probe_latent_causal_paths(self):
        parser = self.module.build_arg_parser()

        args = parser.parse_args([])

        self.assertIn("core_off", args.ablation_mode)
        self.assertIn("workspace_off", args.ablation_mode)
        self.assertIn("core_context_off", args.ablation_mode)
        self.assertIn("core_to_text_off", args.ablation_mode)

    def test_mode_forward_kwargs_match_training_ablation_names(self):
        self.assertEqual(self.module.ablation_forward_kwargs("core_off"), {"disable_core": True})
        self.assertEqual(
            self.module.ablation_forward_kwargs("workspace_off"),
            {"disable_workspace": True},
        )
        self.assertEqual(
            self.module.ablation_forward_kwargs("core_context_off"),
            {"disable_core_context": True},
        )
        self.assertEqual(
            self.module.ablation_forward_kwargs("core_to_text_off"),
            {"disable_core_to_text": True},
        )

    def test_summarize_records_reports_mean_margins_by_mode(self):
        summary = self.module.summarize_probe_records(
            [
                {
                    "full_logp": -1.0,
                    "ablation_logps": {
                        "core_off": -1.5,
                        "workspace_off": -0.75,
                    },
                },
                {
                    "full_logp": -2.0,
                    "ablation_logps": {
                        "core_off": -2.25,
                        "workspace_off": -2.5,
                    },
                },
            ],
            causal_margin_threshold=0.2,
        )

        self.assertEqual(summary["num_records"], 2)
        self.assertAlmostEqual(summary["full_logp_mean"], -1.5, places=6)
        self.assertAlmostEqual(summary["mode_margins"]["core_off"]["mean"], 0.375, places=6)
        self.assertAlmostEqual(
            summary["mode_margins"]["workspace_off"]["mean"],
            0.125,
            places=6,
        )
        self.assertTrue(summary["mode_margins"]["core_off"]["causal"])
        self.assertFalse(summary["mode_margins"]["workspace_off"]["causal"])

    def test_batch_logprobs_accepts_amp_flag_for_donor_bfloat16_probe(self):
        import inspect

        source = inspect.getsource(self.module.batch_logprobs)

        self.assertIn("use_amp", source)
        self.assertIn("torch.amp.autocast", source)


if __name__ == "__main__":
    unittest.main()
