import importlib.util
import unittest
from pathlib import Path

import torch


def load_module():
    path = Path("scripts/246_build_donor_unembedding_aligned_checkpoint.py")
    spec = importlib.util.spec_from_file_location("donor_unembedding_aligned_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class DonorUnembeddingAlignedCheckpointScriptTests(unittest.TestCase):
    def test_pinv_mapping_recovers_least_squares_unembedding_when_projector_is_identity_slice(self):
        module = load_module()
        projector = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        donor = torch.tensor(
            [
                [2.0, 3.0, 5.0],
                [-1.0, 7.0, 11.0],
            ]
        )

        mapping = module.donor_to_qtrm_mapping(projector, method="pinv", rcond=1.0e-6)
        projected = module.project_unembedding_chunk(donor, mapping)

        self.assertEqual(projected.shape, (2, 2))
        torch.testing.assert_close(projected, donor[:, :2])

    def test_direct_project_mapping_uses_projector_transpose(self):
        module = load_module()
        projector = torch.tensor(
            [
                [1.0, 2.0, 0.0],
                [0.0, 3.0, 4.0],
            ]
        )
        donor = torch.tensor([[2.0, 3.0, 5.0]])

        mapping = module.donor_to_qtrm_mapping(projector, method="project", rcond=1.0e-6)
        projected = module.project_unembedding_chunk(donor, mapping)

        torch.testing.assert_close(projected, donor @ projector.t())

    def test_match_mean_row_norm_matches_reference_scale(self):
        module = load_module()
        weight = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
        reference = torch.tensor([[0.3, 0.4], [0.3, 0.4]])

        scaled, report = module.match_mean_row_norm(weight, reference)

        self.assertLess(
            abs(float(scaled.norm(dim=1).mean()) - float(reference.norm(dim=1).mean())),
            1e-6,
        )
        self.assertLess(report["scale"], 1.0)

    def test_parser_exposes_untied_probe_defaults(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--checkpoint",
                "in.pt",
                "--output",
                "out.pt",
            ]
        )

        self.assertEqual(args.source, "output")
        self.assertEqual(args.method, "pinv")
        self.assertTrue(args.match_existing_row_norm)
        self.assertFalse(args.set_text_embed)
        self.assertFalse(args.resolve_base_chain)
        self.assertFalse(args.set_renderer_use_lm_head)

    def test_parser_exposes_resolved_l4_renderer_config_flags(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--checkpoint",
                "in.pt",
                "--output",
                "out.pt",
                "--resolve-base-chain",
                "--set-renderer-use-lm-head",
            ]
        )

        self.assertTrue(args.resolve_base_chain)
        self.assertTrue(args.set_renderer_use_lm_head)


if __name__ == "__main__":
    unittest.main()
