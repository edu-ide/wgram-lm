import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch


def load_module():
    path = Path("scripts/349_average_qtrm_native_checkpoints.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_checkpoint_average", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QTRMNativeCheckpointAverageTests(unittest.TestCase):
    def test_average_model_states_interpolates_floating_tensors(self):
        module = load_module()
        base = {
            "w": torch.tensor([1.0, 3.0]),
            "ids": torch.tensor([1, 2], dtype=torch.long),
        }
        candidate = {
            "w": torch.tensor([5.0, 7.0]),
            "ids": torch.tensor([1, 2], dtype=torch.long),
        }

        averaged = module.average_model_states(base, candidate, alpha=0.25)

        self.assertTrue(torch.allclose(averaged["w"], torch.tensor([2.0, 4.0])))
        self.assertTrue(torch.equal(averaged["ids"], torch.tensor([1, 2])))

    def test_cli_build_preserves_chars_and_records_sources(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_path = tmp_path / "base.pt"
            candidate_path = tmp_path / "candidate.pt"
            torch.save(
                {
                    "model_state": {"w": torch.tensor([0.0])},
                    "args": {"stage": "base"},
                    "chars": ("a", "b"),
                },
                base_path,
            )
            torch.save(
                {
                    "model_state": {"w": torch.tensor([10.0])},
                    "args": {"stage": "candidate"},
                    "chars": ("a", "b"),
                },
                candidate_path,
            )
            args = module.build_arg_parser().parse_args(
                [
                    "--base-checkpoint",
                    str(base_path),
                    "--candidate-checkpoint",
                    str(candidate_path),
                    "--alpha",
                    "0.5",
                    "--out",
                    str(tmp_path / "avg.pt"),
                ]
            )

            averaged = module.build_averaged_checkpoint(args)

        self.assertTrue(torch.allclose(averaged["model_state"]["w"], torch.tensor([5.0])))
        self.assertEqual(averaged["chars"], ("a", "b"))
        self.assertEqual(averaged["checkpoint_average"]["alpha"], 0.5)
        self.assertEqual(averaged["args"]["stage"], "base")

    def test_cli_build_supports_integrated_model_key(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_path = tmp_path / "base.pt"
            candidate_path = tmp_path / "candidate.pt"
            torch.save({"model": {"w": torch.tensor([1.0])}, "report": {"seed": 1}}, base_path)
            torch.save({"model": {"w": torch.tensor([3.0])}, "report": {"seed": 2}}, candidate_path)
            args = module.build_arg_parser().parse_args(
                [
                    "--base-checkpoint",
                    str(base_path),
                    "--candidate-checkpoint",
                    str(candidate_path),
                    "--alpha",
                    "0.25",
                    "--out",
                    str(tmp_path / "avg.pt"),
                ]
            )

            averaged = module.build_averaged_checkpoint(args)

        self.assertTrue(torch.allclose(averaged["model"]["w"], torch.tensor([1.5])))
        self.assertNotIn("_average_state_key", averaged)


if __name__ == "__main__":
    unittest.main()
