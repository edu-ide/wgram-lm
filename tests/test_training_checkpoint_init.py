import tempfile
import unittest

import torch


class TrainingCheckpointInitTests(unittest.TestCase):
    def test_load_initial_checkpoint_restores_model_state(self):
        from qtrm_mm.training.train import load_initial_checkpoint

        model = torch.nn.Linear(2, 2)
        wanted = torch.nn.Linear(2, 2)
        with torch.no_grad():
            wanted.weight.fill_(3.0)
            wanted.bias.fill_(-2.0)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save({"model": wanted.state_dict()}, f.name)
            missing, unexpected = load_initial_checkpoint(model, f.name, map_location="cpu")

        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        self.assertTrue(torch.equal(model.weight, wanted.weight))
        self.assertTrue(torch.equal(model.bias, wanted.bias))


if __name__ == "__main__":
    unittest.main()
