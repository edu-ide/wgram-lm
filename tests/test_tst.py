import unittest

import torch
import torch.nn.functional as F

from qtrm_mm.tst import (
    multi_hot_cross_entropy,
    next_token_bags,
    superpose_embeddings,
)


class TokenSuperpositionTrainingTests(unittest.TestCase):
    def test_next_token_bags_preserves_causal_bag_order(self):
        tokens = torch.tensor([[10, 11, 12, 13, 14, 15]])

        input_bags, target_bags = next_token_bags(tokens, bag_size=2)

        self.assertEqual(input_bags.tolist(), [[[10, 11], [12, 13]]])
        self.assertEqual(target_bags.tolist(), [[[12, 13], [14, 15]]])

    def test_next_token_bags_rejects_short_sequences(self):
        with self.assertRaises(ValueError):
            next_token_bags(torch.tensor([[1, 2, 3]]), bag_size=2)

    def test_superpose_embeddings_averages_each_bag(self):
        embedding = torch.tensor(
            [
                [0.0, 0.0],
                [2.0, 4.0],
                [4.0, 8.0],
                [8.0, 16.0],
            ]
        )
        bags = torch.tensor([[[1, 2], [2, 3]]])

        out = superpose_embeddings(embedding, bags)

        expected = torch.tensor([[[3.0, 6.0], [6.0, 12.0]]])
        self.assertTrue(torch.allclose(out, expected))

    def test_multi_hot_cross_entropy_matches_mean_standard_ce(self):
        logits = torch.tensor(
            [
                [
                    [3.0, 1.0, 0.0, -1.0],
                    [0.0, 2.0, 1.0, -1.0],
                ]
            ]
        )
        target_bags = torch.tensor([[[0, 2], [1, 2]]])

        loss = multi_hot_cross_entropy(logits, target_bags)

        expected = F.cross_entropy(
            logits.unsqueeze(2)
            .expand(1, 2, 2, 4)
            .reshape(-1, 4),
            target_bags.reshape(-1),
        )
        self.assertTrue(torch.allclose(loss, expected))


if __name__ == "__main__":
    unittest.main()
