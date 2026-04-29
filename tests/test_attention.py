import unittest

import torch


class AttentionTests(unittest.TestCase):
    def test_causal_attention_remains_causal_when_padding_mask_is_present(self):
        from qtrm_mm.attention import GroupedQueryAttention

        torch.manual_seed(0)
        attn = GroupedQueryAttention(
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=8,
            causal=True,
            dropout=0.0,
        )
        attn.eval()
        x = torch.randn(2, 6, 16)
        mask = torch.ones(2, 6, dtype=torch.long)

        no_mask = attn(x)
        all_valid_mask = attn(x, attention_mask=mask)

        self.assertTrue(torch.allclose(no_mask, all_valid_mask, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
