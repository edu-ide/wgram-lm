import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
HRM_TEXT_ROOT = ROOT / "references" / "official" / "hrm-text"
if str(HRM_TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(HRM_TEXT_ROOT))


def _naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    probs = scores.softmax(dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v)


class HRMTextAttentionFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["HRMTEXT_ATTENTION_BACKEND"] = "sdpa"

    def test_sdpa_prefixlm_matches_naive_masking_and_backpropagates(self):
        from models.flash_attention_prefixlm_v2 import compute_aux_seq_tensors_scalars, flash_attn_varlen_prefixlm

        prefix_lens = np.array([3, 2], dtype=np.int32)
        causal_lens = np.array([2, 1], dtype=np.int32)
        tensors, scalars = compute_aux_seq_tensors_scalars(prefix_lens, causal_lens, batch_max_tokens=10)

        torch.manual_seed(7)
        q = torch.randn(10, 2, 8, dtype=torch.float32, requires_grad=True)
        k = torch.randn(10, 2, 8, dtype=torch.float32, requires_grad=True)
        v = torch.randn(10, 2, 8, dtype=torch.float32, requires_grad=True)

        kwargs = {name: torch.from_numpy(value) for name, value in tensors.items()}
        kwargs |= {name: torch.tensor(value) for name, value in scalars.items()}
        out = flash_attn_varlen_prefixlm(q, k, v, False, **kwargs)

        expected = torch.zeros_like(out)
        cu = tensors["cu_seqlens"]
        for idx, (prefix_len, causal_len) in enumerate(zip(prefix_lens.tolist(), causal_lens.tolist())):
            start = int(cu[idx])
            total_len = prefix_len + causal_len
            prefix_slice = slice(start, start + prefix_len)
            expected[prefix_slice] = _naive_attention(
                q[prefix_slice],
                k[prefix_slice],
                v[prefix_slice],
                torch.ones(prefix_len, prefix_len, dtype=torch.bool),
            )

            query_slice = slice(start + prefix_len, start + total_len)
            rows = torch.arange(causal_len).unsqueeze(-1)
            cols = torch.arange(total_len).unsqueeze(0)
            mask = cols <= (prefix_len + rows)
            expected[query_slice] = _naive_attention(q[query_slice], k[start:start + total_len], v[start:start + total_len], mask)

        self.assertTrue(torch.allclose(out, expected, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.equal(out[scalars["total_seqlen"]:], torch.zeros_like(out[scalars["total_seqlen"]:])))

        out.square().mean().backward()
        self.assertIsNotNone(q.grad)
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_layers_import_without_flash_attn_interface(self):
        import models.layers as layers

        self.assertTrue(hasattr(layers, "Attention"))


if __name__ == "__main__":
    unittest.main()
