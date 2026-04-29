from __future__ import annotations
import torch
from torch.utils.data import IterableDataset


class SyntheticTextVisionDataset(IterableDataset):
    def __init__(self, vocab_size: int, seq_len: int, visual_dim: int, max_visual_tokens: int, multimodal: bool = True):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.visual_dim = visual_dim
        self.max_visual_tokens = max_visual_tokens
        self.multimodal = multimodal

    def __iter__(self):
        while True:
            # structured random task: repeated arithmetic-ish token pattern.
            x = torch.randint(32, min(self.vocab_size, 256), (self.seq_len,), dtype=torch.long)
            if self.seq_len > 16:
                x[-8:] = torch.flip(x[:8], dims=[0])
            sample = {"input_ids": x}
            if self.multimodal:
                v = torch.randn(self.max_visual_tokens, self.visual_dim)
                sample["visual_features"] = v
            yield sample


def collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    out = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    if "visual_features" in batch[0]:
        out["visual_features"] = torch.stack([b["visual_features"] for b in batch])
    return out
