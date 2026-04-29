from __future__ import annotations

import json
import hashlib
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset


class HashTokenizer:
    """Tiny deterministic tokenizer for architecture/debug training.

    Production route should swap this for the Qwen3.5 donor tokenizer or a trained tokenizer.
    This tokenizer exists so downloaded datasets can train the scaffold without an external tokenizer.
    """
    def __init__(self, vocab_size: int, bos_id: int = 1, eos_id: int = 2, pad_id: int = 0):
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        pieces = re.findall(r"<image>|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text or "")
        ids = [self.bos_id]
        for p in pieces[: max_len - 2]:
            if p == "<image>":
                ids.append(3)
            else:
                digest = hashlib.blake2b(p.encode("utf-8"), digest_size=8).digest()
                h = int.from_bytes(digest, "big") % max(1, self.vocab_size - 32)
                ids.append(32 + h)
        ids.append(self.eos_id)
        if len(ids) < max_len:
            ids.extend([self.pad_id] * (max_len - len(ids)))
        return torch.tensor(ids[:max_len], dtype=torch.long)


class HFTokenizerAdapter:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.pad_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_id is None:
            self.pad_id = getattr(tokenizer, "eos_token_id", None)
        if self.pad_id is None:
            self.pad_id = 0

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        encoded = self.tokenizer(
            text or "",
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=False,
            add_special_tokens=True,
        )
        ids = encoded["input_ids"][0].to(dtype=torch.long)[:max_len]
        if ids.numel() < max_len:
            pad = torch.full((max_len - ids.numel(),), int(self.pad_id), dtype=torch.long)
            ids = torch.cat([ids, pad], dim=0)
        return ids


def build_text_tokenizer(
    vocab_size: int,
    tokenizer_model_id: Optional[str] = None,
    tokenizer: Optional[Any] = None,
):
    if tokenizer is not None:
        return HFTokenizerAdapter(tokenizer)
    if tokenizer_model_id:
        from transformers import AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, trust_remote_code=True)
        return HFTokenizerAdapter(hf_tokenizer)
    return HashTokenizer(vocab_size)


def image_to_features(paths: List[str], visual_dim: int, max_visual_tokens: int) -> torch.Tensor:
    """Deterministic low-res image featurizer for scaffold training.

    Production route should replace this with Qwen3.5 vision encoder features or SigLIP/Qwen visual embeddings.
    """
    feats = []
    for path in paths[: max(1, max_visual_tokens // 16)]:
        try:
            img = Image.open(path).convert("RGB").resize((4, 4))
            arr = np.asarray(img).astype("float32") / 255.0
            for y in range(4):
                for x in range(4):
                    rgb = arr[y, x]
                    base = np.array([rgb[0], rgb[1], rgb[2], x / 3.0, y / 3.0, 1.0], dtype="float32")
                    rep = np.resize(base, visual_dim)
                    feats.append(rep)
        except Exception:
            continue
    if not feats:
        feats = [np.zeros((visual_dim,), dtype="float32")]
    feats = np.stack(feats, axis=0)
    if feats.shape[0] < max_visual_tokens:
        pad = np.zeros((max_visual_tokens - feats.shape[0], visual_dim), dtype="float32")
        feats = np.concatenate([feats, pad], axis=0)
    return torch.tensor(feats[:max_visual_tokens], dtype=torch.float32)


class JsonlTextVisionDataset(IterableDataset):
    def __init__(
        self,
        files: List[str],
        vocab_size: int,
        seq_len: int,
        visual_dim: int,
        max_visual_tokens: int,
        multimodal: bool = True,
        shuffle_buffer: int = 2048,
        tokenizer_model_id: Optional[str] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.files = [str(f) for f in files]
        self.tok = build_text_tokenizer(vocab_size, tokenizer_model_id, tokenizer)
        self.seq_len = seq_len
        self.visual_dim = visual_dim
        self.max_visual_tokens = max_visual_tokens
        self.multimodal = multimodal
        self.shuffle_buffer = shuffle_buffer

    def _iter_rows(self):
        while True:
            for fpath in self.files:
                p = Path(fpath)
                if not p.exists():
                    continue
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        yield row

    def __iter__(self):
        buf = []
        for row in self._iter_rows():
            buf.append(row)
            if len(buf) >= self.shuffle_buffer:
                random.shuffle(buf)
                while buf:
                    yield self._make_sample(buf.pop())

    def _make_sample(self, row: Dict):
        text = row.get("text") or ""
        prompt = row.get("prompt") or ""
        answer = row.get("answer") or ""
        supervised = bool(prompt and answer and not text)
        if supervised:
            prompt_prefix = f"{prompt}\n\n"
            text = f"{prompt_prefix}{answer}"
        elif not text:
            text = f"{prompt}\n\n{answer}"
        input_ids = self.tok.encode(text, self.seq_len)
        pad_id = getattr(self.tok, "pad_id", 0)
        sample = {
            "input_ids": input_ids,
            "attention_mask": (input_ids != int(pad_id)).long(),
        }
        if supervised:
            prompt_ids = self.tok.encode(prompt_prefix, self.seq_len)
            prompt_len = _matching_prefix_len(input_ids, prompt_ids, int(pad_id))
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            labels[sample["attention_mask"].to(torch.bool).logical_not()] = -100
            sample["labels"] = labels
        if self.multimodal:
            paths = row.get("images") or []
            sample["visual_features"] = image_to_features(paths, self.visual_dim, self.max_visual_tokens)
        return sample


def _nonpad_values(ids: torch.Tensor, pad_id: int) -> List[int]:
    return [int(x) for x in ids.tolist() if int(x) != pad_id]


def _matching_prefix_len(full_ids: torch.Tensor, prefix_ids: torch.Tensor, pad_id: int) -> int:
    full = _nonpad_values(full_ids, pad_id)
    prefix = _nonpad_values(prefix_ids, pad_id)
    while prefix and full[: len(prefix)] != prefix:
        prefix = prefix[:-1]
    return min(len(prefix), full_ids.numel())


def collate_jsonl(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    if "attention_mask" in batch[0]:
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
    else:
        attention_mask = (input_ids != 0).long()
    out = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "labels" in batch[0]:
        out["labels"] = torch.stack([b["labels"] for b in batch])
    if "visual_features" in batch[0]:
        out["visual_features"] = torch.stack([b["visual_features"] for b in batch])
    return out
