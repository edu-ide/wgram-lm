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

from qtrm_mm.agentic.cognitive_loop import Action
from qtrm_mm.infer import build_prompt_with_memory


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
        self.bos_id = getattr(tokenizer, "bos_token_id", None)
        self.eos_id = getattr(tokenizer, "eos_token_id", None)
        if self.pad_id is None:
            self.pad_id = self.eos_id
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

    def token_span_for_char_span(
        self,
        text: str,
        start_char: int,
        end_char: int,
        max_len: int,
    ) -> Optional[tuple[int, int]]:
        encoded = self.tokenizer(
            text or "",
            truncation=True,
            max_length=max_len,
            padding=False,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping") or []
        token_indices = [
            idx
            for idx, pair in enumerate(offsets[:max_len])
            if len(pair) == 2
            and int(pair[1]) > int(start_char)
            and int(pair[0]) < int(end_char)
            and int(pair[1]) > int(pair[0])
        ]
        if not token_indices:
            return None
        return token_indices[0], token_indices[-1]


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


def split_memory_prompt_for_workspace(prompt: str) -> tuple[str, str]:
    """Split MemoryOS prompt evidence from the visible user prompt.

    Returns `(visible_prompt, evidence_text)`. If the prompt is not in the
    `build_prompt_with_memory` format, `evidence_text` is empty and the original
    prompt is returned unchanged.
    """
    text = str(prompt or "")
    marker = "\n\nUser prompt:\n"
    if not text.startswith("MemoryOS evidence") or marker not in text:
        return text, ""
    before_user_prompt, visible_prompt = text.split(marker, 1)
    evidence_text = before_user_prompt.split("\n\nUse the evidence above", 1)[0].strip()
    if not evidence_text:
        return text, ""
    return visible_prompt.strip(), evidence_text


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
        workspace_evidence_injection: bool = False,
        workspace_evidence_injection_mode: str = "workspace",
    ):
        if workspace_evidence_injection_mode not in {"workspace", "dual", "ssot"}:
            raise ValueError(
                "workspace_evidence_injection_mode must be 'workspace', 'dual', or 'ssot'"
            )
        self.files = [str(f) for f in files]
        self.tok = build_text_tokenizer(vocab_size, tokenizer_model_id, tokenizer)
        self.seq_len = seq_len
        self.visual_dim = visual_dim
        self.max_visual_tokens = max_visual_tokens
        self.multimodal = multimodal
        self.shuffle_buffer = shuffle_buffer
        self.workspace_evidence_injection = workspace_evidence_injection
        self.workspace_evidence_injection_mode = workspace_evidence_injection_mode

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
        if row.get("type") == "trace_replay":
            return self._make_trace_replay_sample(row)
        if row.get("type") == "evidence_span_reader":
            return self._make_evidence_span_reader_sample(row)

        text = row.get("text") or ""
        prompt = row.get("prompt") or ""
        chosen = row.get("chosen") or ""
        rejected = row.get("rejected") or ""
        preference = bool(prompt and chosen and rejected and not text)
        answer = row.get("answer") or (chosen if preference else "")
        supervised = bool(prompt and answer and not text)
        workspace_text = ""
        if supervised:
            if self.workspace_evidence_injection:
                visible_prompt, workspace_text = split_memory_prompt_for_workspace(prompt)
                if self.workspace_evidence_injection_mode == "workspace":
                    prompt = visible_prompt
            prompt_prefix = f"{prompt}\n\n"
            text = f"{prompt_prefix}{answer}"
        elif not text and "controller_signal" in row and prompt and not answer:
            text = prompt
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
            if preference:
                rejected_text = f"{prompt_prefix}{rejected}"
                rejected_ids = self.tok.encode(rejected_text, self.seq_len)
                rejected_mask = (rejected_ids != int(pad_id)).long()
                rejected_labels = rejected_ids.clone()
                rejected_labels[:prompt_len] = -100
                rejected_labels[rejected_mask.to(torch.bool).logical_not()] = -100
                sample["preference_rejected_input_ids"] = rejected_ids
                sample["preference_rejected_attention_mask"] = rejected_mask
                sample["preference_rejected_labels"] = rejected_labels
                preference_weight = row.get("preference_weight", row.get("confidence", 1.0))
                sample["preference_sample_weight"] = torch.tensor(
                    float(preference_weight),
                    dtype=torch.float32,
                )
        if self.workspace_evidence_injection:
            if self.workspace_evidence_injection_mode in {"workspace", "dual"} and workspace_text:
                workspace_ids = self.tok.encode(workspace_text, self.seq_len)
                workspace_mask = (workspace_ids != int(pad_id)).long()
            else:
                workspace_ids = torch.full_like(input_ids, int(pad_id))
                workspace_mask = torch.zeros_like(input_ids)
            sample["workspace_input_ids"] = workspace_ids
            sample["workspace_attention_mask"] = workspace_mask
            counterfactual_workspace_text = (
                row.get("counterfactual_workspace_text")
                or row.get("workspace_counterfactual_text")
                or row.get("counterfactual_evidence")
                or ""
            )
            if counterfactual_workspace_text:
                counterfactual_ids = self.tok.encode(counterfactual_workspace_text, self.seq_len)
                counterfactual_mask = (counterfactual_ids != int(pad_id)).long()
                sample["workspace_counterfactual_input_ids"] = counterfactual_ids
                sample["workspace_counterfactual_attention_mask"] = counterfactual_mask
            if workspace_text and "logical_support_target" not in sample:
                sample["logical_support_target"] = torch.tensor(
                    float(row.get("logical_support_target", 1.0)),
                    dtype=torch.float32,
                )
                sample["causal_evidence_target"] = torch.tensor(
                    float(row.get("causal_evidence_target", 1.0)),
                    dtype=torch.float32,
                )
            for key in ("logical_refute_target", "logical_missing_target"):
                if key in row:
                    sample[key] = torch.tensor(float(row[key]), dtype=torch.float32)
        if self.multimodal:
            paths = row.get("images") or []
            sample["visual_features"] = image_to_features(paths, self.visual_dim, self.max_visual_tokens)
        for key in (
            "generation_verifier_repeat_target",
            "generation_verifier_stop_target",
            "generation_verifier_quality_target",
            "generation_verifier_sample_weight",
            "answer_decision_target",
            "answer_decision_sample_weight",
        ):
            if key in row:
                sample[key] = torch.tensor(float(row[key]), dtype=torch.float32)
        if "answer_decision_features" in row:
            sample["answer_decision_features"] = torch.tensor(
                [float(value) for value in row["answer_decision_features"]],
                dtype=torch.float32,
            )
        if "controller_signal" in row:
            values = [float(value) for value in list(row["controller_signal"])]
            sample["controller_signal"] = torch.tensor(values, dtype=torch.float32)
        elif (
            "controller_world_model_signal" in row
            or "controller_verifier_signal" in row
        ):
            sample["controller_signal"] = torch.tensor(
                [
                    float(row.get("controller_world_model_signal", 0.0)),
                    float(row.get("controller_verifier_signal", 0.0)),
                ],
                dtype=torch.float32,
            )
        return sample

    def _make_trace_replay_sample(self, row: Dict):
        text = _render_trace_replay_action_input(row)
        input_ids = self.tok.encode(text, self.seq_len)
        pad_id = getattr(self.tok, "pad_id", 0)
        sample = {
            "input_ids": input_ids,
            "attention_mask": (input_ids != int(pad_id)).long(),
            "labels": torch.full_like(input_ids, -100),
        }

        workspace_text = (
            row.get("workspace_context")
            or row.get("workspace_text")
            or row.get("workspace_evidence")
            or ""
        )
        if workspace_text:
            workspace_ids = self.tok.encode(workspace_text, self.seq_len)
            workspace_mask = (workspace_ids != int(pad_id)).long()
        else:
            workspace_ids = torch.full_like(input_ids, int(pad_id))
            workspace_mask = torch.zeros_like(input_ids)
        sample["workspace_input_ids"] = workspace_ids
        sample["workspace_attention_mask"] = workspace_mask

        if "action_target" in row:
            sample["action_target"] = torch.tensor(
                _trace_action_id(row["action_target"]),
                dtype=torch.long,
            )
            sample["action_sample_weight"] = torch.tensor(
                float(row.get("action_sample_weight", row.get("reward", 1.0))),
                dtype=torch.float32,
            )
        if "controller_signal" in row:
            values = [float(value) for value in list(row["controller_signal"])]
            sample["controller_signal"] = torch.tensor(values, dtype=torch.float32)
        elif (
            "controller_world_model_signal" in row
            or "controller_verifier_signal" in row
        ):
            sample["controller_signal"] = torch.tensor(
                [
                    float(row.get("controller_world_model_signal", 0.0)),
                    float(row.get("controller_verifier_signal", 0.0)),
                ],
                dtype=torch.float32,
            )
        if self.multimodal:
            paths = row.get("images") or []
            sample["visual_features"] = image_to_features(
                paths,
                self.visual_dim,
                self.max_visual_tokens,
            )
        return sample

    def _make_evidence_span_reader_sample(self, row: Dict):
        prompt = row.get("visible_prompt") or row.get("prompt") or ""
        answer = row.get("answer") or f"Answer: {row.get('answer_text', '')}".strip()
        prompt_prefix = f"{prompt}\n\n"
        text = f"{prompt_prefix}{answer}"
        input_ids = self.tok.encode(text, self.seq_len)
        pad_id = getattr(self.tok, "pad_id", 0)
        sample = {
            "input_ids": input_ids,
            "attention_mask": (input_ids != int(pad_id)).long(),
        }
        prompt_ids = self.tok.encode(prompt_prefix, self.seq_len)
        prompt_len = _matching_prefix_len(input_ids, prompt_ids, int(pad_id))
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[sample["attention_mask"].to(torch.bool).logical_not()] = -100
        sample["labels"] = labels

        workspace_text = row.get("workspace_evidence") or row.get("workspace_text") or ""
        if (
            self.workspace_evidence_injection
            and self.workspace_evidence_injection_mode in {"dual", "ssot"}
            and workspace_text
        ):
            prompt = build_prompt_with_memory(prompt, workspace_text)
            prompt_prefix = f"{prompt}\n\n"
            text = f"{prompt_prefix}{answer}"
            input_ids = self.tok.encode(text, self.seq_len)
            sample["input_ids"] = input_ids
            sample["attention_mask"] = (input_ids != int(pad_id)).long()
            prompt_ids = self.tok.encode(prompt_prefix, self.seq_len)
            prompt_len = _matching_prefix_len(input_ids, prompt_ids, int(pad_id))
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            labels[sample["attention_mask"].to(torch.bool).logical_not()] = -100
            sample["labels"] = labels
        if (
            self.workspace_evidence_injection
            and self.workspace_evidence_injection_mode in {"workspace", "dual"}
            and workspace_text
        ):
            workspace_ids = self.tok.encode(workspace_text, self.seq_len)
            workspace_mask = (workspace_ids != int(pad_id)).long()
        else:
            workspace_ids = torch.full_like(input_ids, int(pad_id))
            workspace_mask = torch.zeros_like(input_ids)
        sample["workspace_input_ids"] = workspace_ids
        sample["workspace_attention_mask"] = workspace_mask

        no_answer = bool(row.get("no_answer"))
        span_text = ""
        span = None
        if isinstance(row.get("answer_span"), dict):
            answer_span = row["answer_span"]
            span_text = str(answer_span.get("text") or answer_span.get("alias") or "")
            start_char = answer_span.get("start_char")
            end_char = answer_span.get("end_char")
            if (
                self.workspace_evidence_injection_mode != "ssot"
                and start_char is not None
                and end_char is not None
                and hasattr(
                    self.tok,
                    "token_span_for_char_span",
                )
            ):
                span = self.tok.token_span_for_char_span(
                    workspace_text,
                    int(start_char),
                    int(end_char),
                    self.seq_len,
                )
        if not span_text:
            span_text = str(row.get("answer_text") or "")
        reader_ids = input_ids if self.workspace_evidence_injection_mode == "ssot" else workspace_ids
        if span is None and not no_answer:
            span = _find_token_subsequence(
                reader_ids,
                self.tok.encode(span_text, self.seq_len),
                pad_id=int(pad_id),
                bos_id=getattr(self.tok, "bos_id", None),
                eos_id=getattr(self.tok, "eos_id", None),
            )
        sample["evidence_span_no_answer_target"] = torch.tensor(float(no_answer), dtype=torch.float32)
        sample["evidence_span_start_target"] = torch.tensor(-100, dtype=torch.long)
        sample["evidence_span_end_target"] = torch.tensor(-100, dtype=torch.long)
        span_weight = 0.0
        if span is not None:
            start, end = span
            if 0 <= start <= end < int(reader_ids.numel()):
                sample["evidence_span_start_target"] = torch.tensor(start, dtype=torch.long)
                sample["evidence_span_end_target"] = torch.tensor(end, dtype=torch.long)
                span_weight = 1.0
        elif no_answer:
            span_weight = 1.0
        sample["evidence_span_sample_weight"] = torch.tensor(span_weight, dtype=torch.float32)

        if self.workspace_evidence_injection:
            sample["logical_support_target"] = torch.tensor(0.0 if no_answer else 1.0, dtype=torch.float32)
            sample["logical_missing_target"] = torch.tensor(1.0 if no_answer else 0.0, dtype=torch.float32)
            sample["causal_evidence_target"] = torch.tensor(0.0 if no_answer else 1.0, dtype=torch.float32)
        if self.multimodal:
            sample["visual_features"] = image_to_features(
                row.get("images") or [],
                self.visual_dim,
                self.max_visual_tokens,
            )
        return sample


def _nonpad_values(ids: torch.Tensor, pad_id: int) -> List[int]:
    return [int(x) for x in ids.tolist() if int(x) != pad_id]


def _matching_prefix_len(full_ids: torch.Tensor, prefix_ids: torch.Tensor, pad_id: int) -> int:
    full = _nonpad_values(full_ids, pad_id)
    prefix = _nonpad_values(prefix_ids, pad_id)
    while prefix and full[: len(prefix)] != prefix:
        prefix = prefix[:-1]
    return min(len(prefix), full_ids.numel())


def _token_content_values(
    ids: torch.Tensor,
    *,
    pad_id: int,
    bos_id: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> List[int]:
    skip = {int(pad_id)}
    if bos_id is not None:
        skip.add(int(bos_id))
    if eos_id is not None:
        skip.add(int(eos_id))
    return [int(x) for x in ids.tolist() if int(x) not in skip]


def _find_token_subsequence(
    haystack_ids: torch.Tensor,
    needle_ids: torch.Tensor,
    *,
    pad_id: int,
    bos_id: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> Optional[tuple[int, int]]:
    needle = _token_content_values(needle_ids, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id)
    if not needle:
        return None
    skip = {int(pad_id)}
    if bos_id is not None:
        skip.add(int(bos_id))
    if eos_id is not None:
        skip.add(int(eos_id))
    haystack = [int(x) for x in haystack_ids.tolist()]
    limit = len(haystack) - len(needle) + 1
    for start in range(max(0, limit)):
        window = haystack[start : start + len(needle)]
        if any(token in skip for token in window):
            continue
        if window == needle:
            return start, start + len(needle) - 1
    return None


def collate_jsonl(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    if _any_has(batch, "attention_mask"):
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
    else:
        attention_mask = (input_ids != 0).long()
    out = {"input_ids": input_ids, "attention_mask": attention_mask}
    if _any_has(batch, "labels"):
        out["labels"] = torch.stack([
            b.get("labels", torch.full_like(b["input_ids"], -100))
            for b in batch
        ])
    if _any_has(batch, "workspace_input_ids"):
        out["workspace_input_ids"] = torch.stack([
            b.get("workspace_input_ids", torch.zeros_like(b["input_ids"]))
            for b in batch
        ])
        out["workspace_attention_mask"] = torch.stack([
            b.get("workspace_attention_mask", torch.zeros_like(b["input_ids"]))
            for b in batch
        ])
    if _any_has(batch, "workspace_counterfactual_input_ids"):
        out["workspace_counterfactual_input_ids"] = torch.stack(
            [
                b.get("workspace_counterfactual_input_ids", torch.zeros_like(b["input_ids"]))
                for b in batch
            ]
        )
        out["workspace_counterfactual_attention_mask"] = torch.stack(
            [
                b.get("workspace_counterfactual_attention_mask", torch.zeros_like(b["input_ids"]))
                for b in batch
            ]
        )
    if _any_has(batch, "controller_signal"):
        template = next(b["controller_signal"] for b in batch if "controller_signal" in b)
        out["controller_signal"] = torch.stack(
            [
                b.get("controller_signal", torch.zeros_like(template))
                for b in batch
            ]
        )
    if _any_has(batch, "answer_decision_features"):
        template = next(b["answer_decision_features"] for b in batch if "answer_decision_features" in b)
        out["answer_decision_features"] = torch.stack(
            [
                b.get("answer_decision_features", torch.zeros_like(template))
                for b in batch
            ]
        )
    for key in (
        "logical_support_target",
        "logical_refute_target",
        "logical_missing_target",
        "causal_evidence_target",
        "generation_verifier_repeat_target",
        "generation_verifier_stop_target",
        "generation_verifier_quality_target",
        "generation_verifier_sample_weight",
        "answer_decision_target",
        "answer_decision_sample_weight",
        "evidence_span_no_answer_target",
        "evidence_span_sample_weight",
        "action_sample_weight",
    ):
        if _any_has(batch, key):
            default = (
                1.0
                if key
                in {
                    "generation_verifier_sample_weight",
                    "answer_decision_sample_weight",
                }
                else 0.0
            )
            out[key] = torch.stack([
                b.get(key, torch.tensor(default, dtype=torch.float32))
                for b in batch
            ])
    for key in (
        "evidence_span_start_target",
        "evidence_span_end_target",
        "action_target",
    ):
        if _any_has(batch, key):
            out[key] = torch.stack([
                b.get(key, torch.tensor(-100, dtype=torch.long))
                for b in batch
            ])
    if _any_has(batch, "preference_rejected_input_ids"):
        out["preference_rejected_input_ids"] = torch.stack(
            [
                b.get("preference_rejected_input_ids", b["input_ids"])
                for b in batch
            ]
        )
        out["preference_rejected_attention_mask"] = torch.stack(
            [
                b.get(
                    "preference_rejected_attention_mask",
                    b.get("attention_mask", (b["input_ids"] != 0).long()),
                )
                for b in batch
            ]
        )
        out["preference_rejected_labels"] = torch.stack(
            [
                b.get("preference_rejected_labels", torch.full_like(b["input_ids"], -100))
                for b in batch
            ]
        )
        out["preference_sample_weight"] = torch.stack(
            [
                b.get("preference_sample_weight", torch.tensor(0.0, dtype=torch.float32))
                for b in batch
            ]
        )
    if _any_has(batch, "visual_features"):
        template = next(b["visual_features"] for b in batch if "visual_features" in b)
        out["visual_features"] = torch.stack([
            b.get("visual_features", torch.zeros_like(template))
            for b in batch
        ])
    return out


def _trace_action_id(value: Any) -> int:
    if isinstance(value, Action):
        return value.id
    if isinstance(value, int):
        return int(value)
    text = str(value)
    try:
        return Action[text].id
    except KeyError:
        return Action(text).id


def _render_trace_replay_action_input(row: Dict) -> str:
    chat_prompt = str(row.get("chat_prompt") or row.get("prompt") or "")
    state_summary = str(row.get("state_summary") or chat_prompt)
    trace_step = int(row.get("step", 0))
    hide_step = bool(row.get("hide_trace_step_from_input"))
    parts = [
        "Controller trace replay.",
        "state_summary:",
        state_summary,
    ]
    if not hide_step:
        parts.insert(1, f"trace_step={trace_step}")
    previous_observation = row.get("previous_observation")
    if previous_observation and not bool(row.get("hide_previous_observation_from_input")):
        parts.extend(["previous_observation:", str(previous_observation)])
    if chat_prompt and chat_prompt != state_summary:
        parts.extend(["chat_prompt:", chat_prompt])
    parts.extend(
        [
            "next_action_query:",
            "state_summary:",
            state_summary,
        ]
    )
    if not hide_step:
        parts.insert(-2, f"trace_step={trace_step}")
    return "\n".join(parts)


def _any_has(batch, key: str) -> bool:
    return any(key in b for b in batch)
