#!/usr/bin/env python3
"""Tiny donorless QTRM-native text preservation probe.

This is an L3 smoke test: it checks whether the native recurrent LM path can
learn a small text slice without collapsing into repeated characters. It is not
a general language capability claim.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from wgram_lm.mixers import FLADeltaMixer, OfficialMamba3Mixer, TorchGatedDeltaMixer
from wgram_lm.training_optimizers import (
    MEMORY_EFFICIENT_OPTIMIZERS,
    build_memory_efficient_optimizer,
)


DEFAULT_TEXT = (
    "QTRM native language probe. A small recurrent model should preserve "
    "ordinary next-token language behavior while its thinking block is active. "
    "The purpose is not fluency at scale, but a non-degenerate causal LM path. "
) * 64

SUPPORTED_BACKBONES = (
    "mha_etd",
    "qtrm_hybrid_3to1",
    "mamba3",
    "trm_official",
    "trm_mamba3",
    "trm_gated_delta",
    "trm_qwen35_3to1",
    "trm_tri_mixer",
)


def load_native_model_class():
    path = Path(__file__).with_name("335_train_qtrm_native_etd_probe.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_etd_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.NativeQTRMETDLM


def summarize_backend(model) -> dict[str, object]:
    fla_total = 0
    fla_official = 0
    torch_delta_total = 0
    mamba3_total = 0
    mamba3_official = 0
    for module in model.modules():
        if isinstance(module, FLADeltaMixer):
            fla_total += 1
            fla_official += int(bool(module.is_official_backend))
        elif isinstance(module, TorchGatedDeltaMixer):
            torch_delta_total += 1
        elif isinstance(module, OfficialMamba3Mixer):
            mamba3_total += 1
            mamba3_official += int(bool(module.is_official_backend))
    return {
        "fla_delta_mixers": fla_total,
        "official_fla_delta_mixers": fla_official,
        "mamba3_mixers": mamba3_total,
        "official_mamba3_mixers": mamba3_official,
        "torch_delta_mixers": torch_delta_total,
        "all_fla_mixers_official": bool(fla_total > 0 and fla_total == fla_official),
        "all_mamba3_mixers_official": bool(mamba3_total > 0 and mamba3_total == mamba3_official),
    }


@dataclass(frozen=True)
class CharTokenizer:
    chars: tuple[str, ...]
    char_to_id: dict[str, int]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = tuple(sorted(set(text)))
        return cls(chars=chars, char_to_id={ch: index for index, ch in enumerate(chars)})

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @property
    def eos_token_id(self) -> int | None:
        for marker in ("\x03", "\x00"):
            if marker in self.char_to_id:
                return int(self.char_to_id[marker])
        return None

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[ch] for ch in text if ch in self.char_to_id]

    def decode(self, token_ids: list[int]) -> str:
        eos_id = self.eos_token_id
        chars: list[str] = []
        for token_id in token_ids:
            index = int(token_id)
            if eos_id is not None and index == int(eos_id):
                continue
            chars.append(self.chars[index])
        return "".join(chars)


def make_windows(tokens: list[int], *, seq_len: int) -> list[tuple[list[int], list[int]]]:
    if int(seq_len) <= 0:
        raise ValueError("seq_len must be positive")
    return [
        (tokens[index : index + int(seq_len)], tokens[index + 1 : index + int(seq_len) + 1])
        for index in range(0, max(0, len(tokens) - int(seq_len)))
    ]


def batch_windows(
    windows: list[tuple[list[int], list[int]]],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = random.sample(windows, k=min(int(batch_size), len(windows)))
    x = torch.tensor([row[0] for row in batch], dtype=torch.long, device=device)
    y = torch.tensor([row[1] for row in batch], dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def eval_loss(
    model,
    windows: list[tuple[list[int], list[int]]],
    *,
    batch_size: int,
    device: torch.device,
    think_steps: int,
    state_reset_each_step: bool = False,
    thinking_block_off: bool = False,
) -> float:
    model.eval()
    losses: list[float] = []
    for start in range(0, len(windows), int(batch_size)):
        batch = windows[start : start + int(batch_size)]
        x = torch.tensor([row[0] for row in batch], dtype=torch.long, device=device)
        y = torch.tensor([row[1] for row in batch], dtype=torch.long, device=device)
        logits = model(
            x,
            think_steps=int(think_steps),
            state_reset_each_step=bool(state_reset_each_step),
            thinking_block_off=bool(thinking_block_off),
        )
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
        )
        losses.append(float(loss.detach().cpu()))
    return float(sum(losses) / max(1, len(losses)))


def repeat_unlikelihood_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Penalize copying the current token when the target is different.

    This is a local unlikelihood objective for the common greedy LM failure
    mode where punctuation or frequent words repeat indefinitely.
    """

    candidate = x.clamp(min=0, max=int(logits.shape[-1]) - 1)
    mask = y.ne(candidate)
    if not bool(mask.any()):
        return logits.new_zeros(())
    probs = torch.softmax(logits.float(), dim=-1)
    candidate_probs = probs.gather(-1, candidate.unsqueeze(-1)).squeeze(-1)
    penalty = -torch.log(torch.clamp(1.0 - candidate_probs, min=1e-6))
    return penalty.masked_select(mask).mean()


def weighted_next_token_ce_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    eos_token_id: int | None,
    eos_loss_weight: float,
) -> torch.Tensor:
    """Cross entropy with optional extra weight on EOS targets."""

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_y = y.reshape(-1)
    weight = float(eos_loss_weight)
    if eos_token_id is None or weight <= 1.0:
        return F.cross_entropy(flat_logits, flat_y)
    per_token = F.cross_entropy(flat_logits, flat_y, reduction="none")
    token_weights = torch.ones_like(per_token)
    token_weights = torch.where(
        flat_y.eq(int(eos_token_id)),
        token_weights.new_full((), weight),
        token_weights,
    )
    return (per_token * token_weights).sum() / token_weights.sum().clamp_min(1.0)


@torch.no_grad()
def generate_text(
    model,
    tokenizer: CharTokenizer,
    *,
    seed_text: str,
    seq_len: int,
    think_steps: int,
    max_new_chars: int,
    device: torch.device,
) -> str:
    model.eval()
    encoded = tokenizer.encode(seed_text)
    if not encoded:
        encoded = [0]
    out = torch.tensor([encoded], dtype=torch.long, device=device)
    for _ in range(int(max_new_chars)):
        x = out[:, -int(seq_len) :]
        logits = model(x, think_steps=int(think_steps))
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_id], dim=1)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None and int(next_id.item()) == int(eos_token_id):
            break
    return tokenizer.decode(out[0].detach().cpu().tolist())


def sample_degeneracy(sample: str) -> dict[str, float]:
    if not sample:
        return {"unique_chars": 0.0, "max_run_fraction": 1.0}
    max_run = 1
    current = 1
    for left, right in zip(sample, sample[1:]):
        if left == right:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    return {
        "unique_chars": float(len(set(sample))),
        "max_run_fraction": float(max_run / max(1, len(sample))),
    }


def parse_depth_sweep(value: str) -> tuple[int, ...]:
    depths: list[int] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        depth = int(raw)
        if depth < 0:
            raise ValueError("depth sweep values must be non-negative")
        depths.append(depth)
    return tuple(dict.fromkeys(depths))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if float(denominator) <= 0.0:
        return None
    return float(numerator) / float(denominator)


def language_reject_reasons(
    args: argparse.Namespace,
    *,
    full_loss: float,
    think0_loss: float,
    off_loss: float,
    baseline_loss: float | None,
    random_loss: float,
    degeneracy: dict[str, float],
) -> list[str]:
    reject_reasons: list[str] = []
    if full_loss > random_loss * float(args.max_random_loss_fraction):
        reject_reasons.append("loss_too_close_to_random")
    if degeneracy["unique_chars"] < float(args.min_unique_chars):
        reject_reasons.append("sample_too_few_unique_chars")
    if degeneracy["max_run_fraction"] > float(args.max_run_fraction):
        reject_reasons.append("sample_repetition_run_too_high")

    full_vs_think0 = _safe_ratio(full_loss, think0_loss)
    if (
        float(args.max_full_vs_think0_loss_ratio) > 0.0
        and full_vs_think0 is not None
        and full_vs_think0 > float(args.max_full_vs_think0_loss_ratio)
    ):
        reject_reasons.append("full_loss_regressed_vs_think0")

    full_vs_off = _safe_ratio(full_loss, off_loss)
    if (
        float(args.max_full_vs_off_loss_ratio) > 0.0
        and full_vs_off is not None
        and full_vs_off > float(args.max_full_vs_off_loss_ratio)
    ):
        reject_reasons.append("full_loss_regressed_vs_thinking_block_off")

    full_vs_baseline = _safe_ratio(full_loss, baseline_loss)
    if (
        float(args.max_full_vs_baseline_loss_ratio) > 0.0
        and full_vs_baseline is not None
        and full_vs_baseline > float(args.max_full_vs_baseline_loss_ratio)
    ):
        reject_reasons.append("full_loss_regressed_vs_baseline")
    return reject_reasons


def load_text(args: argparse.Namespace) -> str:
    parts: list[str] = []
    seen_paths: set[Path] = set()
    if args.text_file:
        path = Path(args.text_file)
        resolved = path.resolve()
        seen_paths.add(resolved)
        parts.append(f"\n\n## FILE: {path}\n\n{path.read_text(encoding='utf-8')}")
    for pattern in getattr(args, "text_glob", []) or []:
        for raw_path in sorted(glob.glob(str(pattern), recursive=True)):
            path = Path(raw_path)
            resolved = path.resolve()
            if path.is_file() and resolved not in seen_paths:
                seen_paths.add(resolved)
                parts.append(f"\n\n## FILE: {path}\n\n{path.read_text(encoding='utf-8')}")
    if parts:
        text = "\n".join(parts)
        if int(args.max_text_chars) > 0:
            return text[: int(args.max_text_chars)]
        return text
    return DEFAULT_TEXT


def build_model(args: argparse.Namespace, *, vocab_size: int):
    model_cls = load_native_model_class()
    return model_cls(
        vocab=int(vocab_size),
        max_seq_len=int(args.seq_len),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_kv_heads=int(args.n_kv_heads),
        d_ff=int(args.d_ff),
        dropout=float(args.dropout),
        backbone=str(args.backbone),
        encode_backbone=str(args.encode_backbone or args.backbone),
        think_backbone=str(args.think_backbone or args.backbone),
        decode_backbone=str(args.decode_backbone or args.backbone),
        think_structure=str(args.think_structure),
        trm_l_cycles=int(args.trm_l_cycles),
        trm_no_grad_inner_cycles=not bool(args.trm_full_grad_cycles),
        hybrid_layers=int(args.hybrid_layers),
        attn_every=int(args.attn_every),
        delta_backend=str(args.delta_backend),
        delta_head_dim=int(args.delta_head_dim) if int(args.delta_head_dim) > 0 else None,
        delta_num_v_heads=int(args.delta_num_v_heads) if int(args.delta_num_v_heads) > 0 else None,
        delta_expand_v=float(args.delta_expand_v),
        delta_mode=str(args.delta_mode),
        delta_use_short_conv=not bool(args.delta_no_short_conv),
        delta_conv_size=int(args.delta_conv_size),
        delta_norm_eps=float(args.delta_norm_eps),
        attention_backend=str(args.attention_backend),
        strict_backends=bool(args.strict_backends),
        tie_embeddings=bool(args.tie_embeddings),
    )


def train_language_model(
    model,
    train_windows: list[tuple[list[int], list[int]]],
    args: argparse.Namespace,
    *,
    device: torch.device,
    steps: int,
    think_steps: int,
    log_prefix: str = "",
) -> float:
    optimizer, optimizer_report = build_memory_efficient_optimizer(
        model,
        optimizer_name=str(getattr(args, "optimizer", "adamw")),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=device,
        galore_rank=int(getattr(args, "galore_rank", 128)),
        galore_update_proj_gap=int(getattr(args, "galore_update_proj_gap", 200)),
        galore_scale=float(getattr(args, "galore_scale", 0.25)),
        galore_proj_type=str(getattr(args, "galore_proj_type", "std")),
        galore_min_dim=int(getattr(args, "galore_min_dim", 128)),
        galore_include_embeddings=bool(getattr(args, "galore_include_embeddings", False)),
    )
    setattr(args, "_last_optimizer_report", optimizer_report)
    last_loss = 0.0
    for step in range(1, int(steps) + 1):
        model.train()
        x, y = batch_windows(
            train_windows,
            batch_size=int(args.batch_size),
            device=device,
        )
        logits = model(x, think_steps=int(think_steps))
        ce_loss = weighted_next_token_ce_loss(
            logits,
            y,
            eos_token_id=getattr(args, "_target_eos_token_id", None),
            eos_loss_weight=float(getattr(args, "eos_loss_weight", 1.0)),
        )
        repeat_weight = float(getattr(args, "repeat_unlikelihood_weight", 0.0))
        repeat_loss = logits.new_zeros(())
        if repeat_weight > 0.0:
            repeat_loss = repeat_unlikelihood_loss(logits, x, y)
        loss = ce_loss + repeat_weight * repeat_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if int(args.log_every) > 0 and (
            step == 1 or step % int(args.log_every) == 0 or step == int(steps)
        ):
            payload = {"step": step, "loss": last_loss}
            if log_prefix:
                payload["model"] = log_prefix
            if step == 1:
                payload["optimizer"] = optimizer_report
            if repeat_weight > 0.0:
                payload["ce_loss"] = float(ce_loss.detach().cpu())
                payload["repeat_unlikelihood_loss"] = float(repeat_loss.detach().cpu())
            if float(getattr(args, "eos_loss_weight", 1.0)) > 1.0:
                payload["eos_loss_weight"] = float(getattr(args, "eos_loss_weight", 1.0))
            print(json.dumps(payload, ensure_ascii=False))
    return last_loss


def train_probe(args: argparse.Namespace) -> dict[str, object]:
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    text = load_text(args)
    tokenizer = CharTokenizer.from_text(text)
    setattr(args, "_target_eos_token_id", tokenizer.eos_token_id)
    tokens = tokenizer.encode(text)
    split = max(int(args.seq_len) + 2, int(0.8 * len(tokens)))
    train_windows = make_windows(tokens[:split], seq_len=int(args.seq_len))
    eval_windows = make_windows(tokens[split - int(args.seq_len) :], seq_len=int(args.seq_len))
    if not train_windows or not eval_windows:
        raise ValueError("text is too short for seq_len")

    model = build_model(args, vocab_size=tokenizer.vocab_size).to(device)
    last_loss = train_language_model(
        model,
        train_windows,
        args,
        device=device,
        steps=int(args.steps),
        think_steps=int(args.train_think_steps),
        log_prefix="recurrent",
    )

    full_loss = eval_loss(
        model,
        eval_windows,
        batch_size=int(args.batch_size),
        device=device,
        think_steps=int(args.eval_think_steps),
    )
    think0_loss = eval_loss(
        model,
        eval_windows,
        batch_size=int(args.batch_size),
        device=device,
        think_steps=0,
    )
    off_loss = eval_loss(
        model,
        eval_windows,
        batch_size=int(args.batch_size),
        device=device,
        think_steps=int(args.eval_think_steps),
        thinking_block_off=True,
    )
    depth_sweep_losses: dict[str, float] = {}
    for depth in parse_depth_sweep(str(args.eval_depth_sweep)):
        if depth == int(args.eval_think_steps):
            depth_sweep_losses[str(depth)] = float(full_loss)
            continue
        if depth == 0:
            depth_sweep_losses[str(depth)] = float(think0_loss)
            continue
        depth_sweep_losses[str(depth)] = eval_loss(
            model,
            eval_windows,
            batch_size=int(args.batch_size),
            device=device,
            think_steps=int(depth),
        )
    shallow_depth_losses = [
        loss
        for depth, loss in depth_sweep_losses.items()
        if int(depth) < int(args.eval_think_steps)
    ]
    best_shallow_loss = min(shallow_depth_losses) if shallow_depth_losses else None
    full_vs_best_shallow = _safe_ratio(full_loss, best_shallow_loss)
    sample = generate_text(
        model,
        tokenizer,
        seed_text=str(args.seed_text),
        seq_len=int(args.seq_len),
        think_steps=int(args.eval_think_steps),
        max_new_chars=int(args.max_new_chars),
        device=device,
    )
    degeneracy = sample_degeneracy(sample)
    baseline_loss: float | None = None
    baseline_last_loss: float | None = None
    if int(args.baseline_steps) > 0:
        baseline_model = build_model(args, vocab_size=tokenizer.vocab_size).to(device)
        setattr(args, "_target_eos_token_id", tokenizer.eos_token_id)
        baseline_last_loss = train_language_model(
            baseline_model,
            train_windows,
            args,
            device=device,
            steps=int(args.baseline_steps),
            think_steps=0,
            log_prefix="think0_baseline",
        )
        baseline_loss = eval_loss(
            baseline_model,
            eval_windows,
            batch_size=int(args.batch_size),
            device=device,
            think_steps=0,
        )
    random_loss = math.log(max(2, tokenizer.vocab_size))
    reject_reasons = language_reject_reasons(
        args,
        full_loss=full_loss,
        think0_loss=think0_loss,
        off_loss=off_loss,
        baseline_loss=baseline_loss,
        random_loss=random_loss,
        degeneracy=degeneracy,
    )
    if (
        float(args.max_full_vs_best_shallow_loss_ratio) > 0.0
        and full_vs_best_shallow is not None
        and full_vs_best_shallow > float(args.max_full_vs_best_shallow_loss_ratio)
    ):
        reject_reasons.append("full_loss_regressed_vs_best_shallow_depth")
    report: dict[str, object] = {
        "status": "complete",
        "target_level": str(args.target_level),
        "train": vars(args),
        "backend_summary": summarize_backend(model),
        "optimizer": getattr(args, "_last_optimizer_report", {}),
        "vocab_size": tokenizer.vocab_size,
        "last_loss": last_loss,
        "baseline_last_loss": baseline_last_loss,
        "random_loss": random_loss,
        "eval_metrics": {
            "think_eval_loss": full_loss,
            "think0_loss": think0_loss,
            "thinking_block_off_loss": off_loss,
            "think0_baseline_loss": baseline_loss,
            "loss_ratios": {
                "full_vs_think0": _safe_ratio(full_loss, think0_loss),
                "full_vs_thinking_block_off": _safe_ratio(full_loss, off_loss),
                "full_vs_baseline": _safe_ratio(full_loss, baseline_loss),
                "full_vs_best_shallow_depth": full_vs_best_shallow,
            },
            "depth_sweep_loss": depth_sweep_losses,
            "best_shallow_depth_loss": best_shallow_loss,
            "sample_degeneracy": degeneracy,
            "sample": sample,
        },
        "accepted": not reject_reasons,
        "decision": str(args.accepted_decision) if not reject_reasons else "rejected",
        "reject_reasons": reject_reasons,
    }
    out_dir = Path(args.out_dir)
    if str(out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "args": vars(args),
                "report": report,
                "chars": tokenizer.chars,
            },
            out_dir / "last.pt",
        )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tiny QTRM-native text probe.")
    parser.add_argument("--out-dir", default="local_eval/qtrm_native_text_probe")
    parser.add_argument("--target-level", default="L3 QTRM-native language-slice smoke")
    parser.add_argument("--accepted-decision", default="accepted_l3_language_slice")
    parser.add_argument("--text-file", default="")
    parser.add_argument("--text-glob", action="append", default=[])
    parser.add_argument("--max-text-chars", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--backbone",
        choices=SUPPORTED_BACKBONES,
        default="mha_etd",
    )
    parser.add_argument(
        "--encode-backbone",
        choices=("", *SUPPORTED_BACKBONES),
        default="",
    )
    parser.add_argument(
        "--think-backbone",
        choices=("", *SUPPORTED_BACKBONES),
        default="",
    )
    parser.add_argument(
        "--decode-backbone",
        choices=("", *SUPPORTED_BACKBONES),
        default="",
    )
    parser.add_argument(
        "--think-structure",
        choices=("single", "trm_dual_z", "trm_dual_z_gated", "trm_dual_z_residual"),
        default="single",
    )
    parser.add_argument("--trm-l-cycles", type=int, default=1)
    parser.add_argument("--trm-full-grad-cycles", action="store_true")
    parser.add_argument("--hybrid-layers", type=int, default=4)
    parser.add_argument("--attn-every", type=int, default=4)
    parser.add_argument("--delta-backend", default="torch_gated_delta")
    parser.add_argument("--delta-head-dim", type=int, default=0)
    parser.add_argument("--delta-num-v-heads", type=int, default=0)
    parser.add_argument("--delta-expand-v", type=float, default=1.0)
    parser.add_argument("--delta-mode", default="chunk")
    parser.add_argument("--delta-no-short-conv", action="store_true")
    parser.add_argument("--delta-conv-size", type=int, default=4)
    parser.add_argument("--delta-norm-eps", type=float, default=1e-6)
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--strict-backends", action="store_true")
    parser.add_argument("--tie-embeddings", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--optimizer", choices=MEMORY_EFFICIENT_OPTIMIZERS, default="adamw")
    parser.add_argument("--galore-rank", type=int, default=128)
    parser.add_argument("--galore-update-proj-gap", type=int, default=200)
    parser.add_argument("--galore-scale", type=float, default=0.25)
    parser.add_argument(
        "--galore-proj-type",
        choices=("std", "reverse_std", "right", "left", "full"),
        default="std",
    )
    parser.add_argument("--galore-min-dim", type=int, default=128)
    parser.add_argument("--galore-include-embeddings", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--repeat-unlikelihood-weight", type=float, default=0.0)
    parser.add_argument("--eos-loss-weight", type=float, default=1.0)
    parser.add_argument("--train-think-steps", type=int, default=4)
    parser.add_argument("--eval-think-steps", type=int, default=4)
    parser.add_argument("--eval-depth-sweep", default="")
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed-text", default="QTRM native ")
    parser.add_argument("--max-new-chars", type=int, default=120)
    parser.add_argument("--max-random-loss-fraction", type=float, default=0.70)
    parser.add_argument("--min-unique-chars", type=float, default=8.0)
    parser.add_argument("--max-run-fraction", type=float, default=0.25)
    parser.add_argument("--max-full-vs-think0-loss-ratio", type=float, default=0.0)
    parser.add_argument("--max-full-vs-off-loss-ratio", type=float, default=0.0)
    parser.add_argument("--baseline-steps", type=int, default=0)
    parser.add_argument("--max-full-vs-baseline-loss-ratio", type=float, default=0.0)
    parser.add_argument("--max-full-vs-best-shallow-loss-ratio", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = train_probe(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if bool(report["accepted"]) else 1)


if __name__ == "__main__":
    main()
