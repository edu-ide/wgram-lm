#!/usr/bin/env python3
"""Train Stage102B provenance-graph reasoner through the same LM head.

Stage101Z showed that writing "Source ledger: S1 = verified" in plain text is
not enough.  Stage102B turns the ledger into a tiny structured graph register:

  source node -> trust edge -> claim support -> authority gate -> same LM head

This is still a falsification script, not a final parser.  It tests whether the
answer path improves when source binding is compiled into a reusable internal
register instead of being left as prose for the recurrent core to rediscover.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE101X = load_module(
    ROOT / "scripts" / "601_train_stage101x_counterfactual_answer_attractor.py",
    "stage102b_stage101x_counterfactual_answer_attractor",
)
STAGE101 = STAGE101X.STAGE101

YES = STAGE101X.YES
NO = STAGE101X.NO
SOURCE_RE = re.compile(r"\b(S\d+)\s*=\s*(verified|unverified)\b", re.IGNORECASE)
VALUE_RE = re.compile(r"Evidence value:\s*(?P<value>[^\n]+)", re.IGNORECASE)


def natural_source_sort_key(source_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"S(\d+)", str(source_id).strip(), flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), str(source_id)
    return 10_000, str(source_id)


def row_source_ids(row: dict[str, Any]) -> list[str]:
    sources: set[str] = set()
    for key in ("original_source", "counterfactual_source"):
        value = row.get(key)
        if value is not None:
            sources.add(str(value))
    for prompt_key in ("original_prompt", "counterfactual_prompt"):
        for match in SOURCE_RE.finditer(str(row.get(prompt_key, ""))):
            sources.add(str(match.group(1)).upper())
    if not sources:
        sources.update({"S1", "S2"})
    return sorted(sources, key=natural_source_sort_key)


def ledger_verified_map(row: dict[str, Any]) -> dict[str, float]:
    verified: dict[str, float] = {}
    for prompt_key in ("original_prompt", "counterfactual_prompt"):
        for match in SOURCE_RE.finditer(str(row.get(prompt_key, ""))):
            source_id = str(match.group(1)).upper()
            label = str(match.group(2)).lower()
            verified[source_id] = 1.0 if label == "verified" else 0.0
    source_ids = row_source_ids(row)
    if not verified and len(source_ids) >= 2:
        verified[source_ids[0]] = 1.0
        verified[source_ids[1]] = 0.0
    for source_id in source_ids:
        verified.setdefault(source_id, 0.0)
    return verified


def source_for_side(row: dict[str, Any], side: str) -> str:
    side = str(side).lower()
    if side not in {"original", "counterfactual"}:
        raise ValueError(f"bad side: {side!r}")
    return str(row.get(f"{side}_source", "")).upper()


def maybe_shuffle_source(row: dict[str, Any], source_id: str, *, source_id_shuffle: bool) -> str:
    if not source_id_shuffle:
        return source_id
    source_ids = row_source_ids(row)
    if len(source_ids) <= 1:
        return source_id
    if source_id not in source_ids:
        return source_id
    index = source_ids.index(source_id)
    return source_ids[(index + 1) % len(source_ids)]


def prompt_for_side(row: dict[str, Any], side: str) -> str:
    side = str(side).lower()
    return str(row.get(f"{side}_prompt", ""))


def evidence_value_supports_claim(row: dict[str, Any], side: str) -> float:
    prompt = prompt_for_side(row, side)
    match = VALUE_RE.search(prompt)
    if not match:
        return 1.0
    value = str(match.group("value")).strip().strip(".").lower()
    claim = str(row.get("claim", "")).strip().strip(".").lower()
    if not value:
        return 0.0
    if not claim:
        return 1.0
    return 1.0 if value in claim else 0.0


def build_graph_features(
    row: dict[str, Any],
    side: str,
    *,
    source_id_shuffle: bool = False,
    trust_edge_shuffle: bool = False,
    claim_support_shuffle: bool = False,
) -> dict[str, Any]:
    source_ids = row_source_ids(row)
    source_id = source_for_side(row, side)
    source_id = maybe_shuffle_source(row, source_id, source_id_shuffle=bool(source_id_shuffle))
    verified = ledger_verified_map(row)
    if trust_edge_shuffle:
        values = [verified.get(source, 0.0) for source in source_ids]
        rotated = values[1:] + values[:1]
        verified = {source: float(value) for source, value in zip(source_ids, rotated, strict=True)}
    support = evidence_value_supports_claim(row, side)
    if claim_support_shuffle:
        support = 1.0 - float(support)
    return {
        "source_id": source_id,
        "source_index": int(source_ids.index(source_id) if source_id in source_ids else 0),
        "source_verified": float(verified.get(source_id, 0.0)),
        "claim_supported": float(support),
    }


class ProvenanceGraphReasoner(nn.Module):
    """Tiny graph-to-register module for source authority and claim support."""

    def __init__(self, d_model: int, max_sources: int = 16, hidden_dim: int | None = None) -> None:
        super().__init__()
        width = int(hidden_dim or d_model)
        self.source_embedding = nn.Embedding(int(max_sources), int(d_model))
        self.trust_proj = nn.Linear(1, int(d_model))
        self.support_proj = nn.Linear(1, int(d_model))
        self.norm = nn.LayerNorm(int(d_model))
        self.message = nn.Sequential(
            nn.Linear(int(d_model), width),
            nn.SiLU(),
            nn.Linear(width, int(d_model)),
            nn.Tanh(),
        )
        self.authority_gate = nn.Sequential(
            nn.Linear(int(d_model), width),
            nn.SiLU(),
            nn.Linear(width, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: list[dict[str, Any]],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not features:
            raise ValueError("features must not be empty")
        source_index = torch.tensor(
            [int(item["source_index"]) for item in features],
            dtype=torch.long,
            device=device,
        )
        source_index = source_index.clamp(0, self.source_embedding.num_embeddings - 1)
        source_verified = torch.tensor(
            [float(item["source_verified"]) for item in features],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
        claim_supported = torch.tensor(
            [float(item["claim_supported"]) for item in features],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
        state = self.source_embedding(source_index)
        state = state + self.trust_proj(source_verified) + self.support_proj(claim_supported)
        state = self.norm(state)
        authority = self.authority_gate(state)
        register = authority * self.message(state)
        metrics = {
            "rows": int(len(features)),
            "mean_authority": float(authority.detach().float().mean().cpu().item()),
            "min_authority": float(authority.detach().float().min().cpu().item()),
            "max_authority": float(authority.detach().float().max().cpu().item()),
            "mean_source_verified": float(source_verified.detach().float().mean().cpu().item()),
            "mean_claim_supported": float(claim_supported.detach().float().mean().cpu().item()),
        }
        return register, metrics


def choice_mean_logprob_with_register(
    model: torch.nn.Module,
    gd_module: Any,
    *,
    prompt: str,
    answer: str,
    register: torch.Tensor,
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    think_steps: int,
) -> tuple[torch.Tensor, int]:
    input_ids, labels, attention_mask = gd_module.build_choice_tensors(
        prompt=str(prompt),
        answer=str(answer),
        seq_len=int(seq_len),
        byte_offset=int(byte_offset),
        device=device,
    )
    if register.ndim == 1:
        register = register.unsqueeze(0)
    logits = model.forward_logits(
        input_ids,
        attention_mask,
        think_steps=int(think_steps),
        external_register=register,
    )
    length = min(int(logits.shape[1]), int(labels.shape[1]))
    log_probs = F.log_softmax(logits[:, :length].float(), dim=-1)
    labels = labels[:, :length]
    mask = labels.ne(STAGE101.IGNORE_LABEL_ID)
    if not bool(mask.any()):
        raise ValueError("choice row has no answer target tokens")
    token_log_probs = log_probs[mask].gather(1, labels[mask].unsqueeze(1)).squeeze(1)
    return token_log_probs.mean(), int(token_log_probs.numel())


def yes_no_logprob_scores_with_register(
    model: torch.nn.Module,
    gd_module: Any,
    *,
    prompt: str,
    register: torch.Tensor,
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    think_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    yes_mean, _yes_tokens = choice_mean_logprob_with_register(
        model,
        gd_module,
        prompt=str(prompt),
        answer=YES,
        register=register,
        seq_len=int(seq_len),
        byte_offset=int(byte_offset),
        device=device,
        think_steps=int(think_steps),
    )
    no_mean, _no_tokens = choice_mean_logprob_with_register(
        model,
        gd_module,
        prompt=str(prompt),
        answer=NO,
        register=register,
        seq_len=int(seq_len),
        byte_offset=int(byte_offset),
        device=device,
        think_steps=int(think_steps),
    )
    return yes_mean, no_mean


def pair_lm_head_loss_with_graph(
    model: torch.nn.Module,
    gd_module: Any,
    reasoner: ProvenanceGraphReasoner,
    row: dict[str, Any],
    *,
    depth: int,
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    target_margin: float,
    target_gap: float,
    gap_weight: float,
    target_nll_weight: float,
    source_id_shuffle: bool = False,
    trust_edge_shuffle: bool = False,
    claim_support_shuffle: bool = False,
    register_off: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    original_features = build_graph_features(
        row,
        "original",
        source_id_shuffle=source_id_shuffle,
        trust_edge_shuffle=trust_edge_shuffle,
        claim_support_shuffle=claim_support_shuffle,
    )
    counterfactual_features = build_graph_features(
        row,
        "counterfactual",
        source_id_shuffle=source_id_shuffle,
        trust_edge_shuffle=trust_edge_shuffle,
        claim_support_shuffle=claim_support_shuffle,
    )
    registers, graph_metrics = reasoner([original_features, counterfactual_features], device=device)
    if register_off:
        registers = torch.zeros_like(registers)

    original_yes, original_no = yes_no_logprob_scores_with_register(
        model,
        gd_module,
        prompt=str(row["original_prompt"]),
        register=registers[0:1],
        seq_len=int(seq_len),
        byte_offset=int(byte_offset),
        device=device,
        think_steps=int(depth),
    )
    counterfactual_yes, counterfactual_no = yes_no_logprob_scores_with_register(
        model,
        gd_module,
        prompt=str(row["counterfactual_prompt"]),
        register=registers[1:2],
        seq_len=int(seq_len),
        byte_offset=int(byte_offset),
        device=device,
        think_steps=int(depth),
    )
    original_yes_minus_no = original_yes.float() - original_no.float()
    counterfactual_yes_minus_no = counterfactual_yes.float() - counterfactual_no.float()
    original_margin = STAGE101X.answer_margin_from_yes_no(
        original_yes_minus_no,
        str(row["original_answer"]),
    )
    counterfactual_margin = STAGE101X.answer_margin_from_yes_no(
        counterfactual_yes_minus_no,
        str(row["counterfactual_answer"]),
    )
    gap = STAGE101X.counterfactual_gap(
        original_yes_minus_no,
        counterfactual_yes_minus_no,
        original_answer=str(row["original_answer"]),
    )
    rank_loss = 0.5 * (
        F.softplus(float(target_margin) - original_margin.float())
        + F.softplus(float(target_margin) - counterfactual_margin.float())
    )
    gap_loss = F.softplus(float(target_gap) - gap.float())
    original_target_nll = -(original_yes if str(row["original_answer"]) == YES else original_no)
    counterfactual_target_nll = -(
        counterfactual_yes if str(row["counterfactual_answer"]) == YES else counterfactual_no
    )
    target_nll = 0.5 * (original_target_nll.float() + counterfactual_target_nll.float())
    loss = rank_loss + float(gap_weight) * gap_loss + float(target_nll_weight) * target_nll
    return loss, {
        "id": row.get("id"),
        "pair_feature": row.get("pair_feature"),
        "depth": int(depth),
        "loss": float(loss.detach().cpu().item()),
        "rank_loss": float(rank_loss.detach().cpu().item()),
        "gap_loss": float(gap_loss.detach().cpu().item()),
        "target_nll": float(target_nll.detach().cpu().item()),
        "original_margin": float(original_margin.detach().cpu().item()),
        "counterfactual_margin": float(counterfactual_margin.detach().cpu().item()),
        "counterfactual_gap": float(gap.detach().cpu().item()),
        "original_correct": bool(float(original_margin.detach().cpu().item()) > 0.0),
        "counterfactual_correct": bool(float(counterfactual_margin.detach().cpu().item()) > 0.0),
        "original_source_index": int(original_features["source_index"]),
        "counterfactual_source_index": int(counterfactual_features["source_index"]),
        "original_source_verified": float(original_features["source_verified"]),
        "counterfactual_source_verified": float(counterfactual_features["source_verified"]),
        **{f"graph_{key}": value for key, value in graph_metrics.items()},
    }


def pair_multi_depth_loss_with_graph(
    model: torch.nn.Module,
    gd_module: Any,
    reasoner: ProvenanceGraphReasoner,
    row: dict[str, Any],
    *,
    depths: list[int],
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    target_margin: float,
    target_gap: float,
    gap_weight: float,
    target_nll_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    losses: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for depth in depths:
        with STAGE101.make_amp_context(device, amp_dtype):
            loss, metrics = pair_lm_head_loss_with_graph(
                model,
                gd_module,
                reasoner,
                row,
                depth=int(depth),
                seq_len=int(seq_len),
                byte_offset=int(byte_offset),
                device=device,
                target_margin=float(target_margin),
                target_gap=float(target_gap),
                gap_weight=float(gap_weight),
                target_nll_weight=float(target_nll_weight),
            )
        losses.append(loss)
        rows.append(metrics)
    total = torch.stack([loss.float() for loss in losses]).mean()
    last = rows[-1]
    return total, {
        "id": row.get("id"),
        "pair_feature": row.get("pair_feature"),
        "depths": [int(depth) for depth in depths],
        "loss": float(total.detach().cpu().item()),
        "last_original_margin": float(last["original_margin"]),
        "last_counterfactual_margin": float(last["counterfactual_margin"]),
        "last_counterfactual_gap": float(last["counterfactual_gap"]),
        "last_pair_correct": bool(last["original_correct"] and last["counterfactual_correct"]),
        "last_graph_mean_authority": float(last["graph_mean_authority"]),
    }


@torch.no_grad()
def evaluate_pairs(
    *,
    model: torch.nn.Module,
    gd_module: Any,
    reasoner: ProvenanceGraphReasoner,
    rows: list[dict[str, Any]],
    split: str,
    depths: list[int],
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    target_margin: float,
    target_gap: float,
    gap_weight: float,
    target_nll_weight: float,
) -> dict[str, Any]:
    model.eval()
    reasoner.eval()
    by_depth: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    for depth in depths:
        depth_rows: list[dict[str, Any]] = []
        for row in rows:
            with STAGE101.make_amp_context(device, amp_dtype):
                _loss, metrics = pair_lm_head_loss_with_graph(
                    model,
                    gd_module,
                    reasoner,
                    row,
                    depth=int(depth),
                    seq_len=int(seq_len),
                    byte_offset=int(byte_offset),
                    device=device,
                    target_margin=float(target_margin),
                    target_gap=float(target_gap),
                    gap_weight=float(gap_weight),
                    target_nll_weight=float(target_nll_weight),
                )
            metrics["split"] = split
            depth_rows.append(metrics)
            detailed_rows.append(metrics)
        by_depth.append(STAGE101X.build_pair_report(depth_rows, split=split, depth=int(depth)))
    return {
        "split": split,
        "depths": by_depth,
        "accepted": bool(by_depth and by_depth[-1]["accepted"]),
        "rows": detailed_rows,
    }


def build_checkpoint_args(ckpt_args: argparse.Namespace, args: argparse.Namespace) -> dict[str, Any]:
    values = vars(ckpt_args).copy()
    values.update(
        {
            "stage102b_provenance_graph_reasoner": True,
            "stage102b_source_checkpoint": str(args.checkpoint),
            "stage102b_train_jsonl": str(args.train_jsonl),
            "stage102b_eval_jsonl": str(args.eval_jsonl),
            "stage102b_depths": [int(depth) for depth in args.depths],
            "stage102b_steps": int(args.steps),
            "stage102b_batch_size": int(args.batch_size),
            "stage102b_lr": float(args.lr),
            "stage102b_base_lr": float(args.base_lr),
        }
    )
    return values


def maybe_load_reasoner_state(reasoner: ProvenanceGraphReasoner, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("graph_reasoner_state_dict"), dict):
        reasoner.load_state_dict(payload["graph_reasoner_state_dict"], strict=False)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    reasoner: ProvenanceGraphReasoner,
    optimizer: torch.optim.Optimizer,
    args_payload: dict[str, Any],
    step: int,
    eval_before: dict[str, Any],
    eval_after: dict[str, Any],
    history: list[dict[str, Any]],
    loaded: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "graph_reasoner_state_dict": reasoner.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": args_payload,
        "dataset": dict(loaded.get("dataset_summary", {})),
        "model": dict(loaded.get("model_summary", {})),
        "loss_history": history,
        "eval_before": eval_before,
        "eval_after": eval_after,
        "stage102b_provenance_graph_reasoner": True,
    }
    tmp = path.with_name(f".{path.name}.tmp.{int(time.time())}")
    try:
        torch.save(payload, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = STAGE101.load_jsonl(Path(args.train_jsonl))
    eval_rows = STAGE101.load_jsonl(Path(args.eval_jsonl))
    device = torch.device(str(args.device))
    amp_dtype = STAGE101.resolve_amp_dtype(str(args.amp_dtype))
    depth_probe = STAGE101.load_depth_probe_module()
    gd_module = STAGE101.load_gd_module()
    _trainer, _prefix, ckpt_args, loaded = depth_probe.load_checkpoint_model(
        checkpoint_path=Path(args.checkpoint),
        sampled_data=str(args.sampled_data),
        out_dir=str(out_dir),
        device=device,
        amp_dtype=str(args.amp_dtype),
    )
    model = loaded["model"]
    reasoner = ProvenanceGraphReasoner(
        d_model=int(ckpt_args.d_model),
        max_sources=int(args.max_sources),
        hidden_dim=int(args.reasoner_hidden_dim or ckpt_args.d_model),
    ).to(device)
    maybe_load_reasoner_state(reasoner, Path(args.checkpoint))

    train_base = float(args.base_lr) > 0.0
    for parameter in model.parameters():
        parameter.requires_grad_(bool(train_base))
    model.train(bool(train_base))
    reasoner.train()

    optimizer_groups: list[dict[str, Any]] = [{"params": list(reasoner.parameters()), "lr": float(args.lr)}]
    if train_base:
        optimizer_groups.append(
            {
                "params": [parameter for parameter in model.parameters() if parameter.requires_grad],
                "lr": float(args.base_lr),
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=float(args.weight_decay),
    )
    tokenizer_info = dict(loaded.get("dataset_summary", {}).get("tokenizer_info") or {})
    byte_offset = int(args.byte_offset if int(args.byte_offset) >= 0 else tokenizer_info.get("byte_offset", 2))
    seq_len = int(args.seq_len or ckpt_args.seq_len)
    depths = sorted({int(depth) for depth in args.depths})

    eval_kwargs = {
        "model": model,
        "gd_module": gd_module,
        "reasoner": reasoner,
        "depths": depths,
        "seq_len": seq_len,
        "byte_offset": byte_offset,
        "device": device,
        "amp_dtype": amp_dtype,
        "target_margin": float(args.target_margin),
        "target_gap": float(args.target_gap),
        "gap_weight": float(args.gap_weight),
        "target_nll_weight": float(args.target_nll_weight),
    }
    eval_before = {
        "train": evaluate_pairs(rows=train_rows, split="train", **eval_kwargs),
        "heldout": evaluate_pairs(rows=eval_rows, split="heldout", **eval_kwargs),
    }

    history: list[dict[str, Any]] = []
    for step in range(1, int(args.steps) + 1):
        batch_rows = STAGE101X.batch_rows_for_step(
            train_rows,
            step=int(step),
            batch_size=int(args.batch_size),
        )
        step_depths = STAGE101X.training_depths_for_step(
            depths,
            step=int(step),
            single_depth=bool(args.single_depth_per_step),
        )
        optimizer.zero_grad(set_to_none=True)
        model.train(bool(train_base))
        reasoner.train()
        row_losses: list[torch.Tensor] = []
        row_metrics: list[dict[str, Any]] = []
        for row in batch_rows:
            row_loss, metrics = pair_multi_depth_loss_with_graph(
                model,
                gd_module,
                reasoner,
                row,
                depths=step_depths,
                seq_len=seq_len,
                byte_offset=byte_offset,
                device=device,
                amp_dtype=amp_dtype,
                target_margin=float(args.target_margin),
                target_gap=float(args.target_gap),
                gap_weight=float(args.gap_weight),
                target_nll_weight=float(args.target_nll_weight),
            )
            row_losses.append(row_loss)
            row_metrics.append(metrics)
        loss = torch.stack([item.float() for item in row_losses]).mean()
        last_metrics = row_metrics[-1]
        metrics = {
            "step": int(step),
            "id": last_metrics.get("id"),
            "pair_feature": last_metrics.get("pair_feature"),
            "depths": step_depths,
            "batch_size": int(len(batch_rows)),
            "loss": float(loss.detach().cpu().item()),
            "batch_pair_accuracy": sum(1 for item in row_metrics if bool(item["last_pair_correct"]))
            / float(len(row_metrics)),
            "min_original_margin": float(min(float(item["last_original_margin"]) for item in row_metrics)),
            "min_counterfactual_margin": float(
                min(float(item["last_counterfactual_margin"]) for item in row_metrics)
            ),
            "min_counterfactual_gap": float(
                min(float(item["last_counterfactual_gap"]) for item in row_metrics)
            ),
            "mean_graph_authority": float(
                sum(float(item["last_graph_mean_authority"]) for item in row_metrics)
                / float(len(row_metrics))
            ),
            "last_original_margin": float(last_metrics["last_original_margin"]),
            "last_counterfactual_margin": float(last_metrics["last_counterfactual_margin"]),
            "last_counterfactual_gap": float(last_metrics["last_counterfactual_gap"]),
            "last_pair_correct": bool(last_metrics["last_pair_correct"]),
        }
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in list(reasoner.parameters()) + list(model.parameters()) if parameter.grad is not None],
            max_norm=float(args.grad_clip),
        )
        optimizer.step()
        if step == 1 or step % int(args.log_every) == 0:
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            history.append(metrics)

    eval_after = {
        "train": evaluate_pairs(rows=train_rows, split="train", **eval_kwargs),
        "heldout": evaluate_pairs(rows=eval_rows, split="heldout", **eval_kwargs),
    }
    accepted = bool(eval_after["heldout"]["accepted"])
    report = {
        "decision": "stage102b_provenance_graph_reasoner_train",
        "accepted": accepted,
        "checkpoint_in": str(args.checkpoint),
        "checkpoint_out": str(out_dir / "last_model.pt"),
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl),
        "depths": depths,
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "train_base": bool(train_base),
        "eval_before": eval_before,
        "eval_after": eval_after,
        "plain_language_read": (
            "Stage102B gives the model a small internal desk: source card, "
            "trust mark, claim-support mark, authority gate, then the same "
            "LM head speaks. It tests source binding, not arbitrary parsing yet."
        ),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_checkpoint(
        out_dir / "last_model.pt",
        model=model,
        reasoner=reasoner,
        optimizer=optimizer,
        args_payload=build_checkpoint_args(ckpt_args, args),
        step=int(args.steps),
        eval_before=eval_before,
        eval_after=eval_after,
        history=history,
        loaded=loaded,
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-jsonl", default="data/eval/stage101z_source_binding_ledger_train_probe.jsonl")
    parser.add_argument("--eval-jsonl", default="data/eval/stage101z_source_binding_ledger_heldout_probe.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-lr", type=float, default=0.0)
    parser.add_argument("--target-margin", type=float, default=0.25)
    parser.add_argument("--target-gap", type=float, default=0.5)
    parser.add_argument("--gap-weight", type=float, default=1.0)
    parser.add_argument("--target-nll-weight", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=80)
    parser.add_argument("--single-depth-per-step", action="store_true")
    parser.add_argument("--max-sources", type=int, default=16)
    parser.add_argument("--reasoner-hidden-dim", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", default="bf16")
    parser.add_argument("--sampled-data", default="")
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--byte-offset", type=int, default=-1)
    return parser


def main() -> None:
    report = run_train(build_arg_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
