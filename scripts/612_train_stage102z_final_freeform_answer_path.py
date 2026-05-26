#!/usr/bin/env python3
"""Train/evaluate the Stage102Z final free-form provenance answer path.

This is the promoted end-to-end gate for the Stage102 provenance thread:

  free-form evidence text
  -> context/observation provenance cards
  -> graph reasoner + data-world model
  -> gated answer register
  -> same BLT LM head yes/no

Reader-only card accuracy is not the promotion metric here.  The run is
accepted only if the same LM head answers clean evidence as yes, corrupted
evidence as no, and the gain disappears when the world signal or register is
disabled.
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


STAGE102E = load_module(
    ROOT / "scripts" / "608_train_stage102e_world_model_gated_answer_register.py",
    "stage102z_stage102e_world_model_gated_answer_register",
)
STAGE102G = load_module(
    ROOT / "scripts" / "610_eval_stage102g_freeform_provenance_frontend.py",
    "stage102z_stage102g_freeform_provenance_frontend",
)
STAGE102B = STAGE102E.STAGE102B
STAGE102D = STAGE102E.STAGE102D
STAGE101 = STAGE102E.STAGE101
YES = STAGE102E.YES
NO = STAGE102E.NO


TRUSTED_RE = re.compile(r"Trusted source for this claim is\s+(?P<source>S\d+)", re.IGNORECASE)
OTHER_RE = re.compile(r"Other source\s+(?P<source>S\d+)\s+is\s+(?P<status>verified|unverified)", re.IGNORECASE)
CLAIM_RE = re.compile(r"Claim under review:\s*(?P<claim>.+?)\s*$", re.IGNORECASE | re.MULTILINE)
EXPECTED_VALUE_RE = re.compile(
    r"Expected support value:\s*(?P<value>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
OBS_SOURCE_RE = re.compile(r"Observed evidence came from\s+(?P<source>S\d+)", re.IGNORECASE)
OBS_STATUS_RE = re.compile(
    r"Observed source status:\s*(?P<status>verified|unverified)",
    re.IGNORECASE,
)
OBS_VALUE_RE = re.compile(r"Observed evidence says\s+(?P<value>.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def other_source(source_id: str) -> str:
    source_id = str(source_id).upper()
    if source_id == "S1":
        return "S2"
    if source_id == "S2":
        return "S1"
    raise ValueError(f"bad source id: {source_id!r}")


def _match(pattern: re.Pattern[str], text: str, group: str) -> str:
    match = pattern.search(str(text))
    if not match:
        raise ValueError(f"prompt missing {group}")
    return str(match.group(group)).strip()


def _support(value: str, claim: str) -> float:
    value = str(value).strip().strip(".").lower()
    claim = str(claim).strip().strip(".").lower()
    if not value or not claim:
        return 0.0
    if value in claim:
        return 1.0
    value_tokens = re.findall(r"[a-z0-9]+", value)
    claim_tokens = set(re.findall(r"[a-z0-9]+", claim))
    if value_tokens and all(token in claim_tokens for token in value_tokens):
        return 1.0
    return 0.0


def _source_ids_from_prompt(prompt: str) -> list[str]:
    sources = {
        _match(TRUSTED_RE, prompt, "source").upper(),
        _match(OBS_SOURCE_RE, prompt, "source").upper(),
    }
    other = OTHER_RE.search(str(prompt))
    if other:
        sources.add(str(other.group("source")).upper())
    return sorted(sources, key=STAGE102B.natural_source_sort_key)


def prompt_to_final_cards(prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
    source_ids = _source_ids_from_prompt(prompt)
    trusted_source = _match(TRUSTED_RE, prompt, "source").upper()
    observed_source = _match(OBS_SOURCE_RE, prompt, "source").upper()
    claim = _match(CLAIM_RE, prompt, "claim")
    expected_value = _match(EXPECTED_VALUE_RE, prompt, "value")
    observed_value = _match(OBS_VALUE_RE, prompt, "value")
    observed_status = _match(OBS_STATUS_RE, prompt, "status").lower()
    trusted_index = int(source_ids.index(trusted_source))
    observed_index = int(source_ids.index(observed_source))
    expected_claim_supported = _support(expected_value, claim)
    observed_claim_supported = _support(observed_value, claim)
    observed_verified = 1.0 if observed_status == "verified" else 0.0
    graph_features = {
        "source_id": trusted_source,
        "source_index": trusted_index,
        "source_verified": 1.0,
        "claim_supported": float(expected_claim_supported),
    }
    world_example = {
        "source_index": observed_index,
        "verified_source_index": trusted_index,
        "context_source_index": trusted_index,
        "context_verified_source_index": trusted_index,
        "expected_source_verified": 1.0,
        "expected_claim_supported": float(expected_claim_supported),
        "observed_source_verified": float(observed_verified),
        "claim_supported": float(observed_claim_supported),
    }
    return graph_features, world_example


def support_conflict_value(value: str) -> str:
    value = str(value).strip().strip(".")
    if value.lower() != "mismatch":
        return "mismatch"
    return "different"


def final_prompt(
    *,
    claim: str,
    trusted_source: str,
    observed_source: str,
    observed_status: str,
    expected_value: str,
    observed_value: str,
) -> str:
    trusted_source = str(trusted_source).upper()
    other = other_source(trusted_source)
    return (
        "Provenance context:\n"
        f"Trusted source for this claim is {trusted_source}. Other source {other} is unverified.\n"
        f"Claim under review: {claim}\n"
        f"Expected support value: {expected_value}\n"
        "Observation:\n"
        f"Observed evidence came from {str(observed_source).upper()}.\n"
        f"Observed source status: {str(observed_status).lower()}.\n"
        f"Observed evidence says {observed_value}.\n"
        "Can answer now? yes or no.\n"
        "A:"
    )


def build_final_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in rows:
        side = STAGE102E.answerable_side(row)
        trusted_source = str(row["verified_source"]).upper()
        prompt_source, value = STAGE102G._source_value_from_template_prompt(str(row[f"{side}_prompt"]))
        if str(prompt_source).upper() != trusted_source:
            raise ValueError(f"answerable side source mismatch for {row.get('id')}")
        claim = str(row["claim"])
        specs = [
            ("clean", trusted_source, "verified", value, YES),
            ("source_id_conflict", other_source(trusted_source), "unverified", value, NO),
            ("trust_edge_conflict", trusted_source, "unverified", value, NO),
            ("support_conflict", trusted_source, "verified", support_conflict_value(value), NO),
        ]
        for corruption, observed_source, observed_status, observed_value, answer in specs:
            prompt = final_prompt(
                claim=claim,
                trusted_source=trusted_source,
                observed_source=observed_source,
                observed_status=observed_status,
                expected_value=value,
                observed_value=observed_value,
            )
            graph_features, world_example = prompt_to_final_cards(prompt)
            cases.append(
                {
                    "id": row.get("id"),
                    "side": side,
                    "prompt": prompt,
                    "corruption": corruption,
                    "graph_features": graph_features,
                    "world_example": world_example,
                    "answer": answer,
                    "negative_answer": NO if answer == YES else YES,
                    "is_clean": corruption == "clean",
                }
            )
    if not cases:
        raise ValueError("no final cases")
    return cases


def case_loss(
    model: torch.nn.Module,
    gd_module: Any,
    gated_register: torch.nn.Module,
    case: dict[str, Any],
    *,
    depth: int,
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    target_margin: float,
    target_nll_weight: float,
    clean_loss_weight: float = 1.0,
    world_off: bool = False,
    register_off: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    register, register_metrics = gated_register(
        case["graph_features"],
        case["world_example"],
        device=device,
        world_off=bool(world_off),
    )
    if bool(register_off):
        register = torch.zeros_like(register)
    target_mean, _target_tokens = STAGE102E.choice_logprob_with_register(
        model,
        gd_module,
        prompt=str(case["prompt"]),
        answer=str(case["answer"]),
        register=register,
        seq_len=int(seq_len),
        byte_offset=int(byte_offset),
        device=device,
        think_steps=int(depth),
    )
    negative_mean, _negative_tokens = STAGE102E.choice_logprob_with_register(
        model,
        gd_module,
        prompt=str(case["prompt"]),
        answer=str(case["negative_answer"]),
        register=register,
        seq_len=int(seq_len),
        byte_offset=int(byte_offset),
        device=device,
        think_steps=int(depth),
    )
    margin = target_mean.float() - negative_mean.float()
    target_nll = -target_mean.float()
    loss = F.softplus(float(target_margin) - margin) + float(target_nll_weight) * target_nll
    if bool(case.get("is_clean")):
        loss = loss * float(clean_loss_weight)
    return loss, {
        "id": case.get("id"),
        "side": case.get("side"),
        "corruption": case.get("corruption"),
        "is_clean": bool(case.get("is_clean")),
        "depth": int(depth),
        "world_off": bool(world_off),
        "register_off": bool(register_off),
        "loss": float(loss.detach().cpu().item()),
        "margin": float(margin.detach().cpu().item()),
        "target_nll": float(target_nll.detach().cpu().item()),
        "clean_loss_weight": float(clean_loss_weight),
        "correct": bool(float(margin.detach().cpu().item()) > 0.0),
        **register_metrics,
    }


def case_report(rows: list[dict[str, Any]], *, split: str, depth: int, world_off: bool, register_off: bool) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows required")
    by_corruption: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_corruption.setdefault(str(row["corruption"]), []).append(row)

    def accuracy(items: list[dict[str, Any]]) -> float:
        return sum(1 for item in items if bool(item["correct"])) / float(len(items)) if items else 0.0

    clean_rows = [row for row in rows if str(row["corruption"]) == "clean"]
    corrupt_rows = [row for row in rows if str(row["corruption"]) != "clean"]
    return {
        "split": split,
        "depth": int(depth),
        "world_off": bool(world_off),
        "register_off": bool(register_off),
        "rows": int(len(rows)),
        "accuracy": accuracy(rows),
        "clean_accuracy": accuracy(clean_rows),
        "corrupt_no_accuracy": accuracy(corrupt_rows),
        "min_margin": float(min(float(row["margin"]) for row in rows)),
        "mean_margin": float(sum(float(row["margin"]) for row in rows) / float(len(rows))),
        "by_corruption": {
            key: {
                "rows": len(items),
                "accuracy": accuracy(items),
                "min_margin": float(min(float(row["margin"]) for row in items)),
            }
            for key, items in sorted(by_corruption.items())
        },
        "accepted": bool(accuracy(rows) == 1.0 and min(float(row["margin"]) for row in rows) > 0.0),
    }


@torch.no_grad()
def evaluate_cases(
    *,
    model: torch.nn.Module,
    gd_module: Any,
    gated_register: torch.nn.Module,
    cases: list[dict[str, Any]],
    split: str,
    depths: list[int],
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    target_margin: float,
    target_nll_weight: float,
    clean_loss_weight: float = 1.0,
    world_off: bool = False,
    register_off: bool = False,
) -> dict[str, Any]:
    model.eval()
    gated_register.eval()
    by_depth: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for depth in depths:
        rows: list[dict[str, Any]] = []
        for case in cases:
            with STAGE101.make_amp_context(device, amp_dtype):
                _loss, metrics = case_loss(
                    model,
                    gd_module,
                    gated_register,
                    case,
                    depth=int(depth),
                    seq_len=int(seq_len),
                    byte_offset=int(byte_offset),
                    device=device,
                    target_margin=float(target_margin),
                    target_nll_weight=float(target_nll_weight),
                    clean_loss_weight=float(clean_loss_weight),
                    world_off=bool(world_off),
                    register_off=bool(register_off),
                )
            metrics["split"] = split
            rows.append(metrics)
            details.append(metrics)
        by_depth.append(
            case_report(
                rows,
                split=split,
                depth=int(depth),
                world_off=world_off,
                register_off=register_off,
            )
        )
    return {
        "split": split,
        "world_off": bool(world_off),
        "register_off": bool(register_off),
        "depths": by_depth,
        "accepted": bool(by_depth and by_depth[-1]["accepted"]),
        "rows": details,
    }


def load_final_stack(args: argparse.Namespace, *, out_dir: Path, device: torch.device) -> tuple[Any, Any, Any, int, int]:
    depth_probe = STAGE101.load_depth_probe_module()
    gd_module = STAGE101.load_gd_module()
    _trainer, _prefix, ckpt_args, loaded = depth_probe.load_checkpoint_model(
        checkpoint_path=Path(args.answer_checkpoint),
        sampled_data=str(args.sampled_data),
        out_dir=str(out_dir),
        device=device,
        amp_dtype=str(args.amp_dtype),
    )
    model = loaded["model"]
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    graph_reasoner = STAGE102B.ProvenanceGraphReasoner(
        d_model=int(ckpt_args.d_model),
        max_sources=int(args.max_sources),
        hidden_dim=int(args.graph_hidden_dim or ckpt_args.d_model),
    ).to(device)
    STAGE102B.maybe_load_reasoner_state(graph_reasoner, Path(args.answer_checkpoint))
    graph_reasoner.eval()
    for parameter in graph_reasoner.parameters():
        parameter.requires_grad_(False)

    world_model, world_info = STAGE102E.load_world_model_checkpoint(Path(args.world_model_checkpoint), device=device)
    gated_register = STAGE102E.WorldModelGatedAnswerRegister(
        d_model=int(ckpt_args.d_model),
        graph_reasoner=graph_reasoner,
        world_model=world_model,
        world_d_model=int(world_info["d_model"]),
        hidden_dim=int(args.fusion_hidden_dim or ckpt_args.d_model),
    ).to(device)
    if args.init_gated_checkpoint:
        payload = torch.load(Path(args.init_gated_checkpoint), map_location="cpu", weights_only=False)
        gated_register.load_state_dict(payload["gated_register_state_dict"], strict=True)
    tokenizer_info = dict(loaded.get("dataset_summary", {}).get("tokenizer_info") or {})
    byte_offset = int(args.byte_offset if int(args.byte_offset) >= 0 else tokenizer_info.get("byte_offset", 2))
    seq_len = int(args.seq_len or ckpt_args.seq_len)
    return model, gd_module, gated_register, seq_len, byte_offset


def batch_cases_for_step(cases: list[dict[str, Any]], *, step: int, batch_size: int) -> list[dict[str, Any]]:
    start = (int(step) - 1) % len(cases)
    return [cases[(start + offset) % len(cases)] for offset in range(max(1, int(batch_size)))]


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(args.device))
    amp_dtype = STAGE101.resolve_amp_dtype(str(args.amp_dtype))
    train_rows = STAGE101.load_jsonl(Path(args.train_jsonl))
    eval_rows = STAGE101.load_jsonl(Path(args.eval_jsonl))
    if int(args.max_train_rows) > 0:
        train_rows = train_rows[: int(args.max_train_rows)]
    if int(args.max_eval_rows) > 0:
        eval_rows = eval_rows[: int(args.max_eval_rows)]
    train_cases = build_final_cases(train_rows)
    eval_cases = build_final_cases(eval_rows)
    model, gd_module, gated_register, seq_len, byte_offset = load_final_stack(args, out_dir=out_dir, device=device)
    trainable = [
        parameter
        for name, parameter in gated_register.named_parameters()
        if name.startswith("world_to_delta.") or name.startswith("world_gate.")
    ]
    for parameter in gated_register.parameters():
        parameter.requires_grad_(False)
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))
    depths = sorted({int(depth) for depth in args.depths})
    eval_depths = sorted({int(depth) for depth in (args.eval_depths or args.depths)})
    eval_kwargs = {
        "model": model,
        "gd_module": gd_module,
        "gated_register": gated_register,
        "depths": eval_depths,
        "seq_len": seq_len,
        "byte_offset": byte_offset,
        "device": device,
        "amp_dtype": amp_dtype,
        "target_margin": float(args.target_margin),
        "target_nll_weight": float(args.target_nll_weight),
        "clean_loss_weight": float(args.clean_loss_weight),
    }
    if bool(args.skip_eval_before):
        eval_before = {"skipped": True, "reason": "--skip-eval-before"}
    else:
        eval_before = {
            "heldout": evaluate_cases(cases=eval_cases, split="heldout", **eval_kwargs),
            "heldout_world_off": evaluate_cases(cases=eval_cases, split="heldout_world_off", world_off=True, **eval_kwargs),
            "heldout_register_off": evaluate_cases(cases=eval_cases, split="heldout_register_off", register_off=True, **eval_kwargs),
        }
    history: list[dict[str, Any]] = []
    for step in range(1, int(args.steps) + 1):
        batch = batch_cases_for_step(train_cases, step=step, batch_size=int(args.batch_size))
        optimizer.zero_grad(set_to_none=True)
        gated_register.train()
        depth = int(depths[(step - 1) % len(depths)] if bool(args.single_depth_per_step) else depths[-1])
        losses: list[torch.Tensor] = []
        rows: list[dict[str, Any]] = []
        for case in batch:
            with STAGE101.make_amp_context(device, amp_dtype):
                loss, metrics = case_loss(
                    model,
                    gd_module,
                    gated_register,
                    case,
                    depth=depth,
                    seq_len=seq_len,
                    byte_offset=byte_offset,
                    device=device,
                    target_margin=float(args.target_margin),
                    target_nll_weight=float(args.target_nll_weight),
                    clean_loss_weight=float(args.clean_loss_weight),
                )
            losses.append(loss)
            rows.append(metrics)
        loss = torch.stack([item.float() for item in losses]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(args.grad_clip))
        optimizer.step()
        if step == 1 or step % int(args.log_every) == 0:
            report = case_report(rows, split="train_batch", depth=depth, world_off=False, register_off=False)
            report["step"] = int(step)
            report["loss"] = float(loss.detach().cpu().item())
            print(json.dumps(report, ensure_ascii=False), flush=True)
            history.append(report)
    eval_after = {
        "heldout": evaluate_cases(cases=eval_cases, split="heldout", **eval_kwargs),
        "heldout_world_off": evaluate_cases(cases=eval_cases, split="heldout_world_off", world_off=True, **eval_kwargs),
        "heldout_register_off": evaluate_cases(cases=eval_cases, split="heldout_register_off", register_off=True, **eval_kwargs),
    }
    accepted = bool(
        eval_after["heldout"]["accepted"]
        and not eval_after["heldout_world_off"]["accepted"]
        and not eval_after["heldout_register_off"]["accepted"]
    )
    report = {
        "decision": "stage102z_final_freeform_provenance_answer_path",
        "accepted": accepted,
        "answer_checkpoint": str(args.answer_checkpoint),
        "world_model_checkpoint": str(args.world_model_checkpoint),
        "init_gated_checkpoint": str(args.init_gated_checkpoint),
        "train_cases": int(len(train_cases)),
        "eval_cases": int(len(eval_cases)),
        "depths": depths,
        "eval_depths": eval_depths,
        "steps": int(args.steps),
        "eval_before": eval_before,
        "eval_after": eval_after,
        "plain_language_read": (
            "Final-path smoke: free-form evidence text builds context and "
            "observation cards, the graph/world register controls the same BLT "
            "LM head, and world/register ablations must break the gain."
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {
            "stage102z_final_freeform_provenance_answer_path": True,
            "gated_register_state_dict": gated_register.state_dict(),
            "args": vars(args),
            "history": history,
            "eval_after": eval_after,
        },
        out_dir / "last_final_gated_register.pt",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-checkpoint", required=True)
    parser.add_argument("--world-model-checkpoint", required=True)
    parser.add_argument("--init-gated-checkpoint", default="")
    parser.add_argument("--train-jsonl", default="data/eval/stage102c_randomized_trust_ledger_train_probe.jsonl")
    parser.add_argument("--eval-jsonl", default="data/eval/stage102c_randomized_trust_ledger_heldout_probe.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--depths", nargs="+", type=int, default=[16])
    parser.add_argument("--eval-depths", nargs="+", type=int, default=[])
    parser.add_argument("--skip-eval-before", action="store_true")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--target-margin", type=float, default=0.25)
    parser.add_argument("--target-nll-weight", type=float, default=0.01)
    parser.add_argument("--clean-loss-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=40)
    parser.add_argument("--single-depth-per-step", action="store_true")
    parser.add_argument("--max-sources", type=int, default=16)
    parser.add_argument("--graph-hidden-dim", type=int, default=0)
    parser.add_argument("--fusion-hidden-dim", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", default="bf16")
    parser.add_argument("--sampled-data", default="")
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--byte-offset", type=int, default=-1)
    return parser


if __name__ == "__main__":
    print(json.dumps(run_train(build_arg_parser().parse_args()), ensure_ascii=False, indent=2), flush=True)
