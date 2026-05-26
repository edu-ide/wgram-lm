#!/usr/bin/env python3
"""Train Stage102E world-model-gated answer register.

Stage102D learns a label-free data-sense signal.  Stage102E routes that signal
into the same BLT LM head used by Stage102C:

  clean evidence      -> graph register can speak yes
  corrupted evidence  -> world-model residual should brake the answer to no

The base BLT, provenance graph reasoner, and provenance world model are frozen
by default.  Only the small fusion adapter is trained.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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


STAGE102B = load_module(
    ROOT / "scripts" / "605_train_stage102b_provenance_graph_reasoner.py",
    "stage102e_stage102b_provenance_graph_reasoner",
)
STAGE102D = load_module(
    ROOT / "scripts" / "607_train_stage102d_provenance_data_world_model.py",
    "stage102e_stage102d_provenance_data_world_model",
)
STAGE101 = STAGE102B.STAGE101
YES = STAGE102B.YES
NO = STAGE102B.NO


def answerable_side(row: dict[str, Any]) -> str:
    if str(row.get("original_answer")) == YES:
        return "original"
    if str(row.get("counterfactual_answer")) == YES:
        return "counterfactual"
    raise ValueError(f"row has no yes side: {row.get('id')}")


def build_world_gated_answer_cases(row: dict[str, Any]) -> list[dict[str, Any]]:
    side = answerable_side(row)
    prompt = str(row[f"{side}_prompt"])
    graph_features = STAGE102B.build_graph_features(row, side)
    world_examples = STAGE102D.build_world_model_examples(row, side)
    cases: list[dict[str, Any]] = []
    for example in world_examples:
        corruption = str(example["corruption"])
        is_clean = corruption == "clean"
        answer = YES if is_clean else NO
        negative = NO if is_clean else YES
        cases.append(
            {
                "id": row.get("id"),
                "side": side,
                "prompt": prompt,
                "corruption": corruption,
                "graph_features": dict(graph_features),
                "world_example": dict(example),
                "answer": answer,
                "negative_answer": negative,
                "is_clean": bool(is_clean),
            }
        )
    return cases


def build_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in rows:
        cases.extend(build_world_gated_answer_cases(row))
    if not cases:
        raise ValueError("no Stage102E cases were built")
    return cases


class WorldModelGatedAnswerRegister(nn.Module):
    """Fuse a Stage102C graph register with Stage102D data-world residuals."""

    def __init__(
        self,
        *,
        d_model: int,
        graph_reasoner: nn.Module,
        world_model: nn.Module,
        world_d_model: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        width = int(hidden_dim or max(int(d_model), int(world_d_model) * 2))
        self.graph_reasoner = graph_reasoner
        self.world_model = world_model
        self.world_to_delta = nn.Sequential(
            nn.Linear(int(world_d_model) + 1, width),
            nn.SiLU(),
            nn.Linear(width, int(d_model)),
        )
        self.world_gate = nn.Sequential(
            nn.Linear(int(world_d_model) + 1, width),
            nn.SiLU(),
            nn.Linear(width, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        graph_features: dict[str, Any],
        world_example: dict[str, Any],
        *,
        device: torch.device,
        world_off: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        graph_register, graph_metrics = self.graph_reasoner([graph_features], device=device)
        if bool(world_off):
            return graph_register, {
                "world_energy": 0.0,
                "world_gate": 0.0,
                **{f"graph_{key}": value for key, value in graph_metrics.items()},
            }
        energy, latent = self.world_model([world_example], device=device)
        signal = torch.cat([latent.float(), energy.float().unsqueeze(1)], dim=1)
        gate = self.world_gate(signal).to(dtype=graph_register.dtype)
        delta = self.world_to_delta(signal).to(dtype=graph_register.dtype)
        register = graph_register + gate * delta
        return register, {
            "world_energy": float(energy.detach().float().mean().cpu().item()),
            "world_gate": float(gate.detach().float().mean().cpu().item()),
            **{f"graph_{key}": value for key, value in graph_metrics.items()},
        }


def load_world_model_checkpoint(path: Path, *, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"bad Stage102D checkpoint: {path}")
    raw_args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
    d_model = int(raw_args.get("d_model", 32))
    max_sources = int(raw_args.get("max_sources", 16))
    hidden_dim = int(raw_args.get("hidden_dim") or max(32, d_model * 2))
    model = STAGE102D.ProvenanceDataWorldModel(
        d_model=d_model,
        max_sources=max_sources,
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {"d_model": d_model, "max_sources": max_sources, "hidden_dim": hidden_dim}


def choice_logprob_with_register(
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
    return STAGE102B.choice_mean_logprob_with_register(
        model,
        gd_module,
        prompt=str(prompt),
        answer=str(answer),
        register=register,
        seq_len=int(seq_len),
        byte_offset=int(byte_offset),
        device=device,
        think_steps=int(think_steps),
    )


def case_loss(
    model: torch.nn.Module,
    gd_module: Any,
    gated_register: WorldModelGatedAnswerRegister,
    case: dict[str, Any],
    *,
    depth: int,
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    target_margin: float,
    target_nll_weight: float,
    world_off: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    register, register_metrics = gated_register(
        case["graph_features"],
        case["world_example"],
        device=device,
        world_off=bool(world_off),
    )
    target_mean, _target_tokens = choice_logprob_with_register(
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
    negative_mean, _negative_tokens = choice_logprob_with_register(
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
    return loss, {
        "id": case.get("id"),
        "side": case.get("side"),
        "corruption": case.get("corruption"),
        "is_clean": bool(case.get("is_clean")),
        "depth": int(depth),
        "loss": float(loss.detach().cpu().item()),
        "margin": float(margin.detach().cpu().item()),
        "target_nll": float(target_nll.detach().cpu().item()),
        "correct": bool(float(margin.detach().cpu().item()) > 0.0),
        **register_metrics,
    }


def batch_cases_for_step(cases: list[dict[str, Any]], *, step: int, batch_size: int) -> list[dict[str, Any]]:
    if not cases:
        raise ValueError("cases must not be empty")
    start = (int(step) - 1) % len(cases)
    return [cases[(start + offset) % len(cases)] for offset in range(max(1, int(batch_size)))]


def case_report(rows: list[dict[str, Any]], *, split: str, depth: int, world_off: bool) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows required")
    by_corruption: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_corruption.setdefault(str(row["corruption"]), []).append(row)
    corrupt_rows = [row for row in rows if str(row["corruption"]) != "clean"]
    clean_rows = [row for row in rows if str(row["corruption"]) == "clean"]

    def accuracy(items: list[dict[str, Any]]) -> float:
        if not items:
            return 0.0
        return sum(1 for item in items if bool(item["correct"])) / float(len(items))

    return {
        "split": split,
        "depth": int(depth),
        "world_off": bool(world_off),
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
    gated_register: WorldModelGatedAnswerRegister,
    cases: list[dict[str, Any]],
    split: str,
    depths: list[int],
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    target_margin: float,
    target_nll_weight: float,
    world_off: bool = False,
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
                    world_off=bool(world_off),
                )
            metrics["split"] = split
            rows.append(metrics)
            details.append(metrics)
        by_depth.append(case_report(rows, split=split, depth=int(depth), world_off=world_off))
    return {
        "split": split,
        "world_off": bool(world_off),
        "depths": by_depth,
        "accepted": bool(by_depth and by_depth[-1]["accepted"]),
        "rows": details,
    }


def save_checkpoint(
    path: Path,
    *,
    gated_register: WorldModelGatedAnswerRegister,
    optimizer: torch.optim.Optimizer,
    args_payload: dict[str, Any],
    history: list[dict[str, Any]],
    eval_before: dict[str, Any],
    eval_after: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage102e_world_model_gated_answer_register": True,
        "gated_register_state_dict": gated_register.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": args_payload,
        "loss_history": history,
        "eval_before": eval_before,
        "eval_after": eval_after,
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
    device = torch.device(str(args.device))
    amp_dtype = STAGE101.resolve_amp_dtype(str(args.amp_dtype))
    train_rows = STAGE101.load_jsonl(Path(args.train_jsonl))
    eval_rows = STAGE101.load_jsonl(Path(args.eval_jsonl))
    if int(args.max_train_rows) > 0:
        train_rows = train_rows[: int(args.max_train_rows)]
    if int(args.max_eval_rows) > 0:
        eval_rows = eval_rows[: int(args.max_eval_rows)]
    train_cases = build_cases(train_rows)
    eval_cases = build_cases(eval_rows)

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

    world_model, world_info = load_world_model_checkpoint(Path(args.world_model_checkpoint), device=device)
    gated_register = WorldModelGatedAnswerRegister(
        d_model=int(ckpt_args.d_model),
        graph_reasoner=graph_reasoner,
        world_model=world_model,
        world_d_model=int(world_info["d_model"]),
        hidden_dim=int(args.fusion_hidden_dim or ckpt_args.d_model),
    ).to(device)
    trainable = [
        parameter
        for name, parameter in gated_register.named_parameters()
        if name.startswith("world_to_delta.") or name.startswith("world_gate.")
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    tokenizer_info = dict(loaded.get("dataset_summary", {}).get("tokenizer_info") or {})
    byte_offset = int(args.byte_offset if int(args.byte_offset) >= 0 else tokenizer_info.get("byte_offset", 2))
    seq_len = int(args.seq_len or ckpt_args.seq_len)
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
    }
    if bool(args.skip_eval_before):
        eval_before = {
            "skipped": True,
            "reason": "--skip-eval-before",
            "train_cases": int(len(train_cases)),
            "eval_cases": int(len(eval_cases)),
        }
    else:
        eval_before = {
            "train": evaluate_cases(cases=train_cases, split="train", **eval_kwargs),
            "heldout": evaluate_cases(cases=eval_cases, split="heldout", **eval_kwargs),
            "heldout_world_off": evaluate_cases(
                cases=eval_cases,
                split="heldout_world_off",
                world_off=True,
                **eval_kwargs,
            ),
        }

    history: list[dict[str, Any]] = []
    for step in range(1, int(args.steps) + 1):
        batch = batch_cases_for_step(train_cases, step=step, batch_size=int(args.batch_size))
        optimizer.zero_grad(set_to_none=True)
        gated_register.train()
        losses: list[torch.Tensor] = []
        rows: list[dict[str, Any]] = []
        depth = int(depths[(step - 1) % len(depths)] if bool(args.single_depth_per_step) else depths[-1])
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
                )
            losses.append(loss)
            rows.append(metrics)
        loss = torch.stack([item.float() for item in losses]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(args.grad_clip))
        optimizer.step()
        if step == 1 or step % int(args.log_every) == 0:
            report = case_report(rows, split="train_batch", depth=depth, world_off=False)
            report["step"] = int(step)
            report["loss"] = float(loss.detach().cpu().item())
            print(json.dumps(report, ensure_ascii=False), flush=True)
            history.append(report)

    eval_after = {
        "train": evaluate_cases(cases=train_cases, split="train", **eval_kwargs),
        "heldout": evaluate_cases(cases=eval_cases, split="heldout", **eval_kwargs),
        "heldout_world_off": evaluate_cases(
            cases=eval_cases,
            split="heldout_world_off",
            world_off=True,
            **eval_kwargs,
        ),
    }
    accepted = bool(eval_after["heldout"]["accepted"])
    report = {
        "decision": "stage102e_world_model_gated_answer_register",
        "accepted": accepted,
        "answer_checkpoint": str(args.answer_checkpoint),
        "world_model_checkpoint": str(args.world_model_checkpoint),
        "checkpoint_out": str(out_dir / "last_gated_register.pt"),
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl),
        "train_cases": int(len(train_cases)),
        "eval_cases": int(len(eval_cases)),
        "depths": depths,
        "eval_depths": eval_depths,
        "steps": int(args.steps),
        "eval_before": eval_before,
        "eval_after": eval_after,
        "plain_language_read": (
            "Stage102E connects the label-free data-sense signal to the same "
            "LM-head answer mouth. Clean evidence may speak; broken evidence "
            "should be braked to no."
        ),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_checkpoint(
        out_dir / "last_gated_register.pt",
        gated_register=gated_register,
        optimizer=optimizer,
        args_payload={
            **vars(args),
            "answer_d_model": int(ckpt_args.d_model),
            "world_d_model": int(world_info["d_model"]),
        },
        history=history,
        eval_before=eval_before,
        eval_after=eval_after,
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-checkpoint", required=True)
    parser.add_argument("--world-model-checkpoint", required=True)
    parser.add_argument("--train-jsonl", default="data/eval/stage102c_randomized_trust_ledger_train_probe.jsonl")
    parser.add_argument("--eval-jsonl", default="data/eval/stage102c_randomized_trust_ledger_heldout_probe.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--eval-depths", nargs="+", type=int, default=[])
    parser.add_argument("--skip-eval-before", action="store_true")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--target-margin", type=float, default=0.25)
    parser.add_argument("--target-nll-weight", type=float, default=0.01)
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


def main() -> None:
    report = run_train(build_arg_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
