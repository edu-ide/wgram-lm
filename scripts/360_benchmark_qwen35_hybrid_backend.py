#!/usr/bin/env python3
"""Benchmark Qwen3.5-style hybrid backend kernels for QTRM-native.

The goal is not model quality. It isolates whether the 3:1
GatedDelta/attention path is blocked by kernel availability, first-step compile
time, or slow PyTorch fallback execution.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from qtrm_mm.blocks import QTRMBlockStack
from qtrm_mm.config import QTRMConfig
from qtrm_mm.mixers import FLADeltaMixer, TorchGatedDeltaMixer, build_delta_mixer
from qtrm_mm.training_optimizers import build_memory_efficient_optimizer


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_ms(device: torch.device, fn) -> tuple[float, object]:
    sync(device)
    start = time.perf_counter()
    value = fn()
    sync(device)
    return (time.perf_counter() - start) * 1000.0, value


def backend_summary(module: torch.nn.Module) -> dict[str, object]:
    fla_total = 0
    fla_official = 0
    torch_delta = 0
    for child in module.modules():
        if isinstance(child, FLADeltaMixer):
            fla_total += 1
            fla_official += int(bool(child.is_official_backend))
        elif isinstance(child, TorchGatedDeltaMixer):
            torch_delta += 1
    return {
        "fla_delta_mixers": fla_total,
        "official_fla_delta_mixers": fla_official,
        "torch_delta_mixers": torch_delta,
        "all_fla_mixers_official": bool(fla_total > 0 and fla_total == fla_official),
    }


def make_cfg(args: argparse.Namespace, *, backend: str) -> QTRMConfig:
    return QTRMConfig(
        vocab_size=1024,
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_kv_heads=int(args.n_kv_heads),
        d_ff=int(args.d_ff),
        max_seq_len=int(args.seq_len),
        dropout=0.0,
        rope_theta=float(args.rope_theta),
        attn_every=int(args.attn_every),
        delta_backend=str(backend),
        delta_head_dim=int(args.delta_head_dim) if int(args.delta_head_dim) > 0 else None,
        delta_num_v_heads=(
            int(args.delta_num_v_heads) if int(args.delta_num_v_heads) > 0 else None
        ),
        delta_expand_v=float(args.delta_expand_v),
        delta_mode=str(args.delta_mode),
        delta_use_short_conv=not bool(args.delta_no_short_conv),
        delta_conv_size=int(args.delta_conv_size),
        delta_norm_eps=float(args.delta_norm_eps),
        attention_backend=str(args.attention_backend),
        strict_backends=bool(args.strict_backends),
    )


def build_case(args: argparse.Namespace, *, case: str, backend: str) -> torch.nn.Module:
    if case == "delta_only":
        return build_delta_mixer(
            d_model=int(args.d_model),
            n_heads=int(args.n_heads),
            backend=str(backend),
            strict=bool(args.strict_backends),
            dropout=0.0,
            head_dim=(
                int(args.delta_head_dim)
                if int(args.delta_head_dim) > 0
                else int(args.d_model) // max(1, int(args.n_heads))
            ),
            num_v_heads=(
                int(args.delta_num_v_heads)
                if int(args.delta_num_v_heads) > 0
                else int(args.n_heads)
            ),
            expand_v=float(args.delta_expand_v),
            mode=str(args.delta_mode),
            use_short_conv=not bool(args.delta_no_short_conv),
            conv_size=int(args.delta_conv_size),
            norm_eps=float(args.delta_norm_eps),
        )
    cfg = make_cfg(args, backend=str(backend))
    if case == "qtrm_hybrid_3to1":
        return QTRMBlockStack(cfg, n_layers=4, causal=True, attn_every=4)
    if case == "attention_only":
        return QTRMBlockStack(cfg, n_layers=1, causal=True, attn_every=1)
    raise ValueError(f"unknown benchmark case: {case}")


def train_step(
    module: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    def run_forward_backward() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        y = module(x)
        target = torch.zeros_like(y)
        loss = F.mse_loss(y, target)
        loss.backward()
        return loss.detach()

    step_ms, loss = elapsed_ms(device, run_forward_backward)
    opt_ms, _ = elapsed_ms(device, optimizer.step)
    return step_ms, opt_ms


def benchmark_one(args: argparse.Namespace, *, case: str, backend: str) -> dict[str, object]:
    device = torch.device(str(args.device))
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    build_ms, module = elapsed_ms(device, lambda: build_case(args, case=case, backend=backend))
    module = module.to(device)
    module.train()
    x = torch.randn(
        int(args.batch_size),
        int(args.seq_len),
        int(args.d_model),
        dtype=torch.float32,
        device=device,
    )
    summary = backend_summary(module)
    optimizer, optimizer_report = build_memory_efficient_optimizer(
        module,
        optimizer_name=str(args.optimizer),
        lr=float(args.lr),
        weight_decay=0.0,
        device=device,
        galore_rank=int(args.galore_rank),
        galore_update_proj_gap=int(args.galore_update_proj_gap),
        galore_scale=float(args.galore_scale),
        galore_proj_type=str(args.galore_proj_type),
        galore_min_dim=int(args.galore_min_dim),
        galore_include_embeddings=False,
    )
    forward_ms, _ = elapsed_ms(device, lambda: module(x))
    first_train_ms, first_opt_ms = train_step(module, x, device, optimizer)
    repeat_train_ms: list[float] = []
    repeat_opt_ms: list[float] = []
    for _ in range(max(0, int(args.repeat_steps))):
        step_ms, opt_ms = train_step(module, x, device, optimizer)
        repeat_train_ms.append(float(step_ms))
        repeat_opt_ms.append(float(opt_ms))
    peak_mib = None
    if device.type == "cuda":
        peak_mib = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    return {
        "case": case,
        "backend": backend,
        "build_ms": float(build_ms),
        "forward_ms": float(forward_ms),
        "first_forward_backward_ms": float(first_train_ms),
        "first_optimizer_ms": float(first_opt_ms),
        "repeat_forward_backward_ms": repeat_train_ms,
        "repeat_optimizer_ms": repeat_opt_ms,
        "peak_allocated_mib": peak_mib,
        "backend_summary": summary,
        "optimizer_report": optimizer_report,
    }


def run_benchmarks(args: argparse.Namespace) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for case in str(args.cases).split(","):
        case = case.strip()
        if not case:
            continue
        for backend in str(args.backends).split(","):
            backend = backend.strip()
            if not backend:
                continue
            try:
                rows.append(benchmark_one(args, case=case, backend=backend))
            except Exception as exc:
                rows.append(
                    {
                        "case": case,
                        "backend": backend,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    report = {
        "status": "complete",
        "device": str(args.device),
        "settings": vars(args),
        "results": rows,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="local_eval/qtrm_native_qwen35_backend_bench")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cases", default="delta_only,qtrm_hybrid_3to1,attention_only")
    parser.add_argument("--backends", default="torch_gated_delta,fla_gated_delta,fla_kda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=384)
    parser.add_argument("--attn-every", type=int, default=4)
    parser.add_argument("--rope-theta", type=float, default=10000000.0)
    parser.add_argument("--delta-head-dim", type=int, default=0)
    parser.add_argument("--delta-num-v-heads", type=int, default=0)
    parser.add_argument("--delta-expand-v", type=float, default=1.0)
    parser.add_argument("--delta-mode", default="chunk")
    parser.add_argument("--delta-no-short-conv", action="store_true")
    parser.add_argument("--delta-conv-size", type=int, default=4)
    parser.add_argument("--delta-norm-eps", type=float, default=1e-6)
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--strict-backends", action="store_true")
    parser.add_argument("--optimizer", default="adamw8bit")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--galore-rank", type=int, default=16)
    parser.add_argument("--galore-update-proj-gap", type=int, default=400)
    parser.add_argument("--galore-scale", type=float, default=0.25)
    parser.add_argument("--galore-proj-type", default="std")
    parser.add_argument("--galore-min-dim", type=int, default=128)
    parser.add_argument("--repeat-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=360)
    return parser


def main() -> None:
    report = run_benchmarks(build_arg_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
