#!/usr/bin/env python3
"""Plan 4090-sized PrefixLM model configs without allocating real weights."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch


DEFAULT_CANDIDATES: tuple[dict[str, int | str], ...] = (
    {"name": "current_82m", "d_model": 384, "n_heads": 6, "n_kv_heads": 2, "d_ff": 1024},
    {"name": "probe_225m", "d_model": 768, "n_heads": 12, "n_kv_heads": 4, "d_ff": 2048},
    {"name": "probe_357m", "d_model": 1024, "n_heads": 16, "n_kv_heads": 4, "d_ff": 2816},
    {"name": "safe_695m", "d_model": 1536, "n_heads": 16, "n_kv_heads": 4, "d_ff": 4096},
    {"name": "risk_913m", "d_model": 1792, "n_heads": 16, "n_kv_heads": 4, "d_ff": 4864},
    {"name": "risk_1150m", "d_model": 2048, "n_heads": 16, "n_kv_heads": 4, "d_ff": 5504},
)


def load_trainer_module() -> Any:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "534_train_native_prefixlm_dataio.py"
    spec = importlib.util.spec_from_file_location("prefixlm_capacity_trainer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def model_parameter_count(
    trainer: Any,
    *,
    vocab_size: int,
    seq_len: int,
    d_model: int,
    n_heads: int,
    n_kv_heads: int,
    d_ff: int,
    train_think_steps: int,
) -> int:
    parser = trainer.build_arg_parser()
    args = parser.parse_args(
        [
            "--sampled-data",
            "/tmp/unused",
            "--out-dir",
            "/tmp/unused",
            "--model-vocab-size",
            str(vocab_size),
            "--seq-len",
            str(seq_len),
            "--d-model",
            str(d_model),
            "--n-heads",
            str(n_heads),
            "--n-kv-heads",
            str(n_kv_heads),
            "--d-ff",
            str(d_ff),
            "--train-think-steps",
            str(train_think_steps),
        ]
    )
    with torch.device("meta"):
        model = trainer.build_model(args, vocab_size=int(vocab_size))
    return int(sum(parameter.numel() for parameter in model.parameters()))


def gb(num_bytes: float) -> float:
    return float(num_bytes) / float(1024**3)


def estimate_memory_floor(parameter_count: int, *, optimizer: str) -> dict[str, float]:
    params_gb = gb(parameter_count * 4.0)
    grads_gb = gb(parameter_count * 4.0)
    if optimizer == "adamw":
        optim_gb = gb(parameter_count * 8.0)
    elif optimizer in {"adamw8bit", "paged_adamw8bit", "ademamix8bit", "paged_ademamix8bit"}:
        optim_gb = gb(parameter_count * 2.5)
    elif optimizer in {"galore_adamw", "galore_adamw8bit", "auto"}:
        optim_gb = gb(parameter_count * 1.5)
    else:
        optim_gb = gb(parameter_count * 3.0)
    return {
        "params_gb": params_gb,
        "grads_gb": grads_gb,
        "optimizer_state_gb_est": optim_gb,
        "floor_before_activations_gb": params_gb + grads_gb + optim_gb,
    }


def recommendation(floor_gb: float, *, target_vram_gb: float) -> str:
    if floor_gb <= target_vram_gb * 0.35:
        return "green_for_4090_smoke"
    if floor_gb <= target_vram_gb * 0.55:
        return "yellow_requires_small_microbatch"
    if floor_gb <= target_vram_gb * 0.75:
        return "red_only_with_checkpointing_and_tiny_batch"
    return "too_large_for_single_4090"


def launch_command(
    candidate: dict[str, Any],
    *,
    sampled_data: str,
    out_dir: str,
    steps: int,
    batch_size: int,
    seq_len: int,
    optimizer: str,
    lr: float,
    seed: int,
    activation_checkpointing: bool = False,
) -> str:
    parts = [
        ".venv/bin/python scripts/534_train_native_prefixlm_dataio.py",
        f"--sampled-data {sampled_data}",
        f"--out-dir {out_dir}/{candidate['name']}",
        "--device cuda",
        f"--steps {int(steps)}",
        "--checkpoint-every 1000",
        f"--batch-size {int(batch_size)}",
        f"--seq-len {int(seq_len)}",
        f"--d-model {int(candidate['d_model'])}",
        f"--n-heads {int(candidate['n_heads'])}",
        f"--n-kv-heads {int(candidate['n_kv_heads'])}",
        f"--d-ff {int(candidate['d_ff'])}",
        "--train-think-steps 2",
        "--length-bucketed-batches",
        "--trim-batch-to-max-length",
    ]
    if bool(activation_checkpointing):
        parts.append("--activation-checkpointing")
    parts.extend(
        [
            "--loss-kernel auto",
            f"--optimizer {optimizer}",
            "--amp-dtype bf16",
            "--matmul-precision high",
            f"--lr {lr:g}",
            "--lr-warmup-steps 2000",
            "--adam-beta1 0.9",
            "--adam-beta2 0.95",
            "--weight-decay 0.1",
            "--eval-every 1000",
            "--eval-max-rows 128",
            "--eval-batch-size 2",
            "--eval-max-batches 0",
            "--log-every 50",
            f"--seed {int(seed)}",
        ]
    )
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--target-vram-gb", type=float, default=24.0)
    parser.add_argument("--optimizer", default="galore_adamw8bit")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2.2e-4)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--sampled-data", default="/tmp/hrm_text_dataio_sample_stage66_dataio_preflight_20260523/sampled")
    parser.add_argument("--out-dir", default="/tmp/qtrm_eval/4090_prefixlm_capacity")
    parser.add_argument(
        "--candidate",
        default="",
        help="Optional candidate name to print only that row/command.",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    trainer = load_trainer_module()
    rows: list[dict[str, Any]] = []
    for candidate in DEFAULT_CANDIDATES:
        params = model_parameter_count(
            trainer,
            vocab_size=int(args.vocab_size),
            seq_len=int(args.seq_len),
            d_model=int(candidate["d_model"]),
            n_heads=int(candidate["n_heads"]),
            n_kv_heads=int(candidate["n_kv_heads"]),
            d_ff=int(candidate["d_ff"]),
            train_think_steps=2,
        )
        memory = estimate_memory_floor(params, optimizer=str(args.optimizer))
        row = {
            **candidate,
            "parameter_count": params,
            "parameter_count_b": round(params / 1e9, 3),
            **memory,
            "recommendation": recommendation(
                float(memory["floor_before_activations_gb"]),
                target_vram_gb=float(args.target_vram_gb),
            ),
            "launch_command": launch_command(
                candidate,
                sampled_data=str(args.sampled_data),
                out_dir=str(args.out_dir),
                steps=int(args.steps),
                batch_size=int(args.batch_size),
                seq_len=int(args.seq_len),
                optimizer=str(args.optimizer),
                lr=float(args.lr),
                seed=int(args.seed),
            ),
        }
        rows.append(row)

    if str(args.candidate):
        wanted = str(args.candidate)
        known = ", ".join(str(row["name"]) for row in rows)
        rows = [row for row in rows if str(row["name"]) == wanted]
        if not rows:
            raise SystemExit(f"unknown candidate {wanted!r}; choose one of: {known}")

    report = {
        "target": "single_4090_prefixlm_capacity_plan",
        "target_vram_gb": float(args.target_vram_gb),
        "optimizer": str(args.optimizer),
        "assumptions": {
            "parameter_dtype": "fp32",
            "gradient_dtype": "fp32",
            "activation_memory": "not included; controlled by batch, seq, checkpointing, trim, and bucket flags",
            "plain_language_read": (
                "This is the desk floor: model weights, gradients, and optimizer ledger. "
                "Activations are the loose papers still created during the forward pass."
            ),
        },
        "candidates": rows,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if str(args.output):
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
