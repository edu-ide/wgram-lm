#!/usr/bin/env python3
"""Create a tiny sampled PrefixLM shard without copying token payloads.

The official sampler concatenates token arrays into a new tokens.npy. That is
correct for a real book, but too slow when the GPU is waiting. This script makes
a one-task emergency booklet:

  sampled/tokens.npy is a hardlink or symlink to one tokenized task's tokens.npy
  sampled/epoch_0 and sampled/epoch_1 contain shuffled row indices
  sampled/metadata.json follows the Data-IO contract

Plain-language version:
  do not photocopy the book; point to one already printed chapter and make two
  small tables of contents.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


INDEX_NAMES = ("inst_start", "inst_len", "resp_start", "resp_len")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=8192)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--context-size", type=int, default=1025)
    parser.add_argument("--min-resp-length", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9300)
    parser.add_argument("--task-name", default="")
    return parser.parse_args()


def task_is_complete(task_dir: Path) -> bool:
    required = [task_dir / "tokens.npy", task_dir / "metadata.json"]
    required.extend(task_dir / f"{name}.npy" for name in INDEX_NAMES)
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def pick_task(tokenized_root: Path, task_name: str) -> Path:
    if task_name:
        task_dir = tokenized_root / task_name
        if not task_is_complete(task_dir):
            raise SystemExit(f"requested task is incomplete: {task_dir}")
        return task_dir
    for task_dir in sorted(path for path in tokenized_root.iterdir() if path.is_dir()):
        if task_is_complete(task_dir):
            return task_dir
    raise SystemExit(f"no complete tokenized task under {tokenized_root}")


def link_tokens(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(src, dst)


def main() -> None:
    args = parse_args()
    tokenized_root = Path(args.tokenized_root)
    output_dir = Path(args.output_dir)
    task_dir = pick_task(tokenized_root, args.task_name)

    with (tokenized_root / "tokenizer_info.json").open("r", encoding="utf-8") as handle:
        tokenizer_info = json.load(handle)

    arrays = {name: np.load(task_dir / f"{name}.npy") for name in INDEX_NAMES}
    inst_len = arrays["inst_len"].astype(np.int64)
    resp_len = arrays["resp_len"].astype(np.int64)
    valid = np.nonzero(
        (inst_len >= 1)
        & (resp_len >= int(args.min_resp_length))
        & ((inst_len + resp_len - 1) <= int(args.seq_len))
    )[0]
    if valid.shape[0] == 0:
        raise SystemExit(f"no valid rows in {task_dir} for seq_len={args.seq_len}")
    valid = valid[: int(args.max_rows)]

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    link_tokens(task_dir / "tokens.npy", output_dir / "tokens.npy")

    rng = np.random.default_rng(int(args.seed))
    for epoch in range(int(args.epochs)):
        epoch_dir = output_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        perm = rng.permutation(valid)
        for name in INDEX_NAMES:
            np.save(epoch_dir / f"{name}.npy", arrays[name][perm])

    total_length = int(np.sum(arrays["inst_len"][valid]) + np.sum(arrays["resp_len"][valid]))
    metadata = {
        "tokenizer_info": tokenizer_info,
        "vocab_size": tokenizer_info.get("vocab_size"),
        "max_seq_len": int(args.context_size),
        "total_length": total_length,
        "hardlink_micro_source_task": task_dir.name,
        "rows": int(valid.shape[0]),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle)

    print(f"HARDLINK_MICRO_READY:{output_dir}")
    print(f"SOURCE_TASK:{task_dir.name}")
    print(f"ROWS:{valid.shape[0]}")
    print(f"TOKENS:{output_dir / 'tokens.npy'}")


if __name__ == "__main__":
    main()
