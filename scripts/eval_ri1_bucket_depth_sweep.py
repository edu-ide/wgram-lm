#!/usr/bin/env python3
"""
Simple RI-1 Causal Evaluation: Memory Bucket Depth Sweep

Runs the model at different effective depths (think_steps) on pure reasoning + memory cases,
with memory ON vs OFF (by controlling brain_triple_memory attachment).

This is the key test for whether we have achieved "deeper is better" + "memory helps more at higher depth".

Usage:
    PYTHONPATH=src:. python scripts/eval_ri1_bucket_depth_sweep.py \
        --ckpt checkpoints/ri1_substrate_minimal_.../ri1_substrate_stepXXX.pt \
        --depths 1,2,4,8,12 \
        --cases 32
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.abspath('.'))

# Reuse the existing evaluation harness as much as possible
# For now this is a minimal skeleton that can be expanded.

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--depths", type=str, default="1,2,4,8,12")
    p.add_argument("--cases", type=int, default=32)
    p.add_argument("--memory_on", action="store_true")
    args = p.parse_args()

    depths = [int(x) for x in args.depths.split(",")]
    print(f"RI-1 Bucket Depth Sweep on {args.ckpt}")
    print(f"Depths: {depths}")
    print(f"Memory ON: {args.memory_on}")
    print(f"Cases: {args.cases}")

    # TODO: Load model, attach light attractor if needed, run at each depth with/without memory,
    # report accuracy delta (memory ON - OFF) as function of depth.
    # This is the direct test of the RI-1 thesis.

    print("\n[TODO] Full bucket sweep implementation.")
    print("When the minimal long runner finally produces a real checkpoint that survives 100+ steps,")
    print("run this script on it with the light settings to measure the causal gain.")


if __name__ == "__main__":
    main()