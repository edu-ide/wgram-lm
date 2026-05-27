#!/usr/bin/env python3
"""
s2_collect_hybrid_data.py

Helper script for S2 data collection on the hybrid side.
Launches multiple seeds + key ablations using the improved prototype.

Usage examples:
  python scripts/s2_collect_hybrid_data.py --seeds 3 --steps 100
  python scripts/s2_collect_hybrid_data.py --ablation gold_off --steps 100
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_hybrid(steps: int, extra_flags: str = ""):
    cmd = f"PYTHONPATH=. .venv/bin/python scripts/train_556_on_parallel_hybrid_minimal.py --steps {steps} --batch 2 --d_model 96 --log_every 20 --enable_stochastic_breadth {extra_flags}"
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
    return result.stdout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--ablation", type=str, default=None, choices=["stoch_zero", "gold_off", "protection_off"])
    args = parser.parse_args()

    print("=== S2 Hybrid Data Collection ===")

    if args.ablation:
        if args.ablation == "stoch_zero":
            run_hybrid(args.steps, "--stochastic_ablation_zero true")
        else:
            print(f"Ablation {args.ablation} requires temporary script modification for now.")
    else:
        for i in range(args.seeds):
            print(f"\n--- Seed {i+1}/{args.seeds} ---")
            run_hybrid(args.steps)

    print("\nCollection complete. Use the output for S2 comparison script.")

if __name__ == "__main__":
    main()