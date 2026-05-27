#!/usr/bin/env python3
"""
Master Ablation Runner for the current I→G→A plan (Phase 0-2)

Usage examples:
  # Run enlarged Phase 1
  python scripts/master_ablation_runner.py --phase1 --seeds 8

  # Run improved 642 proxy (B)
  python scripts/master_ablation_runner.py --phase0-642

  # Run Phase 2 composition
  python scripts/master_ablation_runner.py --phase2 --seeds 3

  # Run everything in order (recommended)
  python scripts/master_ablation_runner.py --all
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

def run_cmd(script: str, args: list):
    cmd = [str(VENV_PYTHON), str(ROOT / "scripts" / script)] + args
    print(f"\n>>> Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--phase1", action="store_true", help="Run enlarged Phase 1")
    parser.add_argument("--phase0-642", action="store_true", help="Run improved 642 injection experiment")
    parser.add_argument("--phase2", action="store_true", help="Run Phase 2 composition")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--config", type=str, default=None, help="Future: path to experiment config yaml")
    args = parser.parse_args()

    print("=== Master Ablation Runner (순서대로 1→2→3→4) ===")

    if args.all or args.phase1:
        run_cmd("diag_phase1_multi_ablation.py", ["--seeds", str(args.seeds), "--batch", "8", "--seq", "16", "--d", "256"])

    if args.all or args.phase0_642:
        run_cmd("phase0_642_injection_experiment.py", ["--steps", "30", "--binding-weight", "0.15"])

    if args.all or args.phase2:
        run_cmd("diag_phase2_full_composition.py", ["--seeds", str(args.seeds), "--steps", "20"])

    print("\n=== Master Ablation Run Complete ===")
    print("All results should be copied to the wiki Ablation Milestone section.")

if __name__ == "__main__":
    main()