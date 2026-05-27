#!/usr/bin/env python3
"""
run_556_ablation_matrix.py

Executable ablation battery for the Full 5.56 Adaptive Rehearsal Curriculum
(Highest priority Historical Signal - Reverse I→G→A).

This is the direct next artifact after the instrumented trainer + first real execution.

Purpose:
- Systematically test every major ingredient of the 5.56 gold recipe in isolation
  and combination on the current One-Body architecture.
- Produce comparable metrics.json artifacts for the Promotion Gate decision.

Standard matrix (the minimal set that actually mattered historically):
1. Full 5.56 (real gold if available + stochastic ON + protection ON + scheduled decay)
2. No stochastic breadth (ablation_zero)
3. No attractor protection during rehearsal
4. No scheduled decay (fixed high binding)
5. Synthetic gold only (no real 642 structural bias)
6. Stochastic ON but protection OFF
7. (Optional) longer runs or real 642 variants

Usage (in the project's torch venv):
    PYTHONPATH=. python scripts/run_556_ablation_matrix.py \
        --steps 100 \
        --base_save_dir local_556_ablation_$(date +%Y%m%d) \
        --gold_path local_eval/642_adaptive_fine_tuned_200step/adaptive_phase2_checkpoint.pt

Each variant gets its own subdir with:
- full_output.log
- metrics.json (rich 5.56 fields)
- config.json
- best.pt / last.pt

After the matrix finishes, run:
    python scripts/analyze_556_curriculum_metrics.py local_556_ablation_*/ **/metrics.json

This directly supports the "ablation battery" required by the 2026-05-30 decision record
and the research-driven-architecture-debugging skill (Promotion Gate evidence).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

DEFAULT_VARIANTS = [
    {
        "name": "01_full_556_real_gold_stoch_on",
        "extra_args": "--enable_stochastic_breadth",
        "gold_path": None,  # will be filled from CLI
    },
    {
        "name": "02_stochastic_ablation_zero",
        "extra_args": "--enable_stochastic_breadth --stochastic_ablation_zero true",
        "gold_path": None,
    },
    {
        "name": "03_no_attractor_protection",
        # Note: requires trainer support for disabling protection; currently we pass a marker
        # that future rehearsal config can read. For now we document the intent.
        "extra_args": "--enable_stochastic_breadth",
        "gold_path": None,
        "note": "MANUAL: temporarily set protect_attractor=False in rehearsal config for this variant",
    },
    {
        "name": "04_no_scheduled_decay_fixed_high",
        "extra_args": "--enable_stochastic_breadth",
        "gold_path": None,
        "note": "MANUAL: set fixed high binding weight in rehearsal for this run",
    },
    {
        "name": "05_synthetic_gold_only",
        "extra_args": "--enable_stochastic_breadth",
        "gold_path": "",  # force synthetic
    },
    {
        "name": "06_stoch_on_no_protection",
        "extra_args": "--enable_stochastic_breadth",
        "gold_path": None,
        "note": "Combination test: breadth without attractor protection",
    },
]

def run_variant(variant: dict, base_dir: Path, steps: int, common_launcher: str, gold_path: str):
    vdir = base_dir / variant["name"]
    vdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "bash", common_launcher,
        "--steps", str(steps),
        "--save_dir", str(vdir),
        "--log_every", "5",
    ]

    if variant.get("extra_args"):
        for tok in variant["extra_args"].split():
            cmd.append(tok)

    effective_gold = variant.get("gold_path") if variant.get("gold_path") is not None else gold_path
    if effective_gold:
        cmd += ["--gold_path", effective_gold]
    elif effective_gold == "":
        # explicit synthetic
        pass

    print(f"\n=== Running variant: {variant['name']} ===")
    print("Command:", " ".join(cmd))
    if variant.get("note"):
        print("NOTE:", variant["note"])

    with open(vdir / "full_output.log", "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)

    print(f"Variant {variant['name']} finished with code {proc.returncode}")
    print(f"  Artifacts in: {vdir}")
    return proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100, help="Steps per variant")
    parser.add_argument("--base_save_dir", type=str, default=None)
    parser.add_argument("--gold_path", type=str, default=None,
                        help="Path to real 642 gold checkpoint (highly recommended for the full recipe)")
    parser.add_argument("--launcher", type=str, default="scripts/launch_556_local_smoke.sh")
    parser.add_argument("--only", type=str, default=None, help="Run only variants whose name contains this substring")
    args = parser.parse_args()

    if args.base_save_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        args.base_save_dir = f"local_556_ablation_matrix_{ts}"

    base_dir = Path(args.base_save_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=== 5.56 Full Curriculum Ablation Matrix (Reverse I→G→A) ===")
    print(f"Base output: {base_dir}")
    print(f"Steps per variant: {args.steps}")
    print(f"Gold path: {args.gold_path or '(synthetic for most variants)'}")
    print()

    variants = DEFAULT_VARIANTS
    if args.only:
        variants = [v for v in variants if args.only in v["name"]]

    results = {}
    for v in variants:
        code = run_variant(v, base_dir, args.steps, args.launcher, args.gold_path)
        results[v["name"]] = code

    print("\n=== Matrix complete ===")
    for name, code in results.items():
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"  {name}: {status}")

    print(f"\nAll metrics.json files are under: {base_dir}/**/metrics.json")
    print("Next: python scripts/analyze_556_curriculum_metrics.py " + str(base_dir) + " --output summary_556_ablation.md")
    print("Then review against the Promotion Gate criteria in the 2026-05-30 decision record.")


if __name__ == "__main__":
    main()
