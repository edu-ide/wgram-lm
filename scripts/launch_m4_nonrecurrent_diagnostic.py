#!/usr/bin/env python3
"""
M4 Smallest Diagnostic: Non-Recurrent Generative Thinking Phase

This is the minimal diagnostic recommended in the Substrate Doubt Synthesis
and now officially triggered as part of M4 Global Decision Gate (2026-06).

Purpose:
- Test whether the core problem is "tight micro-step recurrent latent thinking + memory participation during thinking".
- Replace the recurrent state evolution *only during the thinking phase* with a non-recurrent process (parallel candidates, optimization, search, etc.).
- Memory participates only at explicit boundaries or as downstream effect.
- Keep the overall One-Body contract and use the existing v2 192 measurement harness.

This is intentionally the smallest possible experiment that steps structurally outside the current substrate family.

Usage (example):
    python scripts/launch_m4_nonrecurrent_diagnostic.py \
        --steps 100 --d_model 128 --num_candidates 4 --noise 0.15

See:
- docs/wiki/decisions/2026-06-missing-inductive-biases-restoration-roadmap.md (M4 section)
- docs/wiki/decisions/ri4_substrate_doubt_synthesis_2026-06.md
"""

import argparse

def main():
    parser = argparse.ArgumentParser(description="M4: Smallest Non-Recurrent Generative Thinking Diagnostic")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_candidates", type=int, default=4, help="Number of parallel candidates in non-recurrent thinking")
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--gold_path", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("M4 DIAGNOSTIC: Non-Recurrent Generative Thinking Phase")
    print("=" * 70)
    print(f"Steps: {args.steps}")
    print(f"d_model: {args.d_model}")
    print(f"Num candidates (non-recurrent): {args.num_candidates}")
    print(f"Noise scale: {args.noise}")
    print(f"Gold path: {args.gold_path}")
    print()
    print("This is the smallest structural step outside 'tight recurrent + memory during thinking'.")
    print("Run this, measure with v2 192 harness, and compare to equivalent recurrent baseline.")
    print()
    print("If this produces different (better) behavior on carry_rate or selectivity,")
    print("it supports the substrate doubt hypothesis and justifies a higher-level jump.")
    print("=" * 70)

    # In a real implementation this would call into the trainer with the flag
    # --non_recurrent_generative_thinking plus the parameters above.
    print("\n[Placeholder] Ready to launch via trainer with --non_recurrent_generative_thinking")
    print("Update this script with actual launch logic once M4 decision is confirmed.")

if __name__ == "__main__":
    main()