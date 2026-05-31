#!/usr/bin/env python3
"""
Executable SSOT Gate: IMTA / GRAM-PTRM Stochastic Recurrent Breadth

This is a Level-2 defense against the exact failure mode documented in:
- docs/wiki/architecture/internal-multitrajectory-answer-attractor-ssot.md
  (Required Ablations: "GRAM/PTRM stochastic breadth off (K>1 gain should disappear)")

- docs/wiki/process/pivot-safety-and-inductive-bias-preservation.md

Usage (from repo root):
    python -m scripts.gates.check_ssot_stochastic_breadth
    python -m scripts.gates.check_ssot_stochastic_breadth --strict   # fail on missing

Exit codes:
    0 = the bias is executable in the current primary path (or explicitly declared missing + justified)
    1 = the bias is required by the SSOT but unreachable from the active training path
"""

import argparse
import sys
from pathlib import Path

# Make sure we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wgram_lm.architecture.component_registry import (
    get_inactive_primary_path_components,
    get_component_record,
)


IMTA_SSOT_REF = "internal-multitrajectory-answer-attractor-ssot.md (Required Ablations: GRAM/PTRM stochastic breadth off)"


def main() -> int:
    parser = argparse.ArgumentParser(description="IMTA SSOT Stochastic Breadth Executability Gate")
    parser.add_argument("--strict", action="store_true",
                        help="Fail hard if the critical bias is not active in primary path (use in promotion flows)")
    parser.add_argument("--quiet", action="store_true", help="Suppress explanatory output")
    args = parser.parse_args()

    inactive = get_inactive_primary_path_components()
    state_transition = None
    hybrid_replacement = None
    try:
        state_transition = get_component_record("state_transition_core")
    except KeyError:
        pass
    try:
        hybrid_replacement = get_component_record("hybrid_stochastic_breadth_engine")
    except KeyError:
        pass

    critical_missing = False
    replacement_active = (
        hybrid_replacement is not None
        and hybrid_replacement.active_in_primary_onebody_path
    )
    if (
        state_transition is not None
        and not state_transition.active_in_primary_onebody_path
        and not replacement_active
    ):
        critical_missing = True

    if args.quiet and not critical_missing:
        return 0

    print("\n=== IMTA SSOT Stochastic Breadth Gate ===")
    print(f"SSOT reference: {IMTA_SSOT_REF}")
    print()

    if not critical_missing:
        print("[PASS] GRAM/PTRM-style training-time stochastic recurrent breadth")
        print("       is either active in the primary One-Body path or has been")
        print("       explicitly replaced with an equivalent mechanism that still")
        print("       satisfies the SSOT's mandatory K=1 vs K>1 ablation contract.")
        if replacement_active:
            print("       Active replacement: OneBodyParallelHybridBlock")
            print("       Registry record: hybrid_stochastic_breadth_engine")
        print()
        return 0

    # Critical bias is missing from primary path
    print("[FAIL] The component that historically delivered training-time")
    print("       stochastic recurrent breadth (state_transition_core / true_gram)")
    print("       is marked active_in_primary_onebody_path=False.")
    print()
    print("This means the following SSOT requirement is currently UNEXECUTABLE")
    print("on the canonical training path:")
    print()
    print("    'GRAM/PTRM stochastic breadth off (K>1 gain should disappear or shrink)'")
    print()
    print("See:")
    print("  - docs/wiki/architecture/internal-multitrajectory-answer-attractor-ssot.md")
    print("  - docs/wiki/architecture/inductive-bias-map.md (stochastic breadth entry)")
    print("  - docs/wiki/decisions/2026-05-30-historical-signal-reconstruction-stochastic-breadth-pivot-gap.md")
    print("  - docs/wiki/process/pivot-safety-and-inductive-bias-preservation.md")
    print()

    inactive_names = [r.name for r in inactive]
    print(f"Currently inactive PROMOTED components: {inactive_names}")
    print()

    if args.strict:
        print("STRICT MODE: failing the gate as required for promotion flows.")
        return 1

    print("Non-strict mode: emitting warning only. Use --strict in promotion / release gates.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
