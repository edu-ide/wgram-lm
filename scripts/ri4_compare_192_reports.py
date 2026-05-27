#!/usr/bin/env python3
"""
RI-4 192 Report Comparison Tool (A-Mode 192-entry tooling)

Takes two RI-4 192-style JSON reports (typically proxy vs real tiny heldout)
and produces a focused, human-readable diff centered on hybrid recurrent engine
participation during the forced_choice scoring + thinking phase.

This is the canonical tool for the moment a real checkpoint + 192 tiny heldout
run becomes possible. It turns the machine-readable artifacts produced by
`ri4_192_proxy_report.py` (and future real 192 runs) into an immediate,
actionable comparison.

Usage examples:
  python scripts/ri4_compare_192_reports.py --proxy proxy.json --real real_192.json
  python scripts/ri4_compare_192_reports.py proxy_report.json real_report.json

Part of the A-Mode + Most-Deficient discipline for RI-4 192 entry.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

KEY_DRIVE = "hybrid_forward_call_count"
KEY_SCORING = "scoring_hybrid_calls"
KEY_EXERCISED = "engine_exercised"
KEY_REAL_HELDOUT = "used_real_heldout"

def load_report(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(2)
    with p.open(encoding="utf-8") as f:
        data = json.load(f)

    # Support both the top-level report and the inner "modes" list
    if "modes" in data:
        return data
    if isinstance(data, list):
        return {"modes": data}
    # If someone passed a single mode dict, wrap it
    if "mode" in data:
        return {"modes": [data]}
    print(f"ERROR: Unrecognized report format in {path}", file=sys.stderr)
    sys.exit(2)

def get_modes(report: dict[str, Any]) -> list[dict[str, Any]]:
    return report.get("modes", [])

def build_matrix(modes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for m in modes:
        name = m.get("mode", "unknown")
        out[name] = {
            "drive": m.get(KEY_DRIVE, 0),
            "scoring": m.get(KEY_SCORING, 0),
            "exercised": m.get(KEY_EXERCISED, False),
            "real_heldout": m.get(KEY_REAL_HELDOUT, False),
        }
    return out

def print_comparison(proxy: dict[str, Any], real: dict[str, Any] | None = None) -> None:
    p_modes = build_matrix(get_modes(proxy))
    r_modes = build_matrix(get_modes(real)) if real else {}

    print("=== RI-4 192 Report Comparison (Hybrid Engine Participation) ===")
    print("Focus: hybrid forward calls during recurrent drive + scoring/thinking phase")
    print()

    all_modes = sorted(set(p_modes.keys()) | set(r_modes.keys()))

    header = "Mode                              | Proxy Drive | Proxy Score | Real Drive | Real Score | Delta Score | Exercised (P/R)"
    print(header)
    print("-" * len(header))

    total_delta = 0
    for mode in all_modes:
        pm = p_modes.get(mode, {})
        rm = r_modes.get(mode, {})

        p_drive = pm.get("drive", 0)
        p_score = pm.get("scoring", 0)
        r_drive = rm.get("drive", 0)
        r_score = rm.get("scoring", 0)

        delta = r_score - p_score
        total_delta += delta

        p_ex = pm.get("exercised", False)
        r_ex = rm.get("exercised", False) if rm else "N/A"

        print(
            f"{mode:34} | "
            f"{p_drive:11} | "
            f"{p_score:11} | "
            f"{r_drive:10} | "
            f"{r_score:10} | "
            f"{delta:+11} | "
            f"{p_ex}/{r_ex}"
        )

    print()
    print(f"Total scoring-phase delta (real - proxy): {total_delta:+d}")
    print()

    if real:
        print("Interpretation guide:")
        print("  Positive delta in scoring phase = hybrid engine was more active in real run than proxy predicted.")
        print("  Large negative delta on an ablation mode = that ablation is effectively disabling the engine in the real harness.")
        print("  All modes exercised=True on both sides = strong confirmation of the verified engine contract.")
    else:
        print("Proxy-only view (no real report provided).")
        print("Run with --real <real_192_json> once a checkpoint + tiny 192 heldout is available.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two RI-4 192-style JSON reports (proxy vs real).")
    parser.add_argument("proxy", nargs="?", help="Proxy JSON report (or use --proxy)")
    parser.add_argument("real", nargs="?", help="Real 192 JSON report (optional)")
    parser.add_argument("--proxy", dest="proxy_flag", help="Proxy report path")
    parser.add_argument("--real", dest="real_flag", help="Real report path")
    args = parser.parse_args()

    proxy_path = args.proxy_flag or args.proxy
    real_path = args.real_flag or args.real

    if not proxy_path:
        print("ERROR: Proxy report path is required (positional or --proxy)", file=sys.stderr)
        parser.print_usage()
        sys.exit(2)

    proxy = load_report(proxy_path)
    real = load_report(real_path) if real_path else None

    print_comparison(proxy, real)

if __name__ == "__main__":
    main()
