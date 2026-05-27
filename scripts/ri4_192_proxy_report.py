#!/usr/bin/env python3
"""
RI-4 192 Proxy Report Extractor (A-Mode tooling)

Reliable, single-purpose consumer for the machine-readable artifact emitted by
ri4_hybrid_synthetic_192_style_test.py.

Usage:
  python scripts/ri4_192_proxy_report.py

This script is the canonical way to obtain the "RI-4 192-Style Readiness Report"
in both human and machine-readable form. It will be the foundation for direct
JSON diffing once real 192 tiny heldout runs with the hybrid engine are possible.

Part of the A-Mode + Most-Deficient discipline for RI-4 192 entry preparation.
"""

from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path

DELIM_START = "## RI4_192_PROXY_REPORT_JSON_START"
DELIM_END = "## RI4_192_PROXY_REPORT_JSON_END"

def extract_json_block(text: str) -> dict | None:
    m = re.search(
        rf"{re.escape(DELIM_START)}\s*(.*?)\s*{re.escape(DELIM_END)}",
        text,
        re.DOTALL,
    )
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

def run_proxy_and_get_report() -> dict | None:
    # Run the known proxy test (prefer python, fall back to python3, respect venv)
    candidates = [
        "source .venv/bin/activate 2>/dev/null || true; python scripts/ri4_hybrid_synthetic_192_style_test.py",
        "source .venv/bin/activate 2>/dev/null || true; python3 scripts/ri4_hybrid_synthetic_192_style_test.py",
    ]
    for cmd in candidates:
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = result.stdout + result.stderr
            data = extract_json_block(output)
            if data is not None:
                return data
        except Exception:
            continue
    return None

def print_human_matrix(data: dict) -> None:
    modes = data.get("modes", [])
    print("=== RI-4 192-Style Readiness Report (A-Mode Proxy) ===")
    print("Machine-readable artifact for direct future diff against real 192 tiny heldout.")
    print()
    print("Mode                              | Drive | Score+Think | Cases | Exercised | RealHeldout")
    print("----------------------------------|-------|-------------|-------|-----------|-------------")
    for m in modes:
        print(
            f"{m.get('mode', ''):34} | "
            f"{m.get('hybrid_forward_call_count', 0):5} | "
            f"{m.get('scoring_hybrid_calls', 0):11} | "
            f"{m.get('cases_run', 0):5} | "
            f"{str(m.get('engine_exercised', False)):9} | "
            f"{str(m.get('used_real_heldout', False))}"
        )
    print()
    s = data.get("summary", {})
    print(f"Summary: all_exercised={s.get('all_exercised')}  real_heldout_used={s.get('real_heldout_used')}")
    print()
    print("Key: The JSON block below (between the delimiters) is the canonical artifact.")
    print("     Future real 192 tiny heldout runs can be diffed directly against it.")
    print()

def main() -> None:
    data = run_proxy_and_get_report()
    if data is None:
        print("ERROR: Could not obtain RI-4 192 proxy JSON artifact.", file=sys.stderr)
        sys.exit(1)

    print_human_matrix(data)

    # Emit the raw canonical JSON artifact (for diffing / archiving)
    print("## RI4_192_PROXY_REPORT_JSON_START")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print("## RI4_192_PROXY_REPORT_JSON_END")

if __name__ == "__main__":
    main()
