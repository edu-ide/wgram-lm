#!/usr/bin/env python3
"""
Unified Strict-B Probe for Pure Recursive Reasoning Intelligence (원시 추론 지능)

MANDATE (codified from user instruction "같은 벤치마크로 테스트트 해야지 순수 수학 추론이 원시 추론 지능이지"):

  ONLY measurements taken on the IDENTICAL benchmark
    data/eval/pure_recursive_reasoning_heldout_72.jsonl
  (no evidence, retrieval_allowed=false, memoryos_allowed=false, evidence=[])
  using STRICT B scoring (192-style --scoring forced_choice teacher-forced logprob
  OR tool-layer CandidatePoolSelector + QwenRegisterExtractor + state readback)
  are valid for cross-era claims about "원시 추론 지능" / primitive raw reasoning intelligence.

  - Memory-assisted results (with-evidence, span_mask, 71/72 answer-formation) are
    tracked on the separate Memory axis (see 95_eval_memory_retrieval.py).
  - Any other benchmark or A-style crude proxy (answer_logits rank without forced choice)
    does not count for Reasoning-axis historical comparison.

This script is the single entry point for all future "same benchmark" re-evaluations.
It supports graceful dispatch across eras via pre-existing worktrees or PYTHONPATH.

Usage examples:
  # Current d123cdc era, core 4 steps, forced_choice B
  python scripts/unified_pure_reasoning_strict_b_probe.py --era current --mode core4

  # Historical via worktree (re-uses 192 driver if present in that tree)
  python scripts/unified_pure_reasoning_strict_b_probe.py --worktree /tmp/qtrm_worktrees/explore-824be1b --era hybrid-824be1b

  # Full campaign re-probe (recommended after any major pivot)
  bash scripts/run_pure_reasoning_b_campaign.sh

After any run that produces a new strict-B number on the pure 72 set:
  1. Record the exact number + root cause in the wiki decision record.
  2. Execute the printed `git tag -a reasoning-*-* <SHA> -m "..."` (rich message required).
  3. Never claim "X era had better raw reasoning" without a fresh same-benchmark B number.

See SKILL.md > Performance Metric Tagging Discipline + the new Pure Reasoning
Intelligence Comparison Mandate section (added in this campaign).

The script itself does not mutate git state; it only produces reproducible commands
and a machine-readable comparison table.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

PURE_BENCHMARK = "data/eval/pure_recursive_reasoning_heldout_72.jsonl"
DEFAULT_CORE_STEPS = 4

# Known historical anchors (full SHAs resolved in campaign)
KNOWN_ERAS: Dict[str, Dict[str, Any]] = {
    "0def926b": {
        "sha": "0def926b475a",  # short is sufficient for tag lookup
        "worktree": "/tmp/qtrm_worktrees/explore-0def926b",
        "note": "CandidatePoolSelector 4-way + QwenRegisterExtractor era (projected 64/72 tool-layer)",
        "axis": "reasoning",
    },
    "7dd5e0c": {
        "sha": "7dd5e0c9dc5",
        "worktree": "/tmp/qtrm_worktrees/7dd5e0c",
        "note": "Weight-Shared Dual-State Recurrent Core peak (bare ~44%)",
        "axis": "reasoning",
    },
    "824be1b": {
        "sha": "824be1b8507ee63e3534605b2c115c38a322da3d",
        "worktree": "/tmp/qtrm_worktrees/explore-824be1b",
        "note": "OneBodyParallelHybridBlock introduction (skeletal hybrid B-probe 19/72)",
        "axis": "efficiency",  # discrimination drop relative to 5xx
    },
    "d123cdc": {
        "sha": "d123cdcd4d44",
        "worktree": None,  # current tree
        "note": "equation_state_binding + LightweightTypedEquationHead (current explore-d123cdc)",
        "axis": "reasoning",
    },
    "5dded277": {
        "sha": "5dded277817e0a9fb08eedfda8738858ed56c067",
        "worktree": None,
        "note": "reasoning gates + evaluation workflows (71/72 memory-assisted experiment point — DO NOT use for pure reasoning claims)",
        "axis": "memory",
    },
}


def run_192_forced_choice(
    cases: str = PURE_BENCHMARK,
    mode: str = f"qtrm_core_steps_{DEFAULT_CORE_STEPS}_no_evidence",
    max_cases: int | None = None,
    worktree: str | None = None,
) -> Dict[str, Any]:
    """Invoke the canonical 192 strict-B driver with forced_choice on the pure benchmark."""
    script = "scripts/192_eval_raw_intelligence.py"
    cmd = [
        sys.executable,
        script,
        "--cases",
        cases,
        "--scoring",
        "forced_choice",
        "--mode",
        mode,
        "--out",
        "/tmp/pure_b_unified_run.jsonl",
    ]
    if max_cases:
        cmd += ["--max-cases", str(max_cases)]

    env = os.environ.copy()
    if worktree:
        # Historical code path
        env["PYTHONPATH"] = str(Path(worktree) / "src") + ":" + env.get("PYTHONPATH", "")

    print(f"[unified-b] Running: {' '.join(cmd)}")
    print(f"[unified-b] PYTHONPATH override: {env.get('PYTHONPATH', '(none)')}")
    try:
        out = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT, text=True, timeout=300)
        # The 192 script writes a report JSONL; we also parse the final summary if printed.
        return {"success": True, "stdout": out[-2000:], "cmd": cmd}
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": e.output[-2000:] if e.output else str(e), "cmd": cmd}
    except Exception as e:
        return {"success": False, "error": str(e), "cmd": cmd}


def run_hybrid_legacy_probe(worktree: str) -> Dict[str, Any]:
    """Fall back to the dedicated hybrid B probe when the 192 driver is not compatible with that era."""
    probe = Path(worktree) / "probe_B_hybrid_824be1b.py"
    if not probe.exists():
        return {"success": False, "error": "probe_B_hybrid_824be1b.py not present in worktree"}

    cmd = [sys.executable, str(probe)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(worktree) / "src") + ":" + env.get("PYTHONPATH", "")
    try:
        out = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT, text=True, timeout=180)
        return {"success": True, "stdout": out, "note": "legacy hybrid probe"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    p = argparse.ArgumentParser(description="Unified strict-B probe enforcing same-benchmark rule for pure reasoning intelligence.")
    p.add_argument("--era", choices=list(KNOWN_ERAS.keys()) + ["current"], default="current",
                   help="Named historical era (uses pre-created worktree when available)")
    p.add_argument("--worktree", default=None, help="Explicit worktree path for any commit")
    p.add_argument("--mode", default=f"qtrm_core_steps_{DEFAULT_CORE_STEPS}_no_evidence",
                   help="192-style mode passed to forced_choice eval")
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--print-tag-command", action="store_true",
                   help="After a successful run, also emit the exact git tag -a command that must be executed")
    args = p.parse_args()

    print("=" * 72)
    print("PURE RECURSIVE REASONING STRICT-B UNIFIED PROBE")
    print("Benchmark locked to: data/eval/pure_recursive_reasoning_heldout_72.jsonl (no evidence)")
    print("Only these numbers are allowed for '원시 추론 지능' cross-era claims.")
    print("=" * 72)

    era = args.era
    info = KNOWN_ERAS.get(era, {})
    wt = args.worktree or info.get("worktree")

    if era == "current":
        wt = None
        print("[unified-b] Using current checkout (explore-d123cdc / d123cdc tree)")

    if wt and not Path(wt).exists():
        print(f"[unified-b] WARNING: worktree {wt} does not exist on this machine. "
              "You must create it first with `git worktree add` or reuse an existing one.")

    result: Dict[str, Any]
    if era in ("824be1b",) and wt:
        # Hybrid skeletal era prefers the dedicated graceful probe
        result = run_hybrid_legacy_probe(wt)
    else:
        result = run_192_forced_choice(
            cases=PURE_BENCHMARK,
            mode=args.mode,
            max_cases=args.max_cases,
            worktree=wt,
        )

    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result.get("success"):
        print("\n[MANDATE REMINDER]")
        print("  If this produced a new strict-B number on the pure 72 set,")
        print("  you MUST now:")
        print("    1. Paste the number + root-cause analysis into the wiki decision record.")
        print("    2. Execute the annotated tag command printed below (or construct one).")
        print("    3. Never compare raw reasoning intelligence across eras using any other benchmark or scoring.")

        if args.print_tag_command or True:
            sha = info.get("sha", "HEAD")
            axis = info.get("axis", "reasoning")
            suggested = f"reasoning-XXX-72-{era}" if axis == "reasoning" else f"efficiency-XXX-{era}"
            print(f"\nSuggested annotated tag (fill in the real number):")
            note = info.get("note", "")
            print("  git tag -a {} {} -m 'strict-B on pure_recursive_reasoning_heldout_72 (forced_choice) | {} | exact acc: Y/Y | root cause: ... | cross-era: ... | see unified_pure_reasoning_strict_b_probe.py'".format(suggested, sha, note))
    else:
        print("\n[ERROR] Probe failed. See stdout above. For very old trees you may need to port a minimal forced_choice scorer or run the historical 5xx_forced_choice_eval.py equivalent inside that worktree.")

    print("\nWorktree status (for reference):")
    print("  0def926b → /tmp/qtrm_worktrees/explore-0def926b")
    print("  7dd5e0c  → /tmp/qtrm_worktrees/7dd5e0c")
    print("  824be1b  → /tmp/qtrm_worktrees/explore-824be1b")
    print("  (others may exist)")


if __name__ == "__main__":
    main()
