#!/bin/bash
set -euo pipefail

# One-command re-probe of all key eras under the Pure Recursive Reasoning Strict-B Mandate.
# Uses the unified driver + pre-existing worktrees from the 5xx-vs-hybrid B-probe campaign.
#
# This is the canonical way to answer "which architecture actually had better 원시 추론 지능?"
# Answer only comes from this exact benchmark + strict forced_choice B.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "======================================================================"
echo "PURE REASONING INTELLIGENCE - SAME BENCHMARK CAMPAIGN"
echo "Benchmark: data/eval/pure_recursive_reasoning_heldout_72.jsonl (evidence=[])"
echo "Scoring:    192-style --scoring forced_choice (strict B)"
echo "======================================================================"

# Current tree (d123cdc era)
echo
echo "=== [1/5] Current tree (explore-d123cdc / d123cdc) ==="
python scripts/unified_pure_reasoning_strict_b_probe.py --era current --mode qtrm_core_steps_4_no_evidence --print-tag-command || true

# 0def926b (CandidatePoolSelector 64/72 projection era)
echo
echo "=== [2/5] 0def926b (tool-layer selector peak) ==="
if [ -d /tmp/qtrm_worktrees/explore-0def926b ]; then
    python scripts/unified_pure_reasoning_strict_b_probe.py \
        --worktree /tmp/qtrm_worktrees/explore-0def926b \
        --era 0def926b \
        --mode qtrm_core_steps_4_no_evidence \
        --print-tag-command || true
else
    echo "  [SKIP] worktree not present on this machine"
fi

# 7dd5e0c (Dual-State Core peak)
echo
echo "=== [3/5] 7dd5e0c (weight-shared dual-state peak) ==="
if [ -d /tmp/qtrm_worktrees/7dd5e0c ]; then
    python scripts/unified_pure_reasoning_strict_b_probe.py \
        --worktree /tmp/qtrm_worktrees/7dd5e0c \
        --era 7dd5e0c \
        --mode qtrm_core_steps_4_no_evidence \
        --print-tag-command || true
else
    echo "  [SKIP] worktree not present"
fi

# 824be1b (hybrid introduction - expected discrimination drop)
echo
echo "=== [4/5] 824be1b (OneBodyParallelHybridBlock skeletal) ==="
if [ -d /tmp/qtrm_worktrees/explore-824be1b ]; then
    python scripts/unified_pure_reasoning_strict_b_probe.py \
        --worktree /tmp/qtrm_worktrees/explore-824be1b \
        --era 824be1b \
        --print-tag-command || true
else
    echo "  [SKIP] worktree not present"
fi

# 5dded277 (memory-assisted 71/72 point - explicitly NOT for pure reasoning claims)
echo
echo "=== [5/5] 5dded277 (memory-assisted experiment point — for Memory axis only) ==="
echo "  This era produced 71/72 with evidence + span_mask selector."
echo "  It is tracked under memory-* tags, NOT reasoning-* tags for 원시 추론 지능."
echo "  (Intentionally skipped for pure-reasoning B comparison.)"

echo
echo "======================================================================"
echo "CAMPAIGN COMPLETE"
echo "Next steps for any new number:"
echo "  1. Record raw accuracy + root cause in wiki (raw-intelligence or architecture decision page)"
echo "  2. git tag -a <reasoning- or efficiency-...> <exact-sha> -m 'rich message with cross-era comparison'"
echo "  3. Update SKILL.md historical anchor table if a new peak or cliff is established"
echo "======================================================================"
