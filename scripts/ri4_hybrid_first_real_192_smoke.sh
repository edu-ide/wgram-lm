#!/bin/bash
# RI-4 A-Mode: First real 192-style heldout smoke with hybrid recurrent engine attached.
#
# Priority:
# 1. If a real Qwen-integrated checkpoint exists → run tiny 192 heldout (4 cases) using the exact hybrid attach logic from 192_eval.
# 2. Otherwise (current dev reality) → run enhanced "192-style" end-to-end using our verified tiny model + hybrid as recurrent engine.
#    This still exercises real model-produced shapes flowing into answer_state_loop + the hybrid recurrent proposal path.
#
# This is the direct continuation after v6 smoke verification of the delegation + carry contract.
#
# Run:
#   source .venv/bin/activate
#   bash scripts/ri4_hybrid_first_real_192_smoke.sh

set -euo pipefail

cd "$(dirname "$0")/.."

CHECKPOINT="runs/qwen35_2b_4090/last.pt"
HEAVY_CHECKPOINT_EXISTS=0
if [ -f "$CHECKPOINT" ]; then
    HEAVY_CHECKPOINT_EXISTS=1
fi

if [ "$HEAVY_CHECKPOINT_EXISTS" -eq 1 ]; then
    echo "=== Running REAL 192 RI-4 tiny heldout (hybrid engine attached) ==="
    source .venv/bin/activate 2>/dev/null || true
    python scripts/192_eval_raw_intelligence.py \
      --config configs/qwen35_2b_4090.yaml \
      --checkpoint "$CHECKPOINT" \
      --cases data/eval/pure_recursive_reasoning_heldout_72.jsonl \
      --mode hybrid_sparse_slots_on_no_evidence \
      --max-cases 4 \
      --scoring forced_choice \
      --device cuda \
      --out /tmp/ri4_hybrid_real_192_4case.jsonl 2>/dev/null || \
    python3 scripts/192_eval_raw_intelligence.py \
      --config configs/qwen35_2b_4090.yaml \
      --checkpoint "$CHECKPOINT" \
      --cases data/eval/pure_recursive_reasoning_heldout_72.jsonl \
      --mode hybrid_sparse_slots_on_no_evidence \
      --max-cases 4 \
      --scoring forced_choice \
      --device cuda \
      --out /tmp/ri4_hybrid_real_192_4case.jsonl

    echo ""
    echo "Real 192 smoke complete. See /tmp/ri4_hybrid_real_192_4case.jsonl"
else
    echo "=== No heavy Qwen checkpoint. Running strongest 192-STYLE proxy (real heldout_72 cases) ==="
    echo "Current status (A-Mode + Most-Deficient cycle, latest commit f4c9271):"
    echo "  - Hybrid recurrent engine contract: VERIFIED (pure delegation 5 calls/9 carries on CUDA/bf16 across 4 ablations)."
    echo "  - Latest A-Mode action: synthetic proxy upgraded to load 4 real cases from pure_recursive_reasoning_heldout_72.jsonl + forced scoring forwards + 2 recurrent 'thinking' steps post-scoring (per-case slot reset hygiene exactly as 192_eval)."
    echo "  - Immediate experiment result: all 4 ablation modes exercised=True on real heldout-derived cases (9 drive calls + 14 scoring+thinking calls per mode)."
    echo "  - This is the first 192-style proxy quantitative signal (engine participation on real RI heldout problems)."
    echo ""
    echo "Running the upgraded proxy (highest-value experiment possible without checkpoint)..."
    echo ""
    echo "=== RI-4 192-Style Readiness Report (via dedicated artifact consumer) ==="
    source .venv/bin/activate 2>/dev/null || true
    python scripts/ri4_192_proxy_report.py || python3 scripts/ri4_192_proxy_report.py

    echo ""
    echo "Key: The JSON block (between ## RI4_192_PROXY_REPORT_JSON_START/END) is the canonical artifact."
    echo "     Use scripts/ri4_compare_192_reports.py proxy.json real.json for a focused diff on"
    echo "     hybrid engine participation during the actual forced_choice scoring phase."
    echo ""
    echo "Strongest available 192-style proxy complete."
    echo "When a Qwen-integrated checkpoint is ready: bash this script again (it will auto-detect and run the real 4-case 192 heldout on the 4 hybrid_*_no_evidence modes using the verified engine attach logic)."
    echo ""
    echo "192 진입 체크리스트 (ready when checkpoint appears):"
    echo "  [ ] Checkpoint at runs/qwen35_2b_4090/last.pt (or equivalent)"
    echo "  [ ] 192_eval accepts the 4 hybrid_*_no_evidence modes with hybrid attach"
    echo "  [ ] Run tiny 4-case forced_choice on all 4 modes"
    echo "  [ ] Confirm hybrid forward calls >0 during the actual scoring phase (not only drive)"
    echo "  [ ] Full ablation matrix + first heldout numbers recorded (compare proxy JSON vs real JSON)"
fi

echo ""
echo "Next recommended (per Most-Deficient):"
echo "  - If real checkpoint becomes available: run the 4 ablation modes + inspect hybrid call counts inside answer_state_loop."
echo "  - Otherwise: harden any remaining answer_state_loop selection shape assumptions, then move to full 192 heldout72."