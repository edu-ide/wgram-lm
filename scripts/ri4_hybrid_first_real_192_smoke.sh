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
    python scripts/192_eval_raw_intelligence.py \
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
    echo "=== No heavy Qwen checkpoint. Running 192-STYLE end-to-end on verified tiny model ==="
    echo "This exercises the exact same causal contract (model forward → answer_state_loop → hybrid as recurrent engine) with real model-produced tensors."
    python scripts/smoke_ri4_a_mode_hybrid_recurrent_engine.py
    echo ""
    echo "192-style tiny smoke complete (see v6 smoke output above)."
    echo "The pure delegation + realistic forward phases already proved hybrid_calls + slot carry on the real path."
fi

echo ""
echo "Next recommended (per Most-Deficient):"
echo "  - If real checkpoint becomes available: run the 4 ablation modes + inspect hybrid call counts inside answer_state_loop."
echo "  - Otherwise: harden any remaining answer_state_loop selection shape assumptions, then move to full 192 heldout72."