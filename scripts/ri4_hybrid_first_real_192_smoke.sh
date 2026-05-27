#!/bin/bash
# RI-4 A-Mode: First real 192-style heldout smoke with hybrid recurrent engine attached.
# Run after the v6 smoke has verified the delegation + carry contract.
#
# This exercises the *actual* causal path used in production 192_eval:
#   model forward / scoring → answer_state_loop (with hybrid as the recurrent proposal engine)
#   using real model-produced trajectory / workspace / context shapes.
#
# Usage (example, adjust checkpoint/config/cases as available in your env):
#   source .venv/bin/activate
#   bash scripts/ri4_hybrid_first_real_192_smoke.sh
#
# Expected outcome (if successful):
#   - A few cases complete without shape/index/unpack crashes.
#   - The attached hybrid block is exercised (we can add light logging if needed).
#   - First quantitative signal on hybrid_*_no_evidence modes vs their ablation counterparts.

set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/192_eval_raw_intelligence.py \
  --config configs/qwen35_2b_4090.yaml \
  --checkpoint runs/qwen35_2b_4090/last.pt \
  --cases data/eval/pure_recursive_reasoning_heldout_72.jsonl \
  --mode hybrid_sparse_slots_on_no_evidence \
  --max-cases 4 \
  --scoring forced_choice \
  --device cuda \
  --out /tmp/ri4_hybrid_first_real_192_smoke.jsonl

echo ""
echo "=== RI-4 hybrid first real 192 smoke complete ==="
echo "Output: /tmp/ri4_hybrid_first_real_192_smoke.jsonl"
echo "Next: compare against the ablation modes + inspect whether hybrid calls were made inside answer_state_loop."