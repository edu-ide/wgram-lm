#!/bin/bash
# Run this after both m1_on and m1_off short continuations finish.

CKPT_ON="experiments/ri1_m1_ablation/m1_on/hybrid_ri4_cont_step36.pt"
CKPT_OFF="experiments/ri1_m1_ablation/m1_off/hybrid_ri4_cont_step36.pt"

echo "=== Measuring M1 ON vs M1 OFF scaling (fastest causal evidence) ==="

for d in 1 4 8; do
  echo ""
  echo "=== Depth $d - M1 ON ==="
  PYTHONPATH=. /home/tripleyoung/.local/bin/uv run python experiments/matched_port_evaluation_a9617cd8/compute_72_accuracy_from_base.py \
    --checkpoint "$CKPT_ON" \
    --all-three \
    --effective-depth $d

  echo ""
  echo "=== Depth $d - M1 OFF (ablation) ==="
  PYTHONPATH=. /home/tripleyoung/.local/bin/uv run python experiments/matched_port_evaluation_a9617cd8/compute_72_accuracy_from_base.py \
    --checkpoint "$CKPT_OFF" \
    --all-three \
    --effective-depth $d
done

echo ""
echo "=== Ablation measurement complete ==="
echo "Compare the two scaling curves. Big difference in d=8 lift = strong causal evidence for M1."
