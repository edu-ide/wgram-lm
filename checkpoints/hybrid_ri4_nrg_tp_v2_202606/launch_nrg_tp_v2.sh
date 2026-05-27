#!/bin/bash
# NRG-TP v2 Diagnostic
# Upgraded non-recurrent generative thinking (parallel candidate sampling)

set -euo pipefail

GOLD="local_eval/642_adaptive_fine_tuned_200step/adaptive_phase2_checkpoint.pt"
BASE="source .venv/bin/activate && PYTHONPATH=. python scripts/train_hybrid_ri4_real_continuation_minimal.py --steps 12 --d_model 512 --batch 1 --gold_path $GOLD --save_every 4 --enable_stochastic_breadth"

OUTDIR="checkpoints/hybrid_ri4_nrg_tp_v2_202606"
mkdir -p "$OUTDIR"

echo "=== NRG-TP v2 Launch ==="
echo "This is the upgraded non-recurrent generative thinking diagnostic (parallel candidates)."
echo ""

nohup bash -c "$BASE --non_recurrent_generative_thinking --out_dir $OUTDIR/nrg_tp_v2 2>&1 | tee $OUTDIR/nrg_tp_v2.log" > /dev/null 2>&1 &
echo "Launched: nrg_tp_v2 (PID $!)"

echo ""
echo "Pre-armed measurement: $OUTDIR/measure_nrg_tp_v2.sh"
echo "Monitor: tail -f $OUTDIR/*.log"
