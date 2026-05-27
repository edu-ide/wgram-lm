#!/bin/bash
# Launcher for the first real Substrate Diagnostic:
# Non-Recurrent Generative Thinking Phase (NRG-TP)

set -euo pipefail

GOLD="local_eval/642_adaptive_fine_tuned_200step/adaptive_phase2_checkpoint.pt"
BASE="source .venv/bin/activate && PYTHONPATH=. python scripts/train_hybrid_ri4_real_continuation_minimal.py --steps 12 --d_model 512 --batch 1 --gold_path $GOLD --save_every 4 --enable_stochastic_breadth"

OUTDIR="checkpoints/hybrid_ri4_nrg_tp_diagnostic_202606"
mkdir -p "$OUTDIR"

echo "=== NRG-TP Diagnostic Launch ==="
echo "This is the first experiment designed to step outside the current recurrent + memory participation substrate family."
echo ""

nohup bash -c "$BASE --non_recurrent_generative_thinking --out_dir $OUTDIR/nrg_tp_v1 2>&1 | tee $OUTDIR/nrg_tp_v1.log" > /dev/null 2>&1 &
echo "Launched: nrg_tp_v1 (PID $!)"

echo ""
echo "Measurement script prepared at: $OUTDIR/measure_nrg_tp.sh"
echo "Monitor with: tail -f $OUTDIR/*.log"
