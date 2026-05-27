#!/bin/bash
# Next Deepest Layer after NRG-TP
# These directions go beyond "non-recurrent thinking" into fundamentally different computational primitives.

set -euo pipefail

GOLD="local_eval/642_adaptive_fine_tuned_200step/adaptive_phase2_checkpoint.pt"
BASE="source .venv/bin/activate && PYTHONPATH=. python scripts/train_hybrid_ri4_real_continuation_minimal.py --steps 12 --d_model 512 --batch 1 --gold_path $GOLD --save_every 4 --enable_stochastic_breadth"

OUTDIR="checkpoints/hybrid_ri4_next_deepest_layer_202606"
mkdir -p "$OUTDIR"

echo "=== LAUNCHING NEXT DEEPEST LAYER (preparing while NRG-TP v2 runs) ==="
echo ""

# 1. Pure parallel latent search (no sequential recurrence at all during thinking)
nohup bash -c "$BASE --pure_parallel_latent_search --out_dir $OUTDIR/pure_parallel_latent_search 2>&1 | tee $OUTDIR/pure_parallel_latent_search.log" > /dev/null 2>&1 &
echo "1. pure_parallel_latent_search launched"

# 2. Evolutionary population in latent space
nohup bash -c "$BASE --evolutionary_latent_population --out_dir $OUTDIR/evolutionary_latent_population 2>&1 | tee $OUTDIR/evolutionary_latent_population.log" > /dev/null 2>&1 &
echo "2. evolutionary_latent_population launched"

# 3. Test-time self-modifying architecture
nohup bash -c "$BASE --test_time_self_modifying_arch --out_dir $OUTDIR/test_time_self_modifying 2>&1 | tee $OUTDIR/test_time_self_modifying.log" > /dev/null 2>&1 &
echo "3. test_time_self_modifying_arch launched"

echo ""
echo "Next deepest layer launched in parallel."
echo "Pre-armed measurement: $OUTDIR/measure_deepest_layer.sh"
echo "Monitor with: tail -f $OUTDIR/*.log"
