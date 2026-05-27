#!/bin/bash
set -euo pipefail
OUTDIR="checkpoints/hybrid_ri4_next_deepest_layer_202606"

echo "=== Forcing measurements for Next Deepest Layer ==="
for name in pure_parallel_latent_search evolutionary_latent_population test_time_self_modifying; do
    CKPT="$OUTDIR/$name/hybrid_ri4_cont_step12.pt"
    if [ -f "$CKPT" ]; then
        LOG="$OUTDIR/measure_${name}.log"
        nohup bash -c "
            source .venv/bin/activate && PYTHONPATH=. python scripts/measure_continuation_hybrid_192.py \
                --checkpoint $CKPT --scout --persistence_ablate --slots_off --router_ablate \
                2>&1 | tee $LOG
        " > /dev/null 2>&1 &
        echo "Measurement launched for $name -> $LOG"
    fi
done
