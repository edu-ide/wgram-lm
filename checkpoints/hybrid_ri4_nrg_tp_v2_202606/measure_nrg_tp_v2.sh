#!/bin/bash
set -euo pipefail
OUTDIR="checkpoints/hybrid_ri4_nrg_tp_v2_202606"

echo "=== Forcing NRG-TP v2 measurements ==="
for name in nrg_tp_v2; do
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
