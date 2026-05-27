#!/bin/bash
# Helper script to run the state ablation robustness probe on all completed variants
# of a G-stage matrix once it finishes.

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <matrix_directory>"
    echo "Example: $0 local_556_real642_full_gstage_matrix_fixed_20260527_1256"
    exit 1
fi

MATRIX_DIR="$1"
PROBE_SCRIPT="scripts/probe_state_ablation_robustness.py"
OUTPUT_DIR="$MATRIX_DIR/probe_results"

VENV_PY="/home/tripleyoung/qtrm-workspace/qtrm_multimodal_memoryos/.venv/bin/python"

mkdir -p "$OUTPUT_DIR"

echo "=== Running state ablation robustness probe on all variants in $MATRIX_DIR ==="

for variant in "$MATRIX_DIR"/*/; do
    if [ -d "$variant" ]; then
        name=$(basename "$variant")
        ckpt="$variant/best.pt"
        
        if [ -f "$ckpt" ]; then
            echo ""
            echo "=== Probing variant: $name ==="
            PYTHONPATH=. "$VENV_PY" "$PROBE_SCRIPT" \
                --ckpt "$ckpt" \
                --steps 40 \
                --trials 6 \
                --ablation zero \
                2>&1 | tee "$OUTPUT_DIR/${name}_probe.txt"
        else
            echo "Skipping $name (no best.pt yet)"
        fi
    fi
done

echo ""
echo "=== All probes completed. Results in $OUTPUT_DIR ==="
