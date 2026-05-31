#!/usr/bin/env bash
set -euo pipefail

# S043 Denoise Modest-1
# First serious step toward real free-generation recovery via self-rollout style training.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/s043_denoise_modest_1.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/s043_phase0_tiny_safe/last.pt}"
DATA_JSONL="${DATA_JSONL:-data/tmp/phase0_mixed_denoise.jsonl}"
OUT_DIR="${OUT_DIR:-runs/s043_denoise_modest_1}"

mkdir -p "${OUT_DIR}"

echo "=== S043 Denoise Modest-1 ==="
echo "Config: $CONFIG"
echo "Init:   $INIT_CHECKPOINT"
echo "Data:   $DATA_JSONL"
echo "Out:    $OUT_DIR"

PYTHONPATH=. python3 -m wgram_lm.training.train \
  --config "$CONFIG" \
  --use-donor \
  --data-jsonl $DATA_JSONL \
  --init-checkpoint "$INIT_CHECKPOINT" \
  2>&1 | tee "${OUT_DIR}/train.log"

echo ""
echo "Modest-1 training finished."
echo "Next: Run 263 smoke on ${OUT_DIR}/last.pt and compare S043 DIAGNOSTIC BLOCK + generation quality."
