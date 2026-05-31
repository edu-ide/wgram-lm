#!/usr/bin/env bash
set -euo pipefail

# S043 Lightweight Denoise Prefix Recovery Experiment (first pass)
# Goal: Expose the model to bad prefixes + continue training with strong preservation + first-token objectives.
# This is the minimal first step toward real self-rollout recovery (DenoiseRL style).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/s043_denoise_tiny_safe.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/s043_phase0_tiny_safe/last.pt}"
OUT_DIR="${OUT_DIR:-runs/s043_denoise_tiny}"
STEPS="${STEPS:-25}"

# Mixed data: good examples + bad prefix examples
# For this lightweight pass we just concatenate them.
MIXED_DATA="data/tmp/phase0_mixed_denoise.jsonl"

mkdir -p "${OUT_DIR}"
mkdir -p data/tmp

# Build mixed dataset (good + bad prefixes)
echo "Building mixed dataset for Denoise experiment..."
cat data/tmp/phase0_tiny_math_40.jsonl data/tmp/phase0_denoise_bad_prefixes_30.jsonl > "${MIXED_DATA}"
echo "Mixed dataset has $(wc -l < "${MIXED_DATA}") lines"

echo "=== S043 Denoise Tiny Experiment ==="
echo "Config: $CONFIG"
echo "Init:   $INIT_CHECKPOINT"
echo "Out:    $OUT_DIR"
echo "Steps:  $STEPS"

PYTHONPATH=. python3 -m wgram_lm.training.train \
  --config "$CONFIG" \
  --use-donor \
  --data-jsonl "${MIXED_DATA}" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  2>&1 | tee "${OUT_DIR}/train.log"

echo ""
echo "Denoise tiny experiment finished."
echo "Next (recommended): Re-run 263 smoke on ${OUT_DIR}/last.pt and compare the S043 DIAGNOSTIC BLOCK."
