#!/usr/bin/env bash
set -euo pipefail

# S043 Mixed Denoise Recovery Training
# Merges synthetic and on-policy harvested failure prefixes to train recovery steering.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/s043_denoise_recovery_mixed.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-checkpoints/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_next_token_decoder_selfrollout_s040_from_s080/last.pt}"

SYNTHETIC_DATA="data/tmp/denoise_bad_prefixes_v3_160.jsonl"
HARVESTED_DATA="data/tmp/real_bad_prefixes_from_current.jsonl"
MIXED_DATA="data/tmp/mixed_denoise_prefixes.jsonl"

OUT_DIR="runs/s043_denoise_recovery_mixed"
mkdir -p "${OUT_DIR}"

echo "=== Merging Datasets ==="
if [ ! -f "$HARVESTED_DATA" ]; then
  echo "Error: harvested data $HARVESTED_DATA not found. Run scripts/generate_real_bad_prefixes.py first!"
  exit 1
fi

cat "$SYNTHETIC_DATA" "$HARVESTED_DATA" > "$MIXED_DATA"
echo "Merged data written to: $MIXED_DATA"
echo "Total samples in merged dataset:"
wc -l "$MIXED_DATA"

echo ""
echo "=== Starting Mixed Denoise Recovery Training ==="
echo "Config:          $CONFIG"
echo "Init Checkpoint: $INIT_CHECKPOINT"
echo "Output Dir:      $OUT_DIR"
echo ""

PYTHONPATH=.:src /mnt/data4tb/venv_sglang_pr23000/bin/python -m wgram_lm.training.train \
  --config "$CONFIG" \
  --use-donor \
  --data-jsonl "$MIXED_DATA" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  2>&1 | tee "${OUT_DIR}/train.log"

echo ""
echo "Mixed Denoise Recovery Training finished!"
echo "Next: Run the steered evaluation under scale=2.0 on the heldout reasoning cases to measure recovery performance!"
