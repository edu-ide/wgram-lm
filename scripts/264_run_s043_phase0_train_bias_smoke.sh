#!/usr/bin/env bash
set -euo pipefail

# S043 Phase 0 Training Smoke
# Trains the tiny residual steering bias with strong donor-correct preservation.
#
# STRICT PRECAUTIONS (do not skip):
# - Start from a known good donor-preserving checkpoint (S040/S042 lineage recommended).
# - Run the corresponding eval smoke (263_...) BEFORE and AFTER training.
# - Watch donor_correct_margin_win_rate and repetition stats very carefully.
# - If donor fluency regresses on held-out donor-correct cases → stop immediately.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/s043_phase0_donor_bias_steering_minimal.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/nvme0n1p2/qtrm-runs/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_loop_joint_decoder_s040_from_selfrollout/last.pt}"
DATA_JSONL="${DATA_JSONL:-data/raw/text_train.jsonl data/raw/math_train.jsonl}"
OUT_DIR="${OUT_DIR:-runs/s043_phase0_bias_smoke}"
STEPS="${STEPS:-80}"

mkdir -p "${OUT_DIR}"

echo "=== S043 Phase 0: Training tiny residual steering bias (with preservation) ==="
echo "Config: $CONFIG"
echo "Init:   $INIT_CHECKPOINT"
echo "Out:    $OUT_DIR"

mkdir -p "$OUT_DIR"

PYTHONPATH=. python3 -m wgram_lm.training.train \
  --config "$CONFIG" \
  --use-donor \
  --data-jsonl $DATA_JSONL \
  --init-checkpoint "$INIT_CHECKPOINT" \
  2>&1 | tee "${OUT_DIR}/train.log"

echo ""
echo "Training done. Next (mandatory - follow Phase 0 Verification Guide):"
echo ""
echo "  1. Re-run the comparison smoke on the trained checkpoint:"
echo "     CHECKPOINT=${OUT_DIR}/last.pt scripts/263_run_s043_phase0_bias_steering_smoke.sh"
echo ""
echo "  2. Carefully read the 'Phase 0 Verification Guide' section in:"
echo "     docs/wiki/decisions/2026-05-30-s041-donor-preserving-freegen-smoke.md"
echo ""
echo "  3. Only proceed to larger experiments or Denoise-style prefix repair"
echo "     if first-token lift exists AND donor-correct fluency did not regress."
echo ""
echo "Detailed first-smoke guide + decision criteria:"
echo "  docs/wiki/decisions/2026-05-30-s041-donor-preserving-freegen-smoke.md"
echo "  → 'Phase 0 - Recommended First Actual Smoke'"