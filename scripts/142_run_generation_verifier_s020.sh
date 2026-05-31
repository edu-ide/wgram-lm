#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/qwen35_2b_4090_generation_verifier_s020.yaml}"
DATA="${DATA:-data/filtered/qtrm_generated_verifier/qtrm_v2_repeat_gate_s50_generation_verifier.jsonl}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/qwen35_2b_4090_hf_first_wave_warmup_v2_s400/last.pt}"

echo "=== Training QTRM generation verifier heads ==="
echo "Config: ${CONFIG}"
echo "Data: ${DATA}"
echo "Init checkpoint: ${INIT_CHECKPOINT}"

uv run python -m wgram_lm.training.train \
  --config "${CONFIG}" \
  --use-donor \
  --data-jsonl "${DATA}" \
  --init-checkpoint "${INIT_CHECKPOINT}"
