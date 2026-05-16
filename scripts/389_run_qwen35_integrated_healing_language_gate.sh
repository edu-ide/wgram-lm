#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}"
export PYTHONPATH="${PYTHONPATH:-src}"

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B-Base}"
CHECKPOINT="${CHECKPOINT:-local_eval/qwen35_integrated_healing_l23_langkl_s100_20260516/last_core.pt}"
OUT_DIR="${OUT_DIR:-local_eval/qwen35_integrated_healing_l23_langkl_s100_language_gate_20260516}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

"${PYTHON}" scripts/367_eval_qwen_backbone_language_gate.py \
  --model-id "${MODEL_ID}" \
  --checkpoint "${CHECKPOINT}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --max-seq-len "${MAX_SEQ_LEN:-128}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-64}" \
  --max-generation-prompts "${MAX_GENERATION_PROMPTS:-12}" \
  --core-impl qwen_layer_wrapped \
  --qwen-core-layer-indices "${QWEN_CORE_LAYER_INDICES:-3}" \
  --mandatory-core \
  --core-adapter-dim "${CORE_ADAPTER_DIM:-128}" \
  --core-delta-adapter-mode "${CORE_DELTA_ADAPTER_MODE:-add}" \
  --residual-scale "${RESIDUAL_SCALE:-0.05}" \
  --min-top1-agreement "${MIN_TOP1_AGREEMENT:-0.75}" \
  --min-top5-agreement "${MIN_TOP5_AGREEMENT:-0.90}" \
  --max-repeated-token-run "${MAX_REPEATED_TOKEN_RUN:-8}" \
  --min-unique-ratio "${MIN_UNIQUE_RATIO:-0.20}"
