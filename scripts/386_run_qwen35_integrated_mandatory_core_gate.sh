#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}"
export PYTHONPATH="${PYTHONPATH:-src}"

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B-Base}"
OUT_DIR="${OUT_DIR:-local_eval/qwen35_integrated_mandatory_core_gate_s300_20260516}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"

"${PYTHON}" scripts/362_train_qwen_backbone_qtrm_core_gate.py \
  --model-id "${MODEL_ID}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --max-seq-len "${MAX_SEQ_LEN:-96}" \
  --core-impl qwen_layer_wrapped \
  --qwen-core-layer-indices "${QWEN_CORE_LAYER_INDICES:-3}" \
  --mandatory-core \
  --core-adapter-dim "${CORE_ADAPTER_DIM:-128}" \
  --core-delta-adapter-mode "${CORE_DELTA_ADAPTER_MODE:-add}" \
  --residual-scale "${RESIDUAL_SCALE:-0.05}" \
  --h-cycles "${H_CYCLES:-1}" \
  --l-cycles "${L_CYCLES:-1}" \
  --outer-steps "${OUTER_STEPS:-1}" \
  --steps "${STEPS:-300}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --train-cases "${TRAIN_CASES:-768}" \
  --eval-cases "${EVAL_CASES:-512}" \
  --case-mode "${CASE_MODE:-hard_v1}" \
  --lr "${LR:-5.0e-4}" \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --grad-clip "${GRAD_CLIP:-1.0}" \
  --kl-weight "${KL_WEIGHT:-0.01}" \
  --eval-every-steps "${EVAL_EVERY_STEPS:-100}" \
  --restore-best-checkpoint \
  --min-reasoning-gain "${MIN_REASONING_GAIN:-0.02}" \
  --min-language-top1-agreement "${MIN_LANGUAGE_TOP1_AGREEMENT:-0.75}" \
  --min-family-gain "${MIN_FAMILY_GAIN:-0.0}" \
  --min-family-core-accuracy "${MIN_FAMILY_CORE_ACCURACY:-0.08}" \
  --seed "${SEED:-20260516}" \
  --log-every "${LOG_EVERY:-20}"
