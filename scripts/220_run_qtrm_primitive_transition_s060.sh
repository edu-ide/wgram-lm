#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_pure_recursive_primitive_transition_s060.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_full_state_sequence_s240/last.pt}"
TRAIN_CASES="${TRAIN_CASES:-data/filtered/pure_recursive_primitive_transition_train_cases.jsonl}"
DATA="${DATA:-data/filtered/pure_recursive_primitive_transition_preferences.jsonl}"
OUT_DIR="${OUT_DIR:-local_eval/qwen35_2b_pure_recursive_primitive_transition_s060}"
CASES_PER_FAMILY="${CASES_PER_FAMILY:-32}"
START_INDEX="${START_INDEX:-5000}"
MAX_REJECTED_PER_CASE="${MAX_REJECTED_PER_CASE:-1}"
STEPS="${STEPS:-60}"
LR="${LR:-}"
PRIMITIVE_TRANSITION_OPERATION_CE_WEIGHT="${PRIMITIVE_TRANSITION_OPERATION_CE_WEIGHT:-1.0}"

echo "=== QTRM primitive transition operation CE smoke ==="
echo "config: ${CONFIG}"
echo "init:   ${INIT_CHECKPOINT}"
echo "data:   ${DATA}"
echo "out:    ${OUT_DIR}"
echo "steps:  ${STEPS}"
echo

LR_ARGS=()
if [[ -n "${LR}" ]]; then
  LR_ARGS+=(--lr "${LR}")
fi

python scripts/190_build_pure_recursive_reasoning_cases.py \
  --out "${TRAIN_CASES}" \
  --cases-per-family "${CASES_PER_FAMILY}" \
  --start-index "${START_INDEX}"

python scripts/194_build_pure_recursive_reasoning_preferences.py \
  --cases "${TRAIN_CASES}" \
  --out "${DATA}" \
  --max-rejected-per-case "${MAX_REJECTED_PER_CASE}"

python scripts/196_train_pure_recursive_depth_supervised.py \
  --config "${CONFIG}" \
  --data-jsonl "${DATA}" \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --tokenizer-model-id Qwen/Qwen3.5-2B-Base \
  --steps "${STEPS}" \
  "${LR_ARGS[@]}" \
  --depth-steps 1,2,4,8 \
  --target-mode staged \
  --out-dir "${OUT_DIR}" \
  --final-logit-ce-weight 0.0 \
  --depth-final-ce-weight 0.0 \
  --all-depth-ce-weight 0.0 \
  --progress-margin-weight 0.0 \
  --choice-margin-weight 0.0 \
  --primitive-transition-operation-ce-weight "${PRIMITIVE_TRANSITION_OPERATION_CE_WEIGHT}"
