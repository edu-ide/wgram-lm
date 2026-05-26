#!/usr/bin/env bash
set -euo pipefail

# Fast tokenizer-free ablation.
#
# This is not full BLT yet.  It is the smallest causal test:
#   same native recurrent PrefixLM trainer,
#   same instruction/response row contract,
#   UTF-8 bytes instead of BPE token ids.

ROOT="${ROOT:-/home/tripleyoung/qtrm-workspace/qtrm_multimodal_memoryos}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-20260524_STAGE94C_LOCAL_BYTEFREE82M_LANGSAMPLE}"
OUT_DIR="${OUT_DIR:-${ROOT}/local_eval/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-/tmp/${RUN_NAME}.log}"
SAMPLED_DATA="${SAMPLED_DATA:-/tmp/20260524_STAGE94_LOCAL_BYTEFREE_SAMPLE/sampled}"
DEVICE="${DEVICE:-cuda}"

STEPS="${STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-6}"
SEQ_LEN="${SEQ_LEN:-384}"
SEED="${SEED:-9501}"
LR="${LR:-2.2e-4}"
EVAL_EVERY="${EVAL_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-25}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
MODEL_CHECKPOINT_EVERY="${MODEL_CHECKPOINT_EVERY:-1000}"

D_MODEL="${D_MODEL:-384}"
N_HEADS="${N_HEADS:-6}"
N_KV_HEADS="${N_KV_HEADS:-2}"
D_FF="${D_FF:-1024}"
TRAIN_THINK_STEPS="${TRAIN_THINK_STEPS:-2}"
NITP_LOSS_WEIGHT="${NITP_LOSS_WEIGHT:-0.0}"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${SAMPLED_DATA}/tokens.npy" ]]; then
  echo "missing byte sampled data: ${SAMPLED_DATA}/tokens.npy" >&2
  exit 2
fi

nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/src" \
  "${PYTHON}" "${ROOT}/scripts/534_train_native_prefixlm_dataio.py" \
    --sampled-data "${SAMPLED_DATA}" \
    --out-dir "${OUT_DIR}" \
    --device "${DEVICE}" \
    --steps "${STEPS}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --model-checkpoint-every "${MODEL_CHECKPOINT_EVERY}" \
    --batch-size "${BATCH_SIZE}" \
    --seq-len "${SEQ_LEN}" \
    --d-model "${D_MODEL}" \
    --n-heads "${N_HEADS}" \
    --n-kv-heads "${N_KV_HEADS}" \
    --d-ff "${D_FF}" \
    --train-think-steps "${TRAIN_THINK_STEPS}" \
    --length-bucketed-batches \
    --trim-batch-to-max-length \
    --loss-kernel auto \
    --optimizer adamw \
    --amp-dtype bf16 \
    --matmul-precision high \
    --lr "${LR}" \
    --lr-warmup-steps 500 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --weight-decay 0.1 \
    --eval-every "${EVAL_EVERY}" \
    --eval-max-rows 256 \
    --eval-batch-size 4 \
    --eval-max-batches 0 \
    --log-every "${LOG_EVERY}" \
    --seed "${SEED}" \
    --tensorboard-dir "${OUT_DIR}/tensorboard" \
    --nitp-loss-weight "${NITP_LOSS_WEIGHT}" \
    --nitp-hidden-dim 0 \
    --nitp-max-targets 256 \
    > "${LOG_FILE}" 2>&1 &

echo "STAGE94_TOKENIZER_FREE_BYTE_LAUNCHED:$!"
echo "LOG:${LOG_FILE}"
echo "OUT:${OUT_DIR}"
echo "SAMPLED_DATA:${SAMPLED_DATA}"
