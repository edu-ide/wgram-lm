#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/tripleyoung/qtrm-workspace/qtrm_multimodal_memoryos}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/local_eval/20260526_STAGE105A_LOCAL_BPE_BROAD_HRM_TEXT_SPLIT}"
RESUME="${RESUME:-${ROOT}/local_eval/20260526_STAGE105A_LOCAL_BPE_BROAD_HRM_TEXT_120/last.pt}"
OUT="${OUT:-${ROOT}/local_eval/20260526_STAGE105B_LOCAL_BPE_BROAD_HRM_TEXT_CONT600_ALLROWS}"
LOG="${LOG:-/tmp/20260526_STAGE105B_LOCAL_BPE_BROAD_HRM_TEXT_CONT600_ALLROWS.log}"
STEPS="${STEPS:-600}"
TRAIN_MAX_ROWS="${TRAIN_MAX_ROWS:-0}"
EVAL_MAX_ROWS="${EVAL_MAX_ROWS:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
LOSS_CHUNK_SIZE="${LOSS_CHUNK_SIZE:-128}"
PTXAS="${PTXAS:-/usr/local/cuda-12.8/bin/ptxas}"

if [[ ! -x "${PTXAS}" ]]; then
  echo "missing local official-GDN2 ptxas: ${PTXAS}" >&2
  exit 2
fi
if [[ ! -f "${RESUME}" ]]; then
  echo "missing resume checkpoint: ${RESUME}" >&2
  exit 2
fi

cd "${ROOT}"
mkdir -p "${OUT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export REQUIRED_TRITON_PTXAS_PATH="${PTXAS}"
export TRITON_PTXAS_PATH="${PTXAS}"

exec "${PYTHON}" scripts/534_train_native_prefixlm_dataio.py \
  --sampled-data "${DATA_ROOT}/train_data/sampled" \
  --eval-sampled-data "${DATA_ROOT}/eval_data/sampled" \
  --out-dir "${OUT}" \
  --resume "${RESUME}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-len 512 \
  --max-rows "${TRAIN_MAX_ROWS}" \
  --eval-max-rows "${EVAL_MAX_ROWS}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --eval-max-batches 0 \
  --eval-every 120 \
  --model-checkpoint-every 120 \
  --checkpoint-every 0 \
  --lr 2.2e-4 \
  --lr-warmup-steps 60 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --amp-dtype bf16 \
  --matmul-precision high \
  --loss-chunk-size "${LOSS_CHUNK_SIZE}" \
  --length-bucketed-batches \
  --d-model 384 \
  --n-heads 6 \
  --n-kv-heads 2 \
  --d-ff 1024 \
  --backbone trm_qwen35_3to1 \
  --think-structure trm_dual_z \
  --train-think-steps 4 \
  --delta-backend official_gated_delta2 \
  --strict-backends \
  --attention-backend sdpa \
  --tensorboard-dir "${OUT}/tensorboard" \
  --seed 10502 \
  "$@" \
  > "${LOG}" 2>&1
