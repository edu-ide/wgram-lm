#!/usr/bin/env bash
set -euo pipefail

# Launch Stage93 as continued training from the Stage92 learned checkpoint.
#
# Plain-language contract:
#   Stage92 is the student who has already learned the first handout.
#   Stage93 gives the same student a much larger textbook.
#   By default we keep the student's weights and restart optimizer state,
#   because the data distribution changes from a tiny shard to a large shard.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT}"

RUN_NAME="${RUN_NAME:-20260523_STAGE93_DGX913M_CONTINUE_LARGE_DATAIO}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/local_eval/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-/tmp/${RUN_NAME}.log}"

SAMPLED_DATA="${SAMPLED_DATA:-${ROOT}/local_eval/stage93_hrm_text_reasoning_nonflan_dataio/sampled}"
RESUME="${RESUME:-${ROOT}/local_eval/20260523_STAGE92_DGX913M_BS8_CONTINUE_TO24K/last_model.pt}"
PYTHON="${PYTHON:-/mnt/data4tb/venv_sglang_pr23000/bin/python}"

# The trainer interprets --steps as the absolute final step after resume.
# If RESUME was saved at step 24000 and STEPS=60000, this runs 36000 more steps.
STEPS="${STEPS:-60000}"
SEED="${SEED:-9301}"
OPTIMIZER="${OPTIMIZER:-galore_adamw8bit}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LEN="${SEQ_LEN:-128}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-8000}"
MODEL_CHECKPOINT_EVERY="${MODEL_CHECKPOINT_EVERY:-2000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
LOG_EVERY="${LOG_EVERY:-50}"
LR="${LR:-1.6e-4}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2000}"
REQUIRED_TRITON_PTXAS_PATH="${REQUIRED_TRITON_PTXAS_PATH:-}"

if [[ -z "${REQUIRED_TRITON_PTXAS_PATH:-}" ]]; then
  echo "missing required ptxas contract: set REQUIRED_TRITON_PTXAS_PATH explicitly" >&2
  exit 5
fi
if [[ -z "${TRITON_PTXAS_PATH:-}" ]]; then
  echo "missing required ptxas: set TRITON_PTXAS_PATH=${REQUIRED_TRITON_PTXAS_PATH}" >&2
  exit 5
fi
if [[ "${TRITON_PTXAS_PATH}" != "${REQUIRED_TRITON_PTXAS_PATH}" ]]; then
  echo "wrong ptxas: TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH}, required=${REQUIRED_TRITON_PTXAS_PATH}" >&2
  exit 5
fi
if [[ ! -x "${TRITON_PTXAS_PATH}" ]]; then
  echo "missing required ptxas: ${TRITON_PTXAS_PATH}" >&2
  exit 5
fi

if [[ ! -d "${SAMPLED_DATA}" || ! -f "${SAMPLED_DATA}/tokens.npy" ]]; then
  echo "sampled data is not ready: ${SAMPLED_DATA}" >&2
  echo "finish scripts/535_prepare_stage93_hrm_text_large_dataio.sh first" >&2
  exit 2
fi

if [[ ! -f "${RESUME}" ]]; then
  echo "resume checkpoint missing: ${RESUME}" >&2
  exit 2
fi

if [[ "${FORCE:-0}" != "1" ]]; then
  active_train="$(pgrep -af "python .*scripts/534_train_native_prefixlm_dataio" 2>/dev/null | awk '$2 ~ /python/ {print}')"
  if [[ -n "${active_train}" ]]; then
    echo "REFUSING_TO_LAUNCH: another PrefixLM training process is already running." >&2
    echo "Set FORCE=1 to launch anyway, or stop/wait for the active run." >&2
    printf '%s\n' "${active_train}" >&2
    exit 3
  fi
fi

mkdir -p "${OUT_ROOT}"

TRAIN_SCRIPT="scripts/534_train_native_prefixlm_dataio.py"
if [[ -f "scripts/534_train_native_prefixlm_dataio_stage90.py" ]]; then
  TRAIN_SCRIPT="scripts/534_train_native_prefixlm_dataio_stage90.py"
fi

CMD=(
  "${PYTHON}" "${TRAIN_SCRIPT}"
  --sampled-data "${SAMPLED_DATA}"
  --out-dir "${OUT_ROOT}"
  --resume "${RESUME}"
  --device cuda
  --steps "${STEPS}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --model-checkpoint-every "${MODEL_CHECKPOINT_EVERY}"
  --batch-size "${BATCH_SIZE}"
  --seq-len "${SEQ_LEN}"
  --d-model 1792
  --n-heads 16
  --n-kv-heads 4
  --d-ff 4864
  --train-think-steps 2
  --length-bucketed-batches
  --trim-batch-to-max-length
  --loss-kernel auto
  --optimizer "${OPTIMIZER}"
  --amp-dtype bf16
  --matmul-precision high
  --lr "${LR}"
  --lr-warmup-steps "${LR_WARMUP_STEPS}"
  --adam-beta1 0.9
  --adam-beta2 0.95
  --weight-decay 0.1
  --eval-every "${EVAL_EVERY}"
  --eval-max-rows 128
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --eval-max-batches 0
  --log-every "${LOG_EVERY}"
  --seed "${SEED}"
  --tensorboard-dir "${OUT_ROOT}/tensorboard"
)

setsid env \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=src \
  QTRM_AIM_REPO="${QTRM_AIM_REPO:-}" \
  TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH}" \
  "${CMD[@]}" > "${LOG_FILE}" 2>&1 < /dev/null &

echo "STAGE93_CONTINUE_LAUNCHED:${!}"
echo "LOG:${LOG_FILE}"
echo "OUT:${OUT_ROOT}"
echo "SAMPLED_DATA:${SAMPLED_DATA}"
echo "RESUME:${RESUME}"
