#!/usr/bin/env bash
set -euo pipefail

# Run the 82M latent-first ablation on the normal HRM-Text/Data-IO PrefixLM path.
# MODE=baseline keeps plain CE. MODE=nitp adds the NITP-style latent target loss.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT}"

MODE="${MODE:-nitp}"
case "${MODE}" in
  baseline|nitp) ;;
  *)
    echo "MODE must be baseline or nitp, got: ${MODE}" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-20260524_STAGE94_${MODE^^}_82M_LATENT_FIRST}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/local_eval/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-/tmp/${RUN_NAME}.log}"
SAMPLED_DATA="${SAMPLED_DATA:-${ROOT}/local_eval/stage93_hrm_text_general_language_curriculum_dataio/sampled}"
PYTHON="${PYTHON:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-2000}"
SEED="${SEED:-9401}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQ_LEN="${SEQ_LEN:-128}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
MODEL_CHECKPOINT_EVERY="${MODEL_CHECKPOINT_EVERY:-1000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
LOG_EVERY="${LOG_EVERY:-25}"
OPTIMIZER="${OPTIMIZER:-adamw}"
NITP_LOSS_WEIGHT="${NITP_LOSS_WEIGHT:-0.05}"
NITP_HIDDEN_DIM="${NITP_HIDDEN_DIM:-0}"
NITP_MAX_TARGETS="${NITP_MAX_TARGETS:-256}"

if [[ ! -d "${SAMPLED_DATA}" || ! -f "${SAMPLED_DATA}/tokens.npy" ]]; then
  echo "sampled data is not ready: ${SAMPLED_DATA}" >&2
  echo "Set SAMPLED_DATA to a ready Data-IO sampled directory, or wait for Stage93B prep." >&2
  exit 2
fi

if [[ "${FORCE:-0}" != "1" ]]; then
  active_train="$(pgrep -af "python .*scripts/534_train_native_prefixlm_dataio" 2>/dev/null || true)"
  if [[ -n "${active_train}" ]]; then
    echo "REFUSING_TO_LAUNCH: another PrefixLM training process is already running." >&2
    echo "Set FORCE=1 to launch anyway, or stop/wait for the active run." >&2
    printf '%s\n' "${active_train}" >&2
    exit 3
  fi
fi

mkdir -p "${OUT_ROOT}"

EXTRA_FLAGS=()
if [[ "${MODE}" == "nitp" ]]; then
  EXTRA_FLAGS+=(
    --nitp-loss-weight "${NITP_LOSS_WEIGHT}"
    --nitp-hidden-dim "${NITP_HIDDEN_DIM}"
    --nitp-max-targets "${NITP_MAX_TARGETS}"
  )
fi

CMD=(
  "${PYTHON}" scripts/534_train_native_prefixlm_dataio.py
  --sampled-data "${SAMPLED_DATA}"
  --out-dir "${OUT_ROOT}"
  --device "${DEVICE}"
  --steps "${STEPS}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --model-checkpoint-every "${MODEL_CHECKPOINT_EVERY}"
  --batch-size "${BATCH_SIZE}"
  --seq-len "${SEQ_LEN}"
  --d-model 384
  --n-heads 6
  --n-kv-heads 2
  --d-ff 1024
  --train-think-steps 2
  --length-bucketed-batches
  --trim-batch-to-max-length
  --loss-kernel auto
  --optimizer "${OPTIMIZER}"
  --amp-dtype bf16
  --matmul-precision high
  --lr 2.2e-4
  --lr-warmup-steps 500
  --adam-beta1 0.9
  --adam-beta2 0.95
  --weight-decay 0.1
  --eval-every "${EVAL_EVERY}"
  --eval-max-rows 256
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --eval-max-batches 0
  --log-every "${LOG_EVERY}"
  --seed "${SEED}"
  --tensorboard-dir "${OUT_ROOT}/tensorboard"
  "${EXTRA_FLAGS[@]}"
)

setsid env PYTHONUNBUFFERED=1 PYTHONPATH=src "${CMD[@]}" > "${LOG_FILE}" 2>&1 < /dev/null &

echo "STAGE94_82M_${MODE^^}_LAUNCHED:${!}"
echo "LOG:${LOG_FILE}"
echo "OUT:${OUT_ROOT}"
echo "SAMPLED_DATA:${SAMPLED_DATA}"
