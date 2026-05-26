#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_NAME="${RUN_NAME:-STAGE86_LOCAL913M_OPTIMIZED_SMOKE}"
OUT_ROOT="${OUT_ROOT:-/tmp/qtrm_eval/20260523_STAGE86_LOCAL913M_OPTIMIZED_SMOKE}"
LOG_FILE="${LOG_FILE:-/tmp/20260523_STAGE86_LOCAL913M_OPTIMIZED_SMOKE.log}"
SAMPLED_DATA="${SAMPLED_DATA:-/tmp/hrm_text_dataio_sample_stage66_dataio_preflight_20260523/sampled}"
STEPS="${STEPS:-5000}"
SEED="${SEED:-8601}"
OPTIMIZER="${OPTIMIZER:-galore_adamw8bit}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-128}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
LOG_EVERY="${LOG_EVERY:-50}"

if [[ "${FORCE:-0}" != "1" ]]; then
  if pgrep -af "scripts/534_train_native_prefixlm_dataio.py" >/dev/null; then
    echo "REFUSING_TO_LAUNCH: another PrefixLM training process is already running." >&2
    echo "Set FORCE=1 to launch anyway, or stop/wait for the active run." >&2
    pgrep -af "scripts/534_train_native_prefixlm_dataio.py" >&2 || true
    exit 3
  fi
fi

mkdir -p "${OUT_ROOT}"

EXTRA_FLAGS=()
if [[ "${ACTIVATION_CHECKPOINTING}" == "1" ]]; then
  EXTRA_FLAGS+=(--activation-checkpointing)
fi

CMD=(
  .venv/bin/python scripts/534_train_native_prefixlm_dataio.py
  --sampled-data "${SAMPLED_DATA}"
  --out-dir "${OUT_ROOT}"
  --device cuda
  --steps "${STEPS}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --batch-size "${BATCH_SIZE}"
  --seq-len "${SEQ_LEN}"
  --d-model 1792
  --n-heads 16
  --n-kv-heads 4
  --d-ff 4864
  --train-think-steps 2
  --length-bucketed-batches
  --trim-batch-to-max-length
  "${EXTRA_FLAGS[@]}"
  --loss-kernel auto
  --optimizer "${OPTIMIZER}"
  --amp-dtype bf16
  --matmul-precision high
  --lr 2.2e-4
  --lr-warmup-steps 2000
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
  --aim-repo "${QTRM_AIM_REPO:-/tmp/qtrm_aim_stage86}"
  --aim-experiment stage86_local913m_optimized_smoke
  --aim-run-name "${RUN_NAME}"
  --aim-description "913M optimized 4090 smoke: bucket trim Liger CCE ${OPTIMIZER}; activation_checkpointing=${ACTIVATION_CHECKPOINTING}"
)

setsid env PYTHONUNBUFFERED=1 "${CMD[@]}" > "${LOG_FILE}" 2>&1 < /dev/null &

echo "LAUNCHED:${!}"
echo "LOG:${LOG_FILE}"
echo "OUT:${OUT_ROOT}"
