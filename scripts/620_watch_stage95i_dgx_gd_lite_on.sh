#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/data4tb/qtrm_multimodal_memoryos}"
PYTHON="${PYTHON:-/mnt/data4tb/venv_sglang_pr23000/bin/python}"
WORK_BASE="${WORK_BASE:-${ROOT}/local_eval}"
RUN_NAME="${RUN_NAME:-20260525_STAGE95I_DGX_1B_OPUS_GD_OFFICIAL_GDN2_ONEBODY_FULL}"
FULL_SAMPLE="${FULL_SAMPLE:-${WORK_BASE}/stage95_blt_foundation_byte_curriculum_broad_240k_opus_gd/sampled}"
FULL_OUT="${FULL_OUT:-${WORK_BASE}/${RUN_NAME}}"
LOG="${LOG:-/tmp/${RUN_NAME}.log}"
STALE_SEC="${STALE_SEC:-1800}"
SLEEP_SEC="${SLEEP_SEC:-120}"

cd "${ROOT}"

trainer_pids() {
  ps -eo pid=,args= |
    awk -v run_name="${RUN_NAME}" '
      $0 ~ /python/ &&
      $0 ~ /scripts\/557_train_blt_d_prefixlm_dataio.py/ &&
      $0 ~ run_name {
        print $1
      }
    '
}

relaunch() {
  env \
    PYTHON="${PYTHON}" \
    ROOT="${ROOT}" \
    WORK_BASE="${WORK_BASE}" \
    REQUIRED_TRITON_PTXAS_PATH="${REQUIRED_TRITON_PTXAS_PATH:-/usr/local/cuda-13.2/bin/ptxas}" \
    TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda-13.2/bin/ptxas}" \
    FULL_SELECTION_MODE=utility \
    FULL_SAMPLE="${FULL_SAMPLE}" \
    FULL_OUT="${FULL_OUT}" \
    FULL_STEPS="${FULL_STEPS:-10000}" \
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-400}" \
    SAVE_OPTIMIZER_CHECKPOINT="${SAVE_OPTIMIZER_CHECKPOINT:-1}" \
    OPTIMIZER_CHECKPOINT_EVERY="${OPTIMIZER_CHECKPOINT_EVERY:-1200}" \
    RESUME_LOAD_OPTIMIZER="${RESUME_LOAD_OPTIMIZER:-auto}" \
    FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-32}" \
    FULL_EVAL_BATCH_SIZE="${FULL_EVAL_BATCH_SIZE:-1}" \
    EVAL_EVERY="${EVAL_EVERY:-20000}" \
    EVAL_MAX_ROWS="${EVAL_MAX_ROWS:-64}" \
    LR="${LR:-1.1e-4}" \
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-300}" \
    TEACHER_DISTILL_WEIGHT="${TEACHER_DISTILL_WEIGHT:-0.0}" \
    ACKNOWLEDGE_PAST_SUCCESS_RESTORATION_GAP=1 \
    GD_LITE_ENABLED=1 \
    bash scripts/559_run_stage95_blt_partial_then_full_dgx.sh launch-full
}

while true; do
  cd "${ROOT}" || exit 2
  if [[ -f "${FULL_OUT}/report.json" ]]; then
    echo "$(date -Is) report_done"
    exit 0
  fi

  now="$(date +%s)"
  log_mtime=0
  if [[ -f "${LOG}" ]]; then
    log_mtime="$(stat -c %Y "${LOG}" 2>/dev/null || echo 0)"
  fi
  age="$((now - log_mtime))"
  pids="$(trainer_pids | xargs echo || true)"

  if [[ -n "${pids}" && "${age}" -le "${STALE_SEC}" ]]; then
    echo "$(date -Is) trainer_alive_progress_fresh age=${age}s pids=${pids}"
  else
    if [[ -n "${pids}" ]]; then
      echo "$(date -Is) trainer_stale_kill age=${age}s pids=${pids}"
      echo "${pids}" | xargs -r kill || true
      sleep 20
      echo "${pids}" | xargs -r kill -9 2>/dev/null || true
    else
      echo "$(date -Is) trainer_missing_relaunch_optimizer_aware_gd_lite_on age=${age}s"
    fi
    relaunch
  fi

  sleep "${SLEEP_SEC}"
done
