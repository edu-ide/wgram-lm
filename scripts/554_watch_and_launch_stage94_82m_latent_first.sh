#!/usr/bin/env bash
set -euo pipefail

# Wait for a Data-IO sampled directory, then launch the 82M baseline and NITP
# latent-first ablations sequentially. Intended for overnight handoff.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT}"

SAMPLED_DATA="${SAMPLED_DATA:-${ROOT}/local_eval/stage93_hrm_text_general_language_curriculum_dataio/sampled}"
PYTHON="${PYTHON:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-120}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-0}"
BASELINE_STEPS="${BASELINE_STEPS:-2000}"
NITP_STEPS="${NITP_STEPS:-2000}"
BASELINE_SEED="${BASELINE_SEED:-9401}"
NITP_SEED="${NITP_SEED:-9401}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-20260524_STAGE94A_BPE82M_BASELINE}"
NITP_RUN_NAME="${NITP_RUN_NAME:-20260524_STAGE94B_BPE82M_NITP}"
BASELINE_LOG_FILE="${BASELINE_LOG_FILE:-/tmp/${BASELINE_RUN_NAME}.log}"
NITP_LOG_FILE="${NITP_LOG_FILE:-/tmp/${NITP_RUN_NAME}.log}"
LAUNCHER="${LAUNCHER:-${ROOT}/scripts/553_run_stage94_latent_first_82m_ablation.sh}"
WAIT_FOR_NO_ACTIVE_TRAIN="${WAIT_FOR_NO_ACTIVE_TRAIN:-1}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

sampled_ready() {
  [[ -d "${SAMPLED_DATA}" ]] \
    && [[ -f "${SAMPLED_DATA}/tokens.npy" ]] \
    && [[ -f "${SAMPLED_DATA}/metadata.json" ]] \
    && [[ -d "${SAMPLED_DATA}/epoch_0" ]]
}

wait_for_sampled() {
  local waited=0
  while ! sampled_ready; do
    log "waiting for sampled data: ${SAMPLED_DATA}; elapsed=${waited}s"
    if [[ "${MAX_WAIT_SECONDS}" != "0" && "${waited}" -ge "${MAX_WAIT_SECONDS}" ]]; then
      log "timeout waiting for sampled data"
      exit 2
    fi
    sleep "${CHECK_INTERVAL_SECONDS}"
    waited=$((waited + CHECK_INTERVAL_SECONDS))
  done
  log "sampled data ready: ${SAMPLED_DATA}"
}

active_prefixlm_train() {
  pgrep -af "python .*scripts/534_train_native_prefixlm_dataio" 2>/dev/null || true
}

wait_for_no_active_train() {
  if [[ "${WAIT_FOR_NO_ACTIVE_TRAIN}" != "1" ]]; then
    return 0
  fi
  while true; do
    local active
    active="$(active_prefixlm_train)"
    if [[ -z "${active}" ]]; then
      return 0
    fi
    log "waiting for active PrefixLM training to finish before Stage94 launch"
    printf '%s\n' "${active}"
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
}

wait_for_log_done() {
  local log_file="$1"
  local mode_name="$2"
  while true; do
    if grep -q '"decision":' "${log_file}" 2>/dev/null; then
      log "${mode_name} completed according to report output"
      return 0
    fi
    if ! pgrep -af "scripts/534_train_native_prefixlm_dataio.py" | grep -F "${mode_name}" >/dev/null 2>&1; then
      if grep -q '"loss_history"' "${log_file}" 2>/dev/null; then
        log "${mode_name} process exited after writing report"
        return 0
      fi
      log "${mode_name} process appears stopped before completion; inspect ${log_file}"
      return 1
    fi
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
}

launch_one() {
  local mode="$1"
  local run_name="$2"
  local steps="$3"
  local seed="$4"
  local log_file="$5"
  log "launching Stage94 ${mode}: ${run_name}"
  MODE="${mode}" \
    RUN_NAME="${run_name}" \
    LOG_FILE="${log_file}" \
    SAMPLED_DATA="${SAMPLED_DATA}" \
    PYTHON="${PYTHON}" \
    DEVICE="${DEVICE}" \
    STEPS="${steps}" \
    SEED="${seed}" \
    FORCE=0 \
    "${LAUNCHER}"
}

wait_for_sampled
wait_for_no_active_train
launch_one baseline "${BASELINE_RUN_NAME}" "${BASELINE_STEPS}" "${BASELINE_SEED}" "${BASELINE_LOG_FILE}"
wait_for_log_done "${BASELINE_LOG_FILE}" "${BASELINE_RUN_NAME}"
wait_for_no_active_train
launch_one nitp "${NITP_RUN_NAME}" "${NITP_STEPS}" "${NITP_SEED}" "${NITP_LOG_FILE}"
wait_for_log_done "${NITP_LOG_FILE}" "${NITP_RUN_NAME}"
log "Stage94 baseline and NITP ablations finished"
