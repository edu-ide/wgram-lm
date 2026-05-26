#!/usr/bin/env bash
set -uo pipefail

# Relaunch the Stage93 overnight supervisor if it exits unexpectedly.
# This wrapper stops only after the full Stage93 target step has been reached.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT}" || exit 2

FULL_RUN_NAME="${FULL_RUN_NAME:-20260524_STAGE93B_DGX913M_FULL_FROM_BEST_TO60K}"
FULL_OUT="${FULL_OUT:-${ROOT}/local_eval/${FULL_RUN_NAME}}"
FULL_TARGET_STEPS="${FULL_TARGET_STEPS:-60000}"
RESTART_SECONDS="${RESTART_SECONDS:-60}"
WRAPPER_LOG="${WRAPPER_LOG:-/tmp/20260524_STAGE93_supervisor_forever.log}"
RUN_LANGUAGE_GATES_ON_TARGET="${RUN_LANGUAGE_GATES_ON_TARGET:-1}"
LANGUAGE_GATES_RUN_NAME="${LANGUAGE_GATES_RUN_NAME:-20260524_STAGE93B_LANGUAGE_GATES}"
LANGUAGE_GATES_OUT_DIR="${LANGUAGE_GATES_OUT_DIR:-${ROOT}/local_eval/${LANGUAGE_GATES_RUN_NAME}}"
LANGUAGE_GATES_LOG="${LANGUAGE_GATES_LOG:-/tmp/${LANGUAGE_GATES_RUN_NAME}.log}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${WRAPPER_LOG}" >/dev/null
}

report_step() {
  local report="$1"
  [[ -f "${report}" ]] || {
    printf '0\n'
    return 0
  }
  python3 - "${report}" <<'PY' 2>/dev/null || printf '0\n'
import json
import sys

steps = []

def walk(x):
    if isinstance(x, dict):
        step = x.get("step")
        if isinstance(step, int):
            steps.append(step)
        for value in x.values():
            walk(value)
    elif isinstance(x, list):
        for value in x:
            walk(value)

try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        walk(json.load(f))
except Exception:
    pass

print(max(steps) if steps else 0)
PY
}

target_reached() {
  local step
  step="$(report_step "${FULL_OUT}/report.json")"
  [[ "${step}" =~ ^[0-9]+$ ]] || step=0
  (( step >= FULL_TARGET_STEPS ))
}

full_checkpoint_for_gates() {
  local ckpt
  for ckpt in "${FULL_OUT}/last_model.pt" "${FULL_OUT}/last.pt"; do
    if [[ -s "${ckpt}" ]]; then
      printf '%s\n' "${ckpt}"
      return 0
    fi
  done
  return 1
}

language_gates_done() {
  [[ -s "${LANGUAGE_GATES_OUT_DIR}/language_heldout_loss.json" ]] \
    && [[ -s "${LANGUAGE_GATES_OUT_DIR}/general_language_heldout_loss.json" ]] \
    && [[ -s "${LANGUAGE_GATES_OUT_DIR}/general_language_generation_probe.json" ]] \
    && [[ -s "${LANGUAGE_GATES_OUT_DIR}/multilingual_generation_probe.json" ]] \
    && [[ -s "${LANGUAGE_GATES_OUT_DIR}/raw_intelligence_suite.json" ]]
}

run_language_gates_once() {
  local ckpt rc
  if [[ "${RUN_LANGUAGE_GATES_ON_TARGET}" != "1" ]]; then
    log "language gates disabled; skipping"
    return 0
  fi
  if language_gates_done; then
    log "language/raw-intelligence gates already complete: ${LANGUAGE_GATES_OUT_DIR}"
    return 0
  fi
  ckpt="$(full_checkpoint_for_gates)" || {
    log "language gates skipped: no full checkpoint is available yet"
    return 3
  }
  log "running language/raw-intelligence gates from ${ckpt}"
  CHECKPOINT="${ckpt}" \
    OUT_DIR="${LANGUAGE_GATES_OUT_DIR}" \
    TENSORBOARD_DIR="${FULL_OUT}/tensorboard" \
    bash "${ROOT}/scripts/545_run_prefixlm_language_gates_dgx.sh" \
    > "${LANGUAGE_GATES_LOG}" 2>&1
  rc="$?"
  if [[ "${rc}" == "0" ]]; then
    log "language/raw-intelligence gates complete: ${LANGUAGE_GATES_OUT_DIR}"
    return 0
  fi
  if [[ "${rc}" == "3" ]]; then
    log "language gates deferred because training is still active"
    return 3
  fi
  log "language gates failed rc=${rc}; see ${LANGUAGE_GATES_LOG}"
  return "${rc}"
}

exit_if_target_ready() {
  if target_reached; then
    if run_language_gates_once; then
      log "full target reached; wrapper exiting"
      return 0
    fi
    log "full target reached but language gates are deferred; waiting ${RESTART_SECONDS}s"
    sleep "${RESTART_SECONDS}"
  fi
  return 1
}

main() {
  : > "${WRAPPER_LOG}"
  log "Stage93 supervisor wrapper started"
  log "root=${ROOT}"
  log "full_out=${FULL_OUT}"
  log "full_target_steps=${FULL_TARGET_STEPS}"
  log "language_gates_out=${LANGUAGE_GATES_OUT_DIR}"

  while true; do
    if exit_if_target_ready; then
      return 0
    fi

    log "starting supervisor"
    bash "${ROOT}/scripts/539_supervise_stage93_overnight.sh"
    rc="$?"

    if exit_if_target_ready; then
      return 0
    fi

    log "supervisor exited rc=${rc}; restarting after ${RESTART_SECONDS}s"
    sleep "${RESTART_SECONDS}"
  done
}

main "$@"
