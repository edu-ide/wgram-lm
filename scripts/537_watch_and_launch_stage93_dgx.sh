#!/usr/bin/env bash
set -euo pipefail

# Watch Stage93 Data-IO binding and launch continuation as soon as the sampled
# token file is complete.
#
# Plain-language contract:
#   tokenization is printing pages;
#   sampled/tokens.npy is the bound textbook;
#   this script starts class as soon as the bound textbook is stable.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT}"

SAMPLED_DATA="${SAMPLED_DATA:-${ROOT}/local_eval/stage93_hrm_text_reasoning_nonflan_dataio/sampled}"
TOKENS_FILE="${TOKENS_FILE:-${SAMPLED_DATA}/tokens.npy}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-${ROOT}/scripts/536_launch_stage93_dgx_continue_prefixlm.sh}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABILITY_SECONDS="${STABILITY_SECONDS:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-0}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

trainer_active() {
  pgrep -af "scripts/534_train_native_prefixlm_dataio" >/dev/null 2>&1
}

tokens_file_stable() {
  if [[ ! -f "${TOKENS_FILE}" ]]; then
    return 1
  fi

  local size_a size_b
  size_a="$(stat -c '%s' "${TOKENS_FILE}")"
  if [[ "${size_a}" == "0" ]]; then
    return 1
  fi

  sleep "${STABILITY_SECONDS}"

  if [[ ! -f "${TOKENS_FILE}" ]]; then
    return 1
  fi
  size_b="$(stat -c '%s' "${TOKENS_FILE}")"

  [[ "${size_a}" == "${size_b}" && "${size_b}" != "0" ]]
}

main() {
  if [[ ! -f "${LAUNCH_SCRIPT}" ]]; then
    log "missing launch script: ${LAUNCH_SCRIPT}"
    exit 2
  fi

  local start elapsed
  start="$(date +%s)"

  log "watching Stage93 binding"
  log "root=${ROOT}"
  log "tokens_file=${TOKENS_FILE}"
  log "launch_script=${LAUNCH_SCRIPT}"
  log "poll_seconds=${POLL_SECONDS}"
  log "stability_seconds=${STABILITY_SECONDS}"
  log "max_wait_seconds=${MAX_WAIT_SECONDS}"

  while true; do
    if tokens_file_stable; then
      log "sampled tokens file is present and stable"
      while trainer_active; do
        log "PrefixLM trainer already active; waiting"
        sleep "${POLL_SECONDS}"
      done

      log "launching Stage93 continuation"
      bash "${LAUNCH_SCRIPT}"
      log "watcher done"
      return 0
    fi

    if [[ "${MAX_WAIT_SECONDS}" != "0" ]]; then
      elapsed="$(( $(date +%s) - start ))"
      if (( elapsed >= MAX_WAIT_SECONDS )); then
        log "max wait reached without stable sampled tokens"
        exit 4
      fi
    fi

    log "sampled tokens not ready yet"
    sleep "${POLL_SECONDS}"
  done
}

main "$@"
