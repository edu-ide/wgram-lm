#!/usr/bin/env bash
set -euo pipefail

# Run Stage93 micro language gates after the micro target checkpoint is ready.
#
# Plain-language contract:
#   After the student finishes this micro handout, ask ordinary language
#   questions before declaring progress. If another training run already owns
#   the GPU, wait instead of stealing it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT}"

TRAIN_LOG="${TRAIN_LOG:-/tmp/20260524_STAGE93A00_DGX913M_MICRO_HARDLINK_TO24500.log}"
TARGET_STEPS="${TARGET_STEPS:-40000}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_SECONDS="${MAX_SECONDS:-28800}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/local_eval/20260524_STAGE93A00_DGX913M_MICRO_HARDLINK_TO24500/last_model.pt}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${ROOT}/local_eval/20260524_STAGE93A00_DGX913M_MICRO_HARDLINK_TO24500/tensorboard}"
OUT_DIR="${OUT_DIR:-${ROOT}/local_eval/20260524_STAGE93A00_MICRO_LANGUAGE_GATES}"
WATCH_LOG="${WATCH_LOG:-/tmp/20260524_STAGE93A00_micro_language_gates_on_target.log}"
GATES_LOG="${GATES_LOG:-/tmp/20260524_STAGE93A00_micro_language_gates.log}"
STABLE_SECONDS="${STABLE_SECONDS:-90}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${WATCH_LOG}" >/dev/null
}

current_step() {
  [[ -f "${TRAIN_LOG}" ]] || {
    printf '0\n'
    return 0
  }
  python3 - "${TRAIN_LOG}" <<'PY' 2>/dev/null || printf '0\n'
from collections import deque
import json
import re
import sys

steps = []
try:
    lines = deque(open(sys.argv[1], encoding="utf-8", errors="replace"), maxlen=5000)
except Exception:
    lines = []

for line in lines:
    line = line.strip()
    if not line:
        continue
    try:
        item = json.loads(line)
    except Exception:
        match = re.search(r'"step"\s*:\s*(\d+)', line)
        if match:
            steps.append(int(match.group(1)))
        continue
    if isinstance(item, dict) and isinstance(item.get("step"), int):
        steps.append(item["step"])

print(max(steps) if steps else 0)
PY
}

checkpoint_stable() {
  local now mtime age
  [[ -s "${CHECKPOINT}" ]] || return 1
  now="$(date +%s)"
  mtime="$(stat -c '%Y' "${CHECKPOINT}")"
  age="$((now - mtime))"
  (( age >= STABLE_SECONDS ))
}

gates_done() {
  [[ -s "${OUT_DIR}/language_heldout_loss.json" ]] \
    && [[ -s "${OUT_DIR}/general_language_heldout_loss.json" ]] \
    && [[ -s "${OUT_DIR}/general_language_generation_probe.json" ]] \
    && [[ -s "${OUT_DIR}/multilingual_generation_probe.json" ]] \
    && [[ -s "${OUT_DIR}/raw_intelligence_suite.json" ]]
}

main() {
  local start now elapsed step rc
  : > "${WATCH_LOG}"
  start="$(date +%s)"
  log "micro language gate watcher started target_steps=${TARGET_STEPS}"

  while true; do
    if gates_done; then
      log "language gates already complete: ${OUT_DIR}"
      return 0
    fi

    now="$(date +%s)"
    elapsed="$((now - start))"
    step="$(current_step)"
    if (( elapsed >= MAX_SECONDS )); then
      log "max_seconds reached without gates; step=${step}"
      return 3
    fi

    if ! [[ "${step}" =~ ^[0-9]+$ ]] || (( step < TARGET_STEPS )); then
      log "waiting for target; step=${step}; elapsed=${elapsed}s"
      sleep "${POLL_SECONDS}"
      continue
    fi

    if ! checkpoint_stable; then
      log "target reached but checkpoint not stable yet; step=${step}"
      sleep "${POLL_SECONDS}"
      continue
    fi

    log "running language gates from ${CHECKPOINT}; step=${step}"
    set +e
    CHECKPOINT="${CHECKPOINT}" \
      OUT_DIR="${OUT_DIR}" \
      TENSORBOARD_DIR="${TENSORBOARD_DIR}" \
      bash "${ROOT}/scripts/545_run_prefixlm_language_gates_dgx.sh" \
      > "${GATES_LOG}" 2>&1
    rc="$?"
    set -e

    if [[ "${rc}" == "0" ]]; then
      log "language gates complete: ${OUT_DIR}"
      return 0
    fi
    if [[ "${rc}" == "3" ]]; then
      log "language gates deferred because training is active"
      sleep "${POLL_SECONDS}"
      continue
    fi
    log "language gates failed rc=${rc}; see ${GATES_LOG}"
    return "${rc}"
  done
}

main "$@"
