#!/usr/bin/env bash
set -euo pipefail

# Keep copy-safe checkpoint aliases fresh for an already-running trainer.
#
# The trainer now writes these aliases directly, but an old in-memory process
# cannot pick up changed Python code. This watcher is a low-I/O bridge: it waits
# until a checkpoint file is stable, then publishes copy_last*.pt by hardlink.

OUT_DIR="${OUT_DIR:?set OUT_DIR to the training output directory}"
LOG_FILE="${LOG_FILE:-}"
TARGET_STEP="${TARGET_STEP:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABILITY_SECONDS="${STABILITY_SECONDS:-5}"
MAX_SECONDS="${MAX_SECONDS:-14400}"
WATCH_LOG="${WATCH_LOG:-/tmp/$(basename "${OUT_DIR}")_copy_checkpoint_watcher.log}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${WATCH_LOG}" >/dev/null
}

publish_alias() {
  local name="$1"
  local src="${OUT_DIR}/${name}"
  local dst="${OUT_DIR}/copy_${name}"
  local tmp="${OUT_DIR}/.copy_${name}.tmp.$$"
  local before after

  [[ -s "${src}" ]] || return 0
  before="$(stat -c '%s:%Y' "${src}" 2>/dev/null || true)"
  sleep "${STABILITY_SECONDS}"
  after="$(stat -c '%s:%Y' "${src}" 2>/dev/null || true)"
  [[ -n "${before}" && "${before}" == "${after}" ]] || return 0
  if [[ -e "${dst}" && "${src}" -ef "${dst}" ]]; then
    log "${dst} already points at current ${name} (${after})"
    return 0
  fi

  rm -f "${tmp}"
  ln "${src}" "${tmp}" 2>/dev/null || cp -p "${src}" "${tmp}"
  mv -f "${tmp}" "${dst}"
  log "published ${dst} from ${name} (${after})"
}

latest_log_step() {
  [[ -n "${LOG_FILE}" && -f "${LOG_FILE}" ]] || {
    printf '0\n'
    return 0
  }
  python3 - "${LOG_FILE}" <<'PY' 2>/dev/null || printf '0\n'
from collections import deque
import json
import re
import sys

steps = []
with open(sys.argv[1], "r", encoding="utf-8", errors="replace") as f:
    for line in deque(f, maxlen=5000):
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

training_active() {
  ps -eo args | awk '/[p]ython scripts\/534_train_native_prefixlm_dataio/ {found=1} END {exit !found}'
}

start_time="$(date +%s)"
log "watching ${OUT_DIR}"

while true; do
  publish_alias "last_model.pt"
  publish_alias "last.pt"

  step="$(latest_log_step)"
  if [[ "${TARGET_STEP}" =~ ^[0-9]+$ ]] && (( TARGET_STEP > 0 && step >= TARGET_STEP )); then
    log "target step reached: ${step}"
    publish_alias "last_model.pt"
    publish_alias "last.pt"
    break
  fi

  if ! training_active; then
    log "training process is no longer active"
    publish_alias "last_model.pt"
    publish_alias "last.pt"
    break
  fi

  if (( $(date +%s) - start_time >= MAX_SECONDS )); then
    log "max watch time reached"
    break
  fi

  sleep "${POLL_SECONDS}"
done
