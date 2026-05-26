#!/usr/bin/env bash
set -euo pipefail

# Keep Stage93 micro training moving by pausing only the full-sample writer.
#
# Plain-language contract:
#   The student keeps studying the already-bound micro handout.
#   The printing press for the full textbook waits in the hallway.
#   When the micro target is reached, the printing press is resumed.

TRAIN_LOG="${TRAIN_LOG:-/tmp/20260524_STAGE93A00_DGX913M_MICRO_HARDLINK_TO24500.log}"
TARGET_STEPS="${TARGET_STEPS:-40000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_SECONDS="${MAX_SECONDS:-28800}"
WATCH_LOG="${WATCH_LOG:-/tmp/20260524_STAGE93_pause_sampler_until_micro_target.log}"
TRAIN_PID="${TRAIN_PID:-}"
SAMPLE_PID="${SAMPLE_PID:-}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${WATCH_LOG}" >/dev/null
}

find_train_pid() {
  if [[ -n "${TRAIN_PID}" && -e "/proc/${TRAIN_PID}" ]]; then
    printf '%s\n' "${TRAIN_PID}"
    return 0
  fi
  pgrep -af "python .*scripts/534_train_native_prefixlm_dataio.*STAGE93A00_DGX913M_MICRO_HARDLINK_TO24500|python .*scripts/534_train_native_prefixlm_dataio.*sampled_micro_hardlink" 2>/dev/null \
    | awk 'NR == 1 {print $1}'
}

find_sample_pid() {
  if [[ -n "${SAMPLE_PID}" && -e "/proc/${SAMPLE_PID}" ]]; then
    printf '%s\n' "${SAMPLE_PID}"
    return 0
  fi
  pgrep -af "sample_tokenized.py tokenized_path=.*/stage93_hrm_text_reasoning_nonflan_dataio/tokenized output_path=.*/stage93_hrm_text_reasoning_nonflan_dataio/sampled" 2>/dev/null \
    | grep -v "sampled_partial" \
    | awk 'NR == 1 {print $1}'
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

pause_sample_writer() {
  local pid="$1"
  ionice -c3 -p "${pid}" 2>/dev/null || true
  renice +10 -p "${pid}" >/dev/null 2>&1 || true
  kill -STOP "${pid}" 2>/dev/null || true
}

resume_sample_writer() {
  local pid="$1"
  kill -CONT "${pid}" 2>/dev/null || true
}

main() {
  local start now elapsed train_pid sample_pid step
  : > "${WATCH_LOG}"
  start="$(date +%s)"
  log "watcher started target_steps=${TARGET_STEPS} max_seconds=${MAX_SECONDS}"

  while true; do
    now="$(date +%s)"
    elapsed="$((now - start))"
    step="$(current_step)"
    train_pid="$(find_train_pid || true)"
    sample_pid="$(find_sample_pid || true)"

    if [[ -z "${sample_pid}" ]]; then
      log "sample writer not found; exiting step=${step}"
      return 0
    fi

    if [[ "${step}" =~ ^[0-9]+$ ]] && (( step >= TARGET_STEPS )); then
      resume_sample_writer "${sample_pid}"
      log "target reached; resumed sample_pid=${sample_pid} step=${step}"
      return 0
    fi

    if [[ -z "${train_pid}" ]]; then
      resume_sample_writer "${sample_pid}"
      log "training not found; resumed sample_pid=${sample_pid} step=${step}"
      return 0
    fi

    if (( elapsed >= MAX_SECONDS )); then
      resume_sample_writer "${sample_pid}"
      log "max_seconds reached; resumed sample_pid=${sample_pid} step=${step}"
      return 0
    fi

    pause_sample_writer "${sample_pid}"
    log "paused sample_pid=${sample_pid}; train_pid=${train_pid}; step=${step}; elapsed=${elapsed}s"
    sleep "${POLL_SECONDS}"
  done
}

main "$@"
