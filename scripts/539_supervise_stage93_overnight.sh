#!/usr/bin/env bash
set -uo pipefail

# Overnight supervisor for Stage93.
#
# Plain-language contract:
#   - keep printing/binding the full textbook;
#   - while the full textbook is not ready, keep the student studying the
#     partial booklet if possible;
#   - when the full textbook is ready, continue from the best checkpoint;
#   - if a run exits early, relaunch from the newest usable checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT}" || exit 2

WORK_DIR="${WORK_DIR:-${ROOT}/local_eval/stage93_hrm_text_reasoning_nonflan_dataio}"
FULL_SAMPLED="${FULL_SAMPLED:-${WORK_DIR}/sampled}"
PARTIAL_SAMPLED="${PARTIAL_SAMPLED:-${WORK_DIR}/sampled_partial}"
MICRO_HARDLINK_SAMPLED="${MICRO_HARDLINK_SAMPLED:-${WORK_DIR}/sampled_micro_hardlink}"

STAGE92_OUT="${STAGE92_OUT:-${ROOT}/local_eval/20260523_STAGE92_DGX913M_BS8_CONTINUE_TO24K}"
PARTIAL_RUN_NAME="${PARTIAL_RUN_NAME:-20260524_STAGE93A_DGX913M_PARTIAL_TO30K}"
PARTIAL_OUT="${PARTIAL_OUT:-${ROOT}/local_eval/${PARTIAL_RUN_NAME}}"
MICRO_RUN_NAME="${MICRO_RUN_NAME:-20260524_STAGE93A00_DGX913M_MICRO_TO24500}"
MICRO_OUT="${MICRO_OUT:-${ROOT}/local_eval/${MICRO_RUN_NAME}}"
MICRO_HARDLINK_RUN_NAME="${MICRO_HARDLINK_RUN_NAME:-20260524_STAGE93A00_DGX913M_MICRO_HARDLINK_TO24500}"
MICRO_HARDLINK_OUT="${MICRO_HARDLINK_OUT:-${ROOT}/local_eval/${MICRO_HARDLINK_RUN_NAME}}"
BOOTSTRAP_RUN_NAME="${BOOTSTRAP_RUN_NAME:-20260524_STAGE93A0_DGX913M_BOOTSTRAP_TO26K}"
BOOTSTRAP_OUT="${BOOTSTRAP_OUT:-${ROOT}/local_eval/${BOOTSTRAP_RUN_NAME}}"
FULL_RUN_NAME="${FULL_RUN_NAME:-20260524_STAGE93B_DGX913M_FULL_FROM_BEST_TO60K}"
FULL_OUT="${FULL_OUT:-${ROOT}/local_eval/${FULL_RUN_NAME}}"

PARTIAL_TARGET_STEPS="${PARTIAL_TARGET_STEPS:-30000}"
MICRO_HARDLINK_TARGET_STEPS="${MICRO_HARDLINK_TARGET_STEPS:-40000}"
FULL_TARGET_STEPS="${FULL_TARGET_STEPS:-60000}"

POLL_SECONDS="${POLL_SECONDS:-120}"
STABILITY_SECONDS="${STABILITY_SECONDS:-120}"
FULL_MIN_INDEX_FILES="${FULL_MIN_INDEX_FILES:-20}"
PARTIAL_MIN_INDEX_FILES="${PARTIAL_MIN_INDEX_FILES:-8}"
MICRO_HARDLINK_MIN_INDEX_FILES="${MICRO_HARDLINK_MIN_INDEX_FILES:-8}"
MIN_PARTIAL_TASKS="${MIN_PARTIAL_TASKS:-64}"
IO_RELIEF_ENABLED="${IO_RELIEF_ENABLED:-1}"
IO_RELIEF_PAUSE_SECONDS="${IO_RELIEF_PAUSE_SECONDS:-300}"
IO_RELIEF_COOLDOWN_SECONDS="${IO_RELIEF_COOLDOWN_SECONDS:-180}"
IO_RELIEF_LAST_FILE="${IO_RELIEF_LAST_FILE:-/tmp/20260524_STAGE93_io_relief.last}"

FULL_LOG="${FULL_LOG:-/tmp/${FULL_RUN_NAME}.log}"
PARTIAL_LOG="${PARTIAL_LOG:-/tmp/${PARTIAL_RUN_NAME}.log}"
MICRO_HARDLINK_LOG="${MICRO_HARDLINK_LOG:-/tmp/${MICRO_HARDLINK_RUN_NAME}.log}"
SUPERVISOR_LOG="${SUPERVISOR_LOG:-/tmp/20260524_STAGE93_overnight_supervisor.log}"

log() {
  local msg="$*"
  printf '[%s] %s\n' "$(date -Is)" "${msg}" | tee -a "${SUPERVISOR_LOG}" >/dev/null
}

mtime_age_seconds() {
  local path="$1"
  local now mtime
  [[ -e "${path}" ]] || return 1
  now="$(date +%s)"
  mtime="$(stat -c '%Y' "${path}")"
  printf '%s\n' "$((now - mtime))"
}

stable_file() {
  local path="$1"
  local age
  [[ -s "${path}" ]] || return 1
  age="$(mtime_age_seconds "${path}")" || return 1
  (( age >= STABILITY_SECONDS ))
}

sample_ready() {
  local sample_dir="$1"
  local min_index_files="$2"
  local index_count
  stable_file "${sample_dir}/tokens.npy" || return 1
  stable_file "${sample_dir}/metadata.json" || return 1
  index_count="$(find "${sample_dir}" -path "${sample_dir}/epoch_*/*.npy" 2>/dev/null | wc -l)"
  (( index_count >= min_index_files ))
}

training_active() {
  training_pid >/dev/null 2>&1
}

training_pid() {
  pgrep -af "python .*scripts/534_train_native_prefixlm_dataio" 2>/dev/null \
    | awk '$2 ~ /python/ {print $1; exit}'
}

full_data_active() {
  pgrep -af "scripts/535_prepare_stage93_hrm_text_large_dataio.sh all|target/release/tokenizer .*/stage93_hrm_text_reasoning_nonflan_dataio/cleaned_subset|sample_tokenized.py tokenized_path=.*/stage93_hrm_text_reasoning_nonflan_dataio/tokenized output_path=.*/stage93_hrm_text_reasoning_nonflan_dataio/sampled" \
    | grep -v "tokenized_partial" \
    | grep -v "sampled_partial" \
    >/dev/null 2>&1
}

full_sample_writer_pid() {
  pgrep -af "sample_tokenized.py tokenized_path=.*/stage93_hrm_text_reasoning_nonflan_dataio/tokenized output_path=.*/stage93_hrm_text_reasoning_nonflan_dataio/sampled" 2>/dev/null \
    | grep -v "tokenized_partial" \
    | grep -v "sampled_partial" \
    | awk 'NR == 1 {print $1}'
}

partial_bind_active() {
  pgrep -af "scripts/538_prepare_stage93_partial_sample|sample_tokenized.py tokenized_path=.*/tokenized_partial output_path=.*/sampled_partial" >/dev/null 2>&1
}

tokenized_count() {
  find "${WORK_DIR}/tokenized" -mindepth 2 -maxdepth 2 -name tokens.npy 2>/dev/null | wc -l
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
        for v in x.values():
            walk(v)
    elif isinstance(x, list):
        for v in x:
            walk(v)

try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        walk(json.load(f))
except Exception:
    pass

print(max(steps) if steps else 0)
PY
}

log_step() {
  local log_file="$1"
  [[ -f "${log_file}" ]] || {
    printf '0\n'
    return 0
  }
  python3 - "${log_file}" <<'PY' 2>/dev/null || printf '0\n'
from collections import deque
import json
import re
import sys

steps = []
try:
    with open(sys.argv[1], "r", encoding="utf-8", errors="replace") as f:
        lines = deque(f, maxlen=5000)
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

run_step() {
  local out_dir="$1"
  local report_max log_max log_file
  report_max="$(report_step "${out_dir}/report.json")"
  [[ "${report_max}" =~ ^[0-9]+$ ]] || report_max=0

  log_file=""
  case "${out_dir}" in
    "${FULL_OUT}") log_file="${FULL_LOG}" ;;
    "${PARTIAL_OUT}") log_file="${PARTIAL_LOG}" ;;
    "${MICRO_HARDLINK_OUT}") log_file="${MICRO_HARDLINK_LOG}" ;;
  esac

  if [[ -n "${log_file}" ]]; then
    log_max="$(log_step "${log_file}")"
    [[ "${log_max}" =~ ^[0-9]+$ ]] || log_max=0
  else
    log_max=0
  fi

  if (( log_max > report_max )); then
    printf '%s\n' "${log_max}"
  else
    printf '%s\n' "${report_max}"
  fi
}

cooldown_elapsed() {
  local marker="$1"
  local cooldown="$2"
  local now last
  now="$(date +%s)"
  if [[ ! -f "${marker}" ]]; then
    return 0
  fi
  last="$(cat "${marker}" 2>/dev/null || printf '0')"
  [[ "${last}" =~ ^[0-9]+$ ]] || last=0
  (( now - last >= cooldown ))
}

maybe_relieve_io_pressure() {
  local train_pid sample_pid train_state train_wchan now
  [[ "${IO_RELIEF_ENABLED}" == "1" ]] || return 0

  train_pid="$(training_pid)"
  sample_pid="$(full_sample_writer_pid)"
  [[ -n "${train_pid}" && -n "${sample_pid}" ]] || return 0
  [[ -r "/proc/${train_pid}/stat" && -r "/proc/${train_pid}/wchan" ]] || return 0

  ionice -c3 -p "${sample_pid}" 2>/dev/null || true
  renice +10 -p "${sample_pid}" >/dev/null 2>&1 || true
  ionice -c2 -n0 -p "${train_pid}" 2>/dev/null || true

  train_state="$(awk '{print $3}' "/proc/${train_pid}/stat" 2>/dev/null || true)"
  train_wchan="$(cat "/proc/${train_pid}/wchan" 2>/dev/null || true)"
  [[ "${train_state}" == "D" ]] || return 0
  [[ "${train_wchan}" =~ rq_qos_wait|io_schedule|balance_dirty_pages|wait_on_page|blk_mq ]] || return 0
  cooldown_elapsed "${IO_RELIEF_LAST_FILE}" "${IO_RELIEF_COOLDOWN_SECONDS}" || {
    log "io relief skipped: cooldown active train_pid=${train_pid} wchan=${train_wchan}"
    return 0
  }

  now="$(date +%s)"
  printf '%s\n' "${now}" > "${IO_RELIEF_LAST_FILE}" 2>/dev/null || true
  log "io relief: pausing full sample writer pid=${sample_pid} for ${IO_RELIEF_PAUSE_SECONDS}s because train_pid=${train_pid} is D/${train_wchan}"
  (
    trap 'kill -CONT "'"${sample_pid}"'" 2>/dev/null || true' EXIT INT TERM
    kill -STOP "${sample_pid}" 2>/dev/null || true
    sleep "${IO_RELIEF_PAUSE_SECONDS}"
    kill -CONT "${sample_pid}" 2>/dev/null || true
    trap - EXIT INT TERM
  )
  train_state="$(awk '{print $3}' "/proc/${train_pid}/stat" 2>/dev/null || true)"
  train_wchan="$(cat "/proc/${train_pid}/wchan" 2>/dev/null || true)"
  log "io relief: resumed full sample writer pid=${sample_pid}; train_pid=${train_pid} state=${train_state:-unknown} wchan=${train_wchan:-unknown}"
}

target_reached() {
  local out_dir="$1"
  local target="$2"
  local step
  step="$(run_step "${out_dir}")"
  [[ "${step}" =~ ^[0-9]+$ ]] || step=0
  (( step >= target ))
}

checkpoint_ready() {
  stable_file "$1"
}

newest_ready_checkpoint() {
  local ckpt best_ckpt best_mtime mtime
  best_ckpt=""
  best_mtime=0
  for ckpt in "$@"; do
    if checkpoint_ready "${ckpt}"; then
      mtime="$(stat -c '%Y' "${ckpt}" 2>/dev/null || printf '0')"
      [[ "${mtime}" =~ ^[0-9]+$ ]] || mtime=0
      if (( mtime > best_mtime )); then
        best_mtime="${mtime}"
        best_ckpt="${ckpt}"
      fi
    fi
  done
  if [[ -n "${best_ckpt}" ]]; then
    printf '%s\n' "${best_ckpt}"
    return 0
  fi
  return 1
}

best_checkpoint_for_full() {
  newest_ready_checkpoint \
    "${FULL_OUT}/last_model.pt" \
    "${FULL_OUT}/last.pt" \
    "${PARTIAL_OUT}/last_model.pt" \
    "${PARTIAL_OUT}/last.pt" \
    "${BOOTSTRAP_OUT}/last_model.pt" \
    "${BOOTSTRAP_OUT}/last.pt" \
    "${MICRO_HARDLINK_OUT}/last_model.pt" \
    "${MICRO_HARDLINK_OUT}/last.pt" \
    "${MICRO_OUT}/last_model.pt" \
    "${MICRO_OUT}/last.pt" \
    "${STAGE92_OUT}/last_model.pt" \
    "${STAGE92_OUT}/last.pt"
}

best_checkpoint_for_partial() {
  newest_ready_checkpoint \
    "${PARTIAL_OUT}/last_model.pt" \
    "${PARTIAL_OUT}/last.pt" \
    "${BOOTSTRAP_OUT}/last_model.pt" \
    "${BOOTSTRAP_OUT}/last.pt" \
    "${MICRO_HARDLINK_OUT}/last_model.pt" \
    "${MICRO_HARDLINK_OUT}/last.pt" \
    "${MICRO_OUT}/last_model.pt" \
    "${MICRO_OUT}/last.pt" \
    "${STAGE92_OUT}/last_model.pt" \
    "${STAGE92_OUT}/last.pt"
}

restart_full_data_prep() {
  log "full sampled data not ready and no full data process is active; restarting data prep"
  PROFILE=reasoning_nonflan bash "${ROOT}/scripts/535_prepare_stage93_hrm_text_large_dataio.sh" launch >> "${SUPERVISOR_LOG}" 2>&1 || true
}

restart_partial_bind_and_launch() {
  local count
  count="$(tokenized_count)"
  if (( count < MIN_PARTIAL_TASKS )); then
    log "partial booklet not started: only ${count} tokenized tasks, need ${MIN_PARTIAL_TASKS}"
    return 0
  fi
  log "partial sampled data not ready and no partial binder is active; restarting partial bind-and-launch"
  nohup env \
    PYTHONUNBUFFERED=1 \
    MAX_TASKS=256 \
    MIN_TASKS="${MIN_PARTIAL_TASKS}" \
    STABLE_SECONDS="${STABILITY_SECONDS}" \
    EPOCHS=2 \
    PARTIAL_STEPS="${PARTIAL_TARGET_STEPS}" \
    bash -lc "scripts/538_prepare_stage93_partial_sample.sh all && scripts/538_prepare_stage93_partial_sample.sh launch" \
    >> /tmp/20260524_STAGE93A_partial_bind_and_launch.log 2>&1 < /dev/null &
  log "partial bind-and-launch pid=${!}"
}

launch_partial_training() {
  local resume
  resume="$(best_checkpoint_for_partial)" || {
    log "partial launch skipped: no stable resume checkpoint"
    return 0
  }
  log "launching partial Stage93A from ${resume}"
  SAMPLED_DATA="${PARTIAL_SAMPLED}" \
    RESUME="${resume}" \
    RUN_NAME="${PARTIAL_RUN_NAME}" \
    OUT_ROOT="${PARTIAL_OUT}" \
    STEPS="${PARTIAL_TARGET_STEPS}" \
    LOG_FILE="${PARTIAL_LOG}" \
    bash "${ROOT}/scripts/536_launch_stage93_dgx_continue_prefixlm.sh" >> "${SUPERVISOR_LOG}" 2>&1 || true
}

launch_micro_hardlink_training() {
  local resume
  resume="$(best_checkpoint_for_partial)" || {
    log "micro hardlink launch skipped: no stable resume checkpoint"
    return 0
  }
  log "launching micro hardlink Stage93A00 from ${resume}"
  SAMPLED_DATA="${MICRO_HARDLINK_SAMPLED}" \
    RESUME="${resume}" \
    RUN_NAME="${MICRO_HARDLINK_RUN_NAME}" \
    OUT_ROOT="${MICRO_HARDLINK_OUT}" \
    STEPS="${MICRO_HARDLINK_TARGET_STEPS}" \
    LOG_FILE="${MICRO_HARDLINK_LOG}" \
    MODEL_CHECKPOINT_EVERY=1000 \
    CHECKPOINT_EVERY=2000 \
    EVAL_EVERY=200 \
    bash "${ROOT}/scripts/536_launch_stage93_dgx_continue_prefixlm.sh" >> "${SUPERVISOR_LOG}" 2>&1 || true
}

launch_full_training() {
  local resume
  resume="$(best_checkpoint_for_full)" || {
    log "full launch skipped: no stable resume checkpoint"
    return 0
  }
  log "launching full Stage93B from ${resume}"
  SAMPLED_DATA="${FULL_SAMPLED}" \
    RESUME="${resume}" \
    RUN_NAME="${FULL_RUN_NAME}" \
    OUT_ROOT="${FULL_OUT}" \
    STEPS="${FULL_TARGET_STEPS}" \
    LOG_FILE="${FULL_LOG}" \
    bash "${ROOT}/scripts/536_launch_stage93_dgx_continue_prefixlm.sh" >> "${SUPERVISOR_LOG}" 2>&1 || true
}

log_state() {
  local full_ready partial_ready micro_ready active_train active_data active_partial partial_step micro_step full_step count
  sample_ready "${FULL_SAMPLED}" "${FULL_MIN_INDEX_FILES}" && full_ready=1 || full_ready=0
  sample_ready "${PARTIAL_SAMPLED}" "${PARTIAL_MIN_INDEX_FILES}" && partial_ready=1 || partial_ready=0
  sample_ready "${MICRO_HARDLINK_SAMPLED}" "${MICRO_HARDLINK_MIN_INDEX_FILES}" && micro_ready=1 || micro_ready=0
  training_active && active_train=1 || active_train=0
  full_data_active && active_data=1 || active_data=0
  partial_bind_active && active_partial=1 || active_partial=0
  partial_step="$(run_step "${PARTIAL_OUT}")"
  micro_step="$(run_step "${MICRO_HARDLINK_OUT}")"
  full_step="$(run_step "${FULL_OUT}")"
  count="$(tokenized_count)"
  log "state full_ready=${full_ready} partial_ready=${partial_ready} micro_ready=${micro_ready} train_active=${active_train} full_data_active=${active_data} partial_bind_active=${active_partial} tokenized=${count} partial_step=${partial_step} micro_step=${micro_step} full_step=${full_step}"
}

main() {
  : > "${SUPERVISOR_LOG}"
  log "Stage93 overnight supervisor started"
  log "root=${ROOT}"
  log "full_sampled=${FULL_SAMPLED}"
  log "partial_sampled=${PARTIAL_SAMPLED}"
  log "micro_hardlink_sampled=${MICRO_HARDLINK_SAMPLED}"
  log "partial_out=${PARTIAL_OUT}"
  log "micro_hardlink_out=${MICRO_HARDLINK_OUT}"
  log "micro_out=${MICRO_OUT}"
  log "bootstrap_out=${BOOTSTRAP_OUT}"
  log "full_out=${FULL_OUT}"
  log "partial_target_steps=${PARTIAL_TARGET_STEPS}"
  log "micro_hardlink_target_steps=${MICRO_HARDLINK_TARGET_STEPS}"
  log "full_target_steps=${FULL_TARGET_STEPS}"
  log "io_relief_enabled=${IO_RELIEF_ENABLED} pause_seconds=${IO_RELIEF_PAUSE_SECONDS} cooldown_seconds=${IO_RELIEF_COOLDOWN_SECONDS}"

  while true; do
    log_state
    maybe_relieve_io_pressure

    if target_reached "${FULL_OUT}" "${FULL_TARGET_STEPS}"; then
      log "full target reached; supervisor exiting"
      return 0
    fi

    if ! sample_ready "${FULL_SAMPLED}" "${FULL_MIN_INDEX_FILES}"; then
      if ! full_data_active; then
        restart_full_data_prep
      fi

      if ! training_active; then
        if sample_ready "${MICRO_HARDLINK_SAMPLED}" "${MICRO_HARDLINK_MIN_INDEX_FILES}" \
          && ! target_reached "${MICRO_HARDLINK_OUT}" "${MICRO_HARDLINK_TARGET_STEPS}"; then
          launch_micro_hardlink_training
        elif sample_ready "${PARTIAL_SAMPLED}" "${PARTIAL_MIN_INDEX_FILES}"; then
          if ! target_reached "${PARTIAL_OUT}" "${PARTIAL_TARGET_STEPS}"; then
            launch_partial_training
          else
            log "partial target reached; waiting for full sampled data"
          fi
        elif ! partial_bind_active; then
          restart_partial_bind_and_launch
        else
          log "partial booklet is still being bound"
        fi
      else
        log "training already active while full sampled data is pending"
      fi

      sleep "${POLL_SECONDS}"
      continue
    fi

    log "full sampled data is ready"
    if training_active; then
      log "training is active; waiting for the current run/checkpoint to finish before full handoff"
      sleep "${POLL_SECONDS}"
      continue
    fi

    launch_full_training
    sleep "${POLL_SECONDS}"
  done
}

main "$@"
