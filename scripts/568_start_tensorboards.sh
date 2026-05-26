#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/568_start_tensorboards.sh [start|status|stop] [local|dgx|both]

Defaults:
  action: start
  target: both

Environment overrides:
  LOCAL_PORT=6007
  LOCAL_HOST=127.0.0.1
  LOCAL_LOGDIR=local_eval
  LOCAL_LOG=/tmp/qtrm_tensorboard_local_6007.log
  LOCAL_PYTHON=.venv/bin/python

  DGX_SSH=dgx
  DGX_PORT=6008
  DGX_HOST=0.0.0.0
  DGX_ROOT=/mnt/data4tb/qtrm_multimodal_memoryos
  DGX_LOGDIR_SPEC=dgx_qtrm:/mnt/data4tb/qtrm_eval,dgx_local:local_eval
  DGX_LOG=/tmp/qtrm_tensorboard_dgx_6008.log
  DGX_PYTHON=/mnt/data4tb/venv_sglang_pr23000/bin/python
USAGE
}

ACTION="${1:-start}"
TARGET="${2:-both}"

LOCAL_PORT="${LOCAL_PORT:-6007}"
LOCAL_HOST="${LOCAL_HOST:-127.0.0.1}"
LOCAL_LOGDIR="${LOCAL_LOGDIR:-local_eval}"
LOCAL_LOG="${LOCAL_LOG:-/tmp/qtrm_tensorboard_local_${LOCAL_PORT}.log}"
LOCAL_PYTHON="${LOCAL_PYTHON:-.venv/bin/python}"

DGX_SSH="${DGX_SSH:-dgx}"
DGX_PORT="${DGX_PORT:-6008}"
DGX_HOST="${DGX_HOST:-0.0.0.0}"
DGX_ROOT="${DGX_ROOT:-/mnt/data4tb/qtrm_multimodal_memoryos}"
DGX_LOGDIR_SPEC="${DGX_LOGDIR_SPEC:-dgx_qtrm:/mnt/data4tb/qtrm_eval,dgx_local:local_eval}"
DGX_LOG="${DGX_LOG:-/tmp/qtrm_tensorboard_dgx_${DGX_PORT}.log}"
DGX_PYTHON="${DGX_PYTHON:-/mnt/data4tb/venv_sglang_pr23000/bin/python}"

if [[ "${ACTION}" == "-h" || "${ACTION}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${ACTION}" != "start" && "${ACTION}" != "status" && "${ACTION}" != "stop" ]]; then
  usage >&2
  exit 2
fi

if [[ "${TARGET}" != "local" && "${TARGET}" != "dgx" && "${TARGET}" != "both" ]]; then
  usage >&2
  exit 2
fi

repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

local_tensorboard_pids() {
  pgrep -f "tensorboard.*--port ${LOCAL_PORT}|tensorboard.main.*--port ${LOCAL_PORT}" || true
}

local_status() {
  echo "[local] URL: http://${LOCAL_HOST}:${LOCAL_PORT}/"
  ss -ltnp 2>/dev/null | grep -E "[:.]${LOCAL_PORT}[[:space:]]" || true
  local pids
  pids="$(local_tensorboard_pids | paste -sd, -)"
  if [[ -n "${pids}" ]]; then
    ps -o pid,stat,etime,cmd -p "${pids}" || true
  else
    echo "[local] no tensorboard process found for port ${LOCAL_PORT}"
  fi
  [[ -f "${LOCAL_LOG}" ]] && tail -n 8 "${LOCAL_LOG}" || true
}

wait_local_http() {
  local url="http://${LOCAL_HOST}:${LOCAL_PORT}/"
  for _ in $(seq 1 20); do
    if curl -fsS -I --max-time 2 "${url}" >/dev/null 2>&1; then
      echo "[local] ready: ${url}"
      return 0
    fi
    sleep 1
  done
  echo "[local] did not become ready; last log follows:" >&2
  tail -n 80 "${LOCAL_LOG}" >&2 || true
  return 1
}

start_local() {
  cd "$(repo_root)"
  if ss -ltnp 2>/dev/null | grep -qE "[:.]${LOCAL_PORT}[[:space:]]"; then
    echo "[local] port ${LOCAL_PORT} already listening; reusing existing TensorBoard"
    local_status
    return 0
  fi
  if [[ ! -x "${LOCAL_PYTHON}" ]]; then
    echo "[local] python not executable: ${LOCAL_PYTHON}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${LOCAL_LOG}")" "${LOCAL_LOGDIR}"
  : > "${LOCAL_LOG}"
  setsid "${LOCAL_PYTHON}" -m tensorboard.main \
    --logdir "${LOCAL_LOGDIR}" \
    --host "${LOCAL_HOST}" \
    --port "${LOCAL_PORT}" \
    --reload_interval 5 \
    > "${LOCAL_LOG}" 2>&1 < /dev/null &
  echo "[local] started pid=$! log=${LOCAL_LOG}"
  wait_local_http
}

stop_local() {
  local pids
  pids="$(local_tensorboard_pids | tr '\n' ' ')"
  if [[ -z "${pids}" ]]; then
    echo "[local] no TensorBoard process to stop"
    return 0
  fi
  kill ${pids} || true
  echo "[local] stopped pids: ${pids}"
}

remote_env_prefix() {
  printf 'REMOTE_ACTION=%q DGX_ROOT=%q DGX_PYTHON=%q DGX_HOST=%q DGX_PORT=%q DGX_LOGDIR_SPEC=%q DGX_LOG=%q' \
    "${ACTION}" "${DGX_ROOT}" "${DGX_PYTHON}" "${DGX_HOST}" "${DGX_PORT}" "${DGX_LOGDIR_SPEC}" "${DGX_LOG}"
}

current_host_is_dgx() {
  [[ "${DGX_SSH}" == "local" || "${DGX_SSH}" == "localhost" ]] && return 0
  [[ "$(hostname 2>/dev/null || true)" == edgexpert-* ]] && return 0
  hostname -I 2>/dev/null | grep -qw "192.168.219.113"
}

dgx_run() {
  local remote_env
  local payload
  remote_env="$(remote_env_prefix)"
  payload="$(cat <<'REMOTE'
set -euo pipefail

tensorboard_pids() {
  pgrep -f "tensorboard.*--port ${DGX_PORT}|tensorboard.main.*--port ${DGX_PORT}" || true
}

status() {
  local ip
  ip="$(hostname -I | awk '{print $1}')"
  echo "[dgx] URL: http://${ip}:${DGX_PORT}/"
  ss -ltnp 2>/dev/null | grep -E "[:.]${DGX_PORT}[[:space:]]" || true
  local pids
  pids="$(tensorboard_pids | paste -sd, -)"
  if [[ -n "${pids}" ]]; then
    ps -o pid,stat,etime,cmd -p "${pids}" || true
  else
    echo "[dgx] no tensorboard process found for port ${DGX_PORT}"
  fi
  [[ -f "${DGX_LOG}" ]] && tail -n 8 "${DGX_LOG}" || true
}

wait_http() {
  for _ in $(seq 1 20); do
    if curl -fsS -I --max-time 2 "http://127.0.0.1:${DGX_PORT}/" >/dev/null 2>&1; then
      local ip
      ip="$(hostname -I | awk '{print $1}')"
      echo "[dgx] ready: http://${ip}:${DGX_PORT}/"
      return 0
    fi
    sleep 1
  done
  echo "[dgx] did not become ready; last log follows:" >&2
  tail -n 80 "${DGX_LOG}" >&2 || true
  return 1
}

case "${REMOTE_ACTION}" in
  start)
    cd "${DGX_ROOT}"
    if ss -ltnp 2>/dev/null | grep -qE "[:.]${DGX_PORT}[[:space:]]"; then
      echo "[dgx] port ${DGX_PORT} already listening; reusing existing TensorBoard"
      status
      exit 0
    fi
    if [[ ! -x "${DGX_PYTHON}" ]]; then
      echo "[dgx] python not executable: ${DGX_PYTHON}" >&2
      exit 1
    fi
    mkdir -p "$(dirname "${DGX_LOG}")"
    : > "${DGX_LOG}"
    setsid "${DGX_PYTHON}" -m tensorboard.main \
      --logdir_spec "${DGX_LOGDIR_SPEC}" \
      --host "${DGX_HOST}" \
      --port "${DGX_PORT}" \
      --reload_interval 5 \
      > "${DGX_LOG}" 2>&1 < /dev/null &
    echo "[dgx] started pid=$! log=${DGX_LOG}"
    wait_http
    ;;
  status)
    status
    ;;
  stop)
    pids="$(tensorboard_pids | tr '\n' ' ')"
    if [[ -z "${pids}" ]]; then
      echo "[dgx] no TensorBoard process to stop"
    else
      kill ${pids} || true
      echo "[dgx] stopped pids: ${pids}"
    fi
    ;;
  *)
    echo "[dgx] unknown REMOTE_ACTION=${REMOTE_ACTION}" >&2
    exit 2
    ;;
esac
REMOTE
)"
  if current_host_is_dgx; then
    eval "${remote_env} bash -s" <<< "${payload}"
  else
    ssh "${DGX_SSH}" "${remote_env} bash -s" <<< "${payload}"
  fi
}

run_dgx() {
  dgx_run
}

run_local() {
  case "${ACTION}" in
    start) start_local ;;
    status) local_status ;;
    stop) stop_local ;;
  esac
}

case "${TARGET}" in
  local)
    run_local
    ;;
  dgx)
    run_dgx
    ;;
  both)
    run_local
    run_dgx
    ;;
esac
