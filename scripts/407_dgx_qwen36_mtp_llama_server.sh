#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-start}"
DGX_HOST="${DGX_HOST:-dgx}"
DGX_HTTP_HOST="${DGX_HTTP_HOST:-192.168.219.113}"
PORT="${PORT:-18082}"
CTX_SIZE="${CTX_SIZE:-131072}"

SERVER_BIN="${SERVER_BIN:-/mnt/data4tb/llama-cpp-turboquant-cuda/build/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-/mnt/data4tb/models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf}"
LOG_FILE="${LOG_FILE:-/mnt/data4tb/qwen36-mtp-llama-server-18082.log}"
PID_FILE="${PID_FILE:-/mnt/data4tb/qwen36-mtp-llama-server-18082.pid}"

usage() {
  cat <<EOF
Usage:
  $0 [start|stop|restart|status|test|logs|url]

Environment overrides:
  DGX_HOST=${DGX_HOST}
  DGX_HTTP_HOST=${DGX_HTTP_HOST}
  PORT=${PORT}
  CTX_SIZE=${CTX_SIZE}
  SERVER_BIN=${SERVER_BIN}
  MODEL_PATH=${MODEL_PATH}
  LOG_FILE=${LOG_FILE}
  PID_FILE=${PID_FILE}

OpenAI-compatible URL:
  http://${DGX_HTTP_HOST}:${PORT}/v1
EOF
}

case "${ACTION}" in
  -h|--help|help)
    usage
    exit 0
    ;;
  url)
    echo "http://${DGX_HTTP_HOST}:${PORT}/v1"
    exit 0
    ;;
esac

ssh "${DGX_HOST}" \
  "ACTION='${ACTION}' PORT='${PORT}' CTX_SIZE='${CTX_SIZE}' SERVER_BIN='${SERVER_BIN}' MODEL_PATH='${MODEL_PATH}' LOG_FILE='${LOG_FILE}' PID_FILE='${PID_FILE}' bash -s" <<'REMOTE'
set -euo pipefail

health() {
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1
}

print_status() {
  if health; then
    echo "status=ready"
    curl -fsS "http://127.0.0.1:${PORT}/v1/models"
    echo
  else
    echo "status=stopped"
  fi
  ss -ltnp | grep ":${PORT}" || true
  pgrep -af "llama-server" | grep "${PORT}" || true
}

stop_server() {
  if [ -f "${PID_FILE}" ]; then
    pid="$(cat "${PID_FILE}" || true)"
    if [ -n "${pid}" ] && kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
      for _ in $(seq 1 30); do
        kill -0 "${pid}" >/dev/null 2>&1 || break
        sleep 1
      done
    fi
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
  fi
  rm -f "${PID_FILE}"
}

start_server() {
  if health; then
    echo "already_ready=true"
    print_status
    return
  fi

  if [ ! -x "${SERVER_BIN}" ]; then
    echo "missing llama-server: ${SERVER_BIN}" >&2
    exit 1
  fi
  if [ ! -f "${MODEL_PATH}" ]; then
    echo "missing model: ${MODEL_PATH}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${LOG_FILE}")"

  nohup "${SERVER_BIN}" \
    -m "${MODEL_PATH}" \
    -c "${CTX_SIZE}" \
    -fa on \
    --cache-type-k turbo4 \
    --cache-type-v turbo4 \
    --kv-unified \
    -ngl 100 \
    -b 1024 \
    -ub 512 \
    -t 8 \
    -tb 16 \
    -np 1 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --jinja \
    --reasoning off \
    --no-cache-idle-slots \
    --timeout 1800 \
    --sleep-idle-seconds 3600 \
    --spec-type draft-mtp,ngram-mod \
    --spec-draft-n-max 3 \
    --spec-draft-p-min 0.75 \
    --spec-ngram-mod-n-match 24 \
    --spec-ngram-mod-n-min 48 \
    --spec-ngram-mod-n-max 64 \
    >"${LOG_FILE}" 2>&1 &

  pid="$!"
  echo "${pid}" > "${PID_FILE}"
  echo "started pid=${pid}"
  echo "log=${LOG_FILE}"

  for _ in $(seq 1 300); do
    if health; then
      echo "ready=true"
      print_status
      return
    fi
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "server exited during startup" >&2
      tail -120 "${LOG_FILE}" >&2 || true
      exit 1
    fi
    sleep 1
  done

  echo "server did not become ready within timeout" >&2
  tail -120 "${LOG_FILE}" >&2 || true
  exit 1
}

run_test() {
  if ! health; then
    echo "server is not ready; run start first" >&2
    exit 1
  fi
  curl -fsS "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"local","messages":[{"role":"user","content":"Answer with one digit only: 2+2="}],"temperature":0,"max_tokens":16,"stream":false}'
  echo
}

case "${ACTION}" in
  start)
    start_server
    ;;
  stop)
    stop_server
    echo "stopped"
    ;;
  restart)
    stop_server
    start_server
    ;;
  status)
    print_status
    ;;
  test)
    run_test
    ;;
  logs)
    tail -f "${LOG_FILE}"
    ;;
  *)
    echo "unknown action: ${ACTION}" >&2
    exit 2
    ;;
esac
REMOTE
