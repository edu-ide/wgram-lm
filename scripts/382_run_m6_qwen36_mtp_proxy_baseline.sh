#!/usr/bin/env bash
set -euo pipefail

SERVER_BIN="${SERVER_BIN:-/mnt/sda1/llama-cpp-turboquant-cuda/build/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme0n1p2/models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf}"
BASE_URL="${BASE_URL:-http://127.0.0.1:18082/v1}"
PORT="${PORT:-18082}"
SUITE_JSONL="${SUITE_JSONL:-local_eval/m6_scoped_raw_reasoning_suite/cases.jsonl}"
OUT_DIR="${OUT_DIR:-local_eval/m6_qwen36_mtp_proxy_baseline}"
OUT_JSON="${OUT_JSON:-${OUT_DIR}/report.json}"
OUT_JSONL="${OUT_JSONL:-${OUT_DIR}/predictions.jsonl}"
MAX_CASES="${MAX_CASES:-0}"
MAX_TOKENS="${MAX_TOKENS:-8}"
ANSWER_FORMAT="${ANSWER_FORMAT:-two_digit}"
CTX_SIZE="${CTX_SIZE:-4096}"
LOG_EVERY="${LOG_EVERY:-32}"
START_SERVER="${START_SERVER:-1}"
SERVER_LOG="${SERVER_LOG:-${OUT_DIR}/server.log}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

mkdir -p "${OUT_DIR}"

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "${START_SERVER}" == "1" ]]; then
  if ! curl -fsS "${BASE_URL}/models" >/dev/null 2>&1; then
    "${SERVER_BIN}" \
      -m "${MODEL_PATH}" \
      -c "${CTX_SIZE}" \
      -fa on \
      --cache-type-k turbo4 \
      --cache-type-v turbo4 \
      --kv-unified \
      -ngl 100 \
      -b 512 \
      -ub 256 \
      -t 8 \
      -tb 16 \
      -np 1 \
      --port "${PORT}" \
      --jinja \
      --reasoning off \
      --no-cache-idle-slots \
      --timeout 1800 >"${SERVER_LOG}" 2>&1 &
    server_pid="$!"
  fi

  for _ in $(seq 1 300); do
    if curl -fsS "${BASE_URL}/models" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  curl -fsS "${BASE_URL}/models" >/dev/null
fi

PYTHONPATH="${PYTHONPATH:-src}" "${PYTHON_BIN}" scripts/381_eval_openai_compatible_scoped_reasoning_baseline.py \
  --base-url "${BASE_URL}" \
  --model local \
  --model-label "Qwen3.6-27B-MTP-GGUF-UD-Q4_K_XL" \
  --model-path "${MODEL_PATH}" \
  --suite-jsonl "${SUITE_JSONL}" \
  --out-json "${OUT_JSON}" \
  --out-jsonl "${OUT_JSONL}" \
  --max-cases "${MAX_CASES}" \
  --max-tokens "${MAX_TOKENS}" \
  --answer-format "${ANSWER_FORMAT}" \
  --log-every "${LOG_EVERY}"
