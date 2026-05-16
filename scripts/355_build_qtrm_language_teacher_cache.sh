#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

OUT="${OUT:-data/qtrm_native_language_teacher/teacher_text.jsonl}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B-Base}"
MAX_RECORDS="${MAX_RECORDS:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
TOP_K_LOGPROBS="${TOP_K_LOGPROBS:-0}"

args=(
  scripts/355_build_qtrm_language_teacher_cache.py
  --out "$OUT"
  --model-id "$MODEL_ID"
  --max-records "$MAX_RECORDS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --top-k-logprobs "$TOP_K_LOGPROBS"
)

if [[ "${LOAD_IN_4BIT:-0}" == "1" ]]; then
  args+=(--load-in-4bit)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

if [[ -n "${TEXT_FILE:-}" ]]; then
  args+=(--text-file "$TEXT_FILE")
fi

if [[ -n "${TEXT_GLOB:-}" ]]; then
  args+=(--text-glob "$TEXT_GLOB")
fi

if [[ -n "${SOURCE_JSONL:-}" ]]; then
  args+=(--source-jsonl "$SOURCE_JSONL")
fi

HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}" PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" "${args[@]}" "$@"
