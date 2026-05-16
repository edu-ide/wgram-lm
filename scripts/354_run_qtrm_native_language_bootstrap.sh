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

OUT_DIR="${OUT_DIR:-local_eval/qtrm_native_language_bootstrap_standard}"
DEVICE="${DEVICE:-cuda}"
TOKENIZER_NAME="${TOKENIZER_NAME:-}"
STAGE_A_STEPS="${STAGE_A_STEPS:-400}"
STAGE_B_STEPS="${STAGE_B_STEPS:-800}"
STAGE_C_STEPS="${STAGE_C_STEPS:-0}"
MAX_TEXT_CHARS="${MAX_TEXT_CHARS:-120000}"

args=(
  scripts/354_train_qtrm_native_language_bootstrap.py
  --out-dir "$OUT_DIR"
  --device "$DEVICE"
  --stage-a-steps "$STAGE_A_STEPS"
  --stage-b-steps "$STAGE_B_STEPS"
  --stage-c-steps "$STAGE_C_STEPS"
  --max-text-chars "$MAX_TEXT_CHARS"
)

if [[ -n "$TOKENIZER_NAME" ]]; then
  args+=(--tokenizer-name "$TOKENIZER_NAME")
fi

if [[ -n "${TEXT_FILE:-}" ]]; then
  args+=(--text-file "$TEXT_FILE")
fi

if [[ -n "${TEXT_GLOB:-}" ]]; then
  args+=(--text-glob "$TEXT_GLOB")
fi

if [[ -n "${TEACHER_TEXT_FILE:-}" ]]; then
  args+=(--teacher-text-file "$TEACHER_TEXT_FILE")
fi

if [[ -n "${TEACHER_JSONL:-}" ]]; then
  args+=(--teacher-jsonl "$TEACHER_JSONL")
fi

PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" "${args[@]}" "$@"
