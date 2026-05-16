#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/sk/ws/llm/models/Qwen3.6-27B}"
SUITE_JSONL="${SUITE_JSONL:-local_eval/m6_scoped_raw_reasoning_suite/cases.jsonl}"
OUT_JSON="${OUT_JSON:-local_eval/m6_qwen36_scoped_baseline/report.json}"
OUT_JSONL="${OUT_JSONL:-local_eval/m6_qwen36_scoped_baseline/predictions.jsonl}"
MAX_CASES="${MAX_CASES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
DEVICE="${DEVICE:-auto}"

PYTHONPATH="${PYTHONPATH:-src}" python scripts/378_eval_qwen36_scoped_raw_reasoning_baseline.py \
  --model-path "${MODEL_PATH}" \
  --suite-jsonl "${SUITE_JSONL}" \
  --out-json "${OUT_JSON}" \
  --out-jsonl "${OUT_JSONL}" \
  --device "${DEVICE}" \
  --max-cases "${MAX_CASES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"
