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

OUT_ROOT="${OUT_ROOT:-local_eval/qtrm_native_m7b_core_depth_gate_20260516}"
CHECKPOINT="${CHECKPOINT:-local_eval/qtrm_native_m7a_final_token_space_mmluproval64_s300_20260516/last.pt}"
EVAL_JSONL="${EVAL_JSONL:-local_eval/m7_public_reasoning_suite/mmlu_pro_test_balanced_256.jsonl}"
DEVICE="${DEVICE:-cuda}"
MAX_CASES="${MAX_CASES:-64}"
MAX_NEW_CHARS="${MAX_NEW_CHARS:-1}"
FULL_THINK_STEPS="${FULL_THINK_STEPS:-4}"
SHALLOW_DEPTHS="${SHALLOW_DEPTHS:-1 2}"
BASELINE_DEPTH="${BASELINE_DEPTH:-0}"
MIN_GAIN_VS_BASELINE="${MIN_GAIN_VS_BASELINE:-0.03}"
MIN_GAIN_VS_BEST_SHALLOW="${MIN_GAIN_VS_BEST_SHALLOW:-0.03}"

mkdir -p "$OUT_ROOT"

run_eval() {
  local depth="$1"
  local out_dir="${OUT_ROOT}/depth${depth}"
  PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" scripts/384_eval_qtrm_native_public_mcq.py \
    --suite-jsonl "$EVAL_JSONL" \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE" \
    --think-steps "$depth" \
    --max-new-chars "$MAX_NEW_CHARS" \
    --max-cases "$MAX_CASES" \
    --benchmark-id mmlu_pro \
    --benchmark-name MMLU-Pro \
    --qwen36-target-percent 86.2 \
    --parity-tolerance 0.02 \
    --min-cases-for-parity "$MAX_CASES" \
    --log-every 0 \
    --out-dir "$out_dir" \
    --out-json "${out_dir}/report.json" \
    --out-jsonl "${out_dir}/predictions.jsonl"
}

run_eval "$BASELINE_DEPTH"
for depth in $SHALLOW_DEPTHS; do
  run_eval "$depth"
done
run_eval "$FULL_THINK_STEPS"

SHALLOW_ARGS=()
for depth in $SHALLOW_DEPTHS; do
  SHALLOW_ARGS+=(--shallow-report "${OUT_ROOT}/depth${depth}/report.json")
done

PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" scripts/402_score_m7b_core_depth_gate.py \
  --full-report "${OUT_ROOT}/depth${FULL_THINK_STEPS}/report.json" \
  --baseline-report "${OUT_ROOT}/depth${BASELINE_DEPTH}/report.json" \
  "${SHALLOW_ARGS[@]}" \
  --out-json "${OUT_ROOT}/m7b_gate_report.json" \
  --min-cases "$MAX_CASES" \
  --min-gain-vs-baseline "$MIN_GAIN_VS_BASELINE" \
  --min-gain-vs-best-shallow "$MIN_GAIN_VS_BEST_SHALLOW"
