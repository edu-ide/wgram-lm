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

OUT_ROOT="${OUT_ROOT:-local_eval/qtrm_native_m7a_final_token_space_mmluproval64_20260516}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_quality_s1000_20260515/last.pt}"
TRAIN_JSONL="${TRAIN_JSONL:-local_eval/m7_public_reasoning_suite/mmlu_pro_validation_64.jsonl}"
EVAL_JSONL="${EVAL_JSONL:-local_eval/m7_public_reasoning_suite/mmlu_pro_test_balanced_256.jsonl}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-300}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-3.0e-4}"
MAX_TRAIN_CASES="${MAX_TRAIN_CASES:-64}"
MAX_EVAL_CASES="${MAX_EVAL_CASES:-64}"
THINK_STEPS="${THINK_STEPS:-4}"
MARGIN_WEIGHT="${MARGIN_WEIGHT:-1.0}"
MARGIN="${MARGIN:-1.0}"
MULTI_DEPTH_CE_WEIGHT="${MULTI_DEPTH_CE_WEIGHT:-0.0}"
MULTI_DEPTH_CE_DEPTHS="${MULTI_DEPTH_CE_DEPTHS:-}"
DEPTH_GAIN_WEIGHT="${DEPTH_GAIN_WEIGHT:-0.0}"
DEPTH_GAIN_MARGIN="${DEPTH_GAIN_MARGIN:-0.25}"
DEPTH_GAIN_SHALLOW_DEPTHS="${DEPTH_GAIN_SHALLOW_DEPTHS:-0,1,2}"
TRAJECTORY_KL_WEIGHT="${TRAJECTORY_KL_WEIGHT:-0.0}"
TRAJECTORY_KL_ANCHOR_DEPTH="${TRAJECTORY_KL_ANCHOR_DEPTH:-$THINK_STEPS}"
TRAJECTORY_KL_COMPARE_DEPTHS="${TRAJECTORY_KL_COMPARE_DEPTHS:-}"
TARGET_RENDERING="${TARGET_RENDERING:-space}"
MAX_NEW_CHARS="${MAX_NEW_CHARS:-1}"
MIN_CASES="${MIN_CASES:-64}"
MAX_INVALID_PRED_RATE="${MAX_INVALID_PRED_RATE:-0.05}"
MAX_PROMPT_ECHO_RATE="${MAX_PROMPT_ECHO_RATE:-0.05}"
MAX_PRED_FRACTION="${MAX_PRED_FRACTION:-0.60}"

TRAIN_DIR="${OUT_ROOT}/train"
EVAL_DIR="${OUT_ROOT}/strict_eval"
GATE_REPORT="${OUT_ROOT}/m7a_gate_report.json"

mkdir -p "$OUT_ROOT"

PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" scripts/400_train_qtrm_native_public_mcq_final_token.py \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --train-jsonl "$TRAIN_JSONL" \
  --eval-jsonl "$EVAL_JSONL" \
  --out-dir "$TRAIN_DIR" \
  --device "$DEVICE" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --think-steps "$THINK_STEPS" \
  --max-train-cases "$MAX_TRAIN_CASES" \
  --max-eval-cases "$MAX_EVAL_CASES" \
  --margin-weight "$MARGIN_WEIGHT" \
  --margin "$MARGIN" \
  --multi-depth-ce-weight "$MULTI_DEPTH_CE_WEIGHT" \
  --multi-depth-ce-depths "$MULTI_DEPTH_CE_DEPTHS" \
  --depth-gain-weight "$DEPTH_GAIN_WEIGHT" \
  --depth-gain-margin "$DEPTH_GAIN_MARGIN" \
  --depth-gain-shallow-depths "$DEPTH_GAIN_SHALLOW_DEPTHS" \
  --trajectory-kl-weight "$TRAJECTORY_KL_WEIGHT" \
  --trajectory-kl-anchor-depth "$TRAJECTORY_KL_ANCHOR_DEPTH" \
  --trajectory-kl-compare-depths "$TRAJECTORY_KL_COMPARE_DEPTHS" \
  --target-rendering "$TARGET_RENDERING"

PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" scripts/384_eval_qtrm_native_public_mcq.py \
  --suite-jsonl "$EVAL_JSONL" \
  --checkpoint "${TRAIN_DIR}/last.pt" \
  --device "$DEVICE" \
  --think-steps "$THINK_STEPS" \
  --max-new-chars "$MAX_NEW_CHARS" \
  --max-cases "$MAX_EVAL_CASES" \
  --benchmark-id mmlu_pro \
  --benchmark-name MMLU-Pro \
  --qwen36-target-percent 86.2 \
  --parity-tolerance 0.02 \
  --min-cases-for-parity "$MAX_EVAL_CASES" \
  --log-every "$MAX_EVAL_CASES" \
  --out-dir "$EVAL_DIR" \
  --out-json "${EVAL_DIR}/report.json" \
  --out-jsonl "${EVAL_DIR}/predictions.jsonl"

PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" scripts/398_score_m7a_answer_only_gate.py \
  --eval-report "${EVAL_DIR}/report.json" \
  --out-json "$GATE_REPORT" \
  --min-cases "$MIN_CASES" \
  --min-accuracy 0.0 \
  --max-invalid-pred-rate "$MAX_INVALID_PRED_RATE" \
  --max-prompt-echo-rate "$MAX_PROMPT_ECHO_RATE" \
  --max-pred-fraction "$MAX_PRED_FRACTION"
