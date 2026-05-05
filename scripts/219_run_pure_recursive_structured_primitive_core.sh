#!/usr/bin/env bash
set -euo pipefail

TRAIN_CASES="${TRAIN_CASES:-data/filtered/pure_recursive_solver_trace_all_family_train_cases.jsonl}"
TRAIN_ROWS="${TRAIN_ROWS:-data/filtered/pure_recursive_solver_trace_all_family_train.jsonl}"
EVAL_CASES="${EVAL_CASES:-data/eval/pure_recursive_solver_trace_all_family_heldout_cases.jsonl}"
EVAL_ROWS="${EVAL_ROWS:-data/eval/pure_recursive_solver_trace_all_family_heldout.jsonl}"
OUT_DIR="${OUT_DIR:-local_eval/pure_recursive_structured_operation_policy_all_family_s240}"
CASES_PER_FAMILY="${CASES_PER_FAMILY:-128}"
EVAL_CASES_PER_FAMILY="${EVAL_CASES_PER_FAMILY:-32}"
TRAIN_START_INDEX="${TRAIN_START_INDEX:-3000}"
EVAL_START_INDEX="${EVAL_START_INDEX:-4000}"
STEPS="${STEPS:-300}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.003}"
D_MODEL="${D_MODEL:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
SEED="${SEED:-240}"

echo "=== Pure Recursive Structured Primitive Core Probe ==="
echo "train rows: ${TRAIN_ROWS}"
echo "eval rows:  ${EVAL_ROWS}"
echo "out dir:    ${OUT_DIR}"
echo

PYTHONPATH="${PYTHONPATH:-src}" python scripts/190_build_pure_recursive_reasoning_cases.py \
  --out "${TRAIN_CASES}" \
  --cases-per-family "${CASES_PER_FAMILY}" \
  --start-index "${TRAIN_START_INDEX}"

PYTHONPATH="${PYTHONPATH:-src}" python scripts/190_build_pure_recursive_reasoning_cases.py \
  --out "${EVAL_CASES}" \
  --cases-per-family "${EVAL_CASES_PER_FAMILY}" \
  --start-index "${EVAL_START_INDEX}"

PYTHONPATH="${PYTHONPATH:-src}" python scripts/215_build_pure_recursive_solver_trace_dataset.py \
  --cases "${TRAIN_CASES}" \
  --out "${TRAIN_ROWS}"

PYTHONPATH="${PYTHONPATH:-src}" python scripts/215_build_pure_recursive_solver_trace_dataset.py \
  --cases "${EVAL_CASES}" \
  --out "${EVAL_ROWS}"

PYTHONPATH="${PYTHONPATH:-src}" python scripts/218_train_pure_recursive_structured_operation_policy.py \
  --train-jsonl "${TRAIN_ROWS}" \
  --eval-jsonl "${EVAL_ROWS}" \
  --out-dir "${OUT_DIR}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --d-model "${D_MODEL}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --seed "${SEED}" \
  --log-every 50
