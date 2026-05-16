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

OUT_ROOT="${OUT_ROOT:-local_eval/qtrm_native_m7a_answer_only_healing_20260516}"
TRAIN_SUITE="${TRAIN_SUITE:-local_eval/m7_public_reasoning_suite/mmlu_aux_all_auxtrain_1024_20260516.jsonl}"
EVAL_SUITE="${EVAL_SUITE:-local_eval/m7_public_reasoning_suite/mmlu_pro_test_balanced_256.jsonl}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-local_eval/qtrm_native_pretrained_init_qwen35_compact_external4500_s3600_20260515/last.pt}"
DEVICE="${DEVICE:-cuda}"
MAX_TRAIN_RECORDS="${MAX_TRAIN_RECORDS:-1024}"
TRAIN_REPEATS="${TRAIN_REPEATS:-1}"
REPAIR_JSONL_REPEATS="${REPAIR_JSONL_REPEATS:-1}"
STAGE_C_STEPS="${STAGE_C_STEPS:-300}"
LR="${LR:-1.0e-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_MAX_CASES="${EVAL_MAX_CASES:-256}"
MIN_CASES="${MIN_CASES:-64}"
MIN_ACCURACY="${MIN_ACCURACY:-0.0}"
MAX_INVALID_PRED_RATE="${MAX_INVALID_PRED_RATE:-0.05}"
MAX_PROMPT_ECHO_RATE="${MAX_PROMPT_ECHO_RATE:-0.05}"
MAX_PRED_FRACTION="${MAX_PRED_FRACTION:-0.60}"

CORPUS_JSONL="${OUT_ROOT}/m7a_answer_only_train.jsonl"
CORPUS_REPORT="${OUT_ROOT}/m7a_answer_only_corpus_report.json"
TRAIN_DIR="${OUT_ROOT}/train"
EVAL_DIR="${OUT_ROOT}/strict_eval"
GATE_REPORT="${OUT_ROOT}/m7a_gate_report.json"

mkdir -p "$OUT_ROOT"

PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" scripts/397_build_m7a_public_mcq_answer_only_corpus.py \
  --suite-jsonl "$TRAIN_SUITE" \
  --out-jsonl "$CORPUS_JSONL" \
  --out-json "$CORPUS_REPORT" \
  --max-records "$MAX_TRAIN_RECORDS" \
  --repeats "$TRAIN_REPEATS"

REPAIR_SEED_TEXTS="$("$PYTHON_BIN" - <<PY
import json
from pathlib import Path
report = json.loads(Path("${CORPUS_REPORT}").read_text(encoding="utf-8"))
print(report["repair_seed_texts"])
PY
)"

set +e
PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" scripts/354_train_qtrm_native_language_bootstrap.py \
  --out-dir "$TRAIN_DIR" \
  --device "$DEVICE" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --repair-jsonl "$CORPUS_JSONL" \
  --repair-jsonl-repeats "$REPAIR_JSONL_REPEATS" \
  --tiny-repeats 0 \
  --textbook-repeats 0 \
  --surface-answer-repeats 0 \
  --diverse-surface-answer-count 0 \
  --gate-anchor-repeats 0 \
  --stage-a-steps 0 \
  --stage-b-steps 0 \
  --stage-c-steps "$STAGE_C_STEPS" \
  --max-text-chars 0 \
  --lr "$LR" \
  --batch-size "$BATCH_SIZE" \
  --max-new-chars 6 \
  --repair-prompt-count 3 \
  --repair-seed-texts "$REPAIR_SEED_TEXTS" \
  --repair-seed-expectations '{}' \
  --min-on-policy-continuation-chars 1 \
  --min-on-policy-keyword-hits 0 \
  --min-on-policy-informative-char-fraction 0.0 \
  --max-on-policy-repeated-word-fraction 1.0 \
  --accepted-decision accepted_m7a_public_mcq_answer_only_healing
TRAIN_EXIT=$?
set -e

if [[ ! -f "${TRAIN_DIR}/last.pt" ]]; then
  echo "training did not produce ${TRAIN_DIR}/last.pt" >&2
  exit "$TRAIN_EXIT"
fi

PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" scripts/384_eval_qtrm_native_public_mcq.py \
  --suite-jsonl "$EVAL_SUITE" \
  --checkpoint "${TRAIN_DIR}/last.pt" \
  --device "$DEVICE" \
  --think-steps 4 \
  --max-new-chars 6 \
  --max-cases "$EVAL_MAX_CASES" \
  --benchmark-id mmlu_pro \
  --benchmark-name MMLU-Pro \
  --qwen36-target-percent 86.2 \
  --parity-tolerance 0.02 \
  --min-cases-for-parity "$EVAL_MAX_CASES" \
  --log-every 64 \
  --out-dir "$EVAL_DIR" \
  --out-json "${EVAL_DIR}/report.json" \
  --out-jsonl "${EVAL_DIR}/predictions.jsonl"

PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" scripts/398_score_m7a_answer_only_gate.py \
  --eval-report "${EVAL_DIR}/report.json" \
  --out-json "$GATE_REPORT" \
  --min-cases "$MIN_CASES" \
  --min-accuracy "$MIN_ACCURACY" \
  --max-invalid-pred-rate "$MAX_INVALID_PRED_RATE" \
  --max-prompt-echo-rate "$MAX_PROMPT_ECHO_RATE" \
  --max-pred-fraction "$MAX_PRED_FRACTION"
