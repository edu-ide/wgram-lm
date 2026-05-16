#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B-Base}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-local_eval/qwen35_integrated_healing_l23_langkl_s100_20260516/last_core.pt}"
TRAIN_JSONL="${TRAIN_JSONL:-local_eval/m7_public_reasoning_suite/mmlu_pro_validation_64.jsonl}"
EVAL_JSONL="${EVAL_JSONL:-local_eval/m7_public_reasoning_suite/mmlu_pro_test_balanced_256.jsonl}"
OUT_DIR="${OUT_DIR:-local_eval/qwen35_integrated_public_mcq_healing_val64_to_test256_s60_20260516}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
STEPS="${STEPS:-60}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-2.0e-5}"
QWEN_LR="${QWEN_LR:-5.0e-7}"
LANGUAGE_KL_WEIGHT="${LANGUAGE_KL_WEIGHT:-0.20}"
BASE_KL_WEIGHT="${BASE_KL_WEIGHT:-0.02}"
CE_FOCUS="${CE_FOCUS:-all}"
QWEN_CORE_LAYER_INDICES="${QWEN_CORE_LAYER_INDICES:-3}"
CORE_ADAPTER_DIM="${CORE_ADAPTER_DIM:-128}"
CORE_DELTA_ADAPTER_MODE="${CORE_DELTA_ADAPTER_MODE:-add}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.05}"
CHECKPOINT_LOAD_MODE="${CHECKPOINT_LOAD_MODE:-strict_shapes}"
MARGIN_WEIGHT="${MARGIN_WEIGHT:-0.0}"
MARGIN_VALUE="${MARGIN_VALUE:-0.5}"
MARGIN_FOCUS="${MARGIN_FOCUS:-base_wrong}"
BALANCED_CATEGORY_SAMPLING="${BALANCED_CATEGORY_SAMPLING:-0}"
CATEGORY_REGRESSION_PENALTY="${CATEGORY_REGRESSION_PENALTY:-0.0}"
MIN_EVAL_CATEGORY_GAIN="${MIN_EVAL_CATEGORY_GAIN:--1.0}"
MIN_EVAL_CATEGORY_HIT_DELTA="${MIN_EVAL_CATEGORY_HIT_DELTA:--1000000}"
CATEGORY_GUARD_MIN_CASES="${CATEGORY_GUARD_MIN_CASES:-1}"
UNFREEZE_QWEN_LAYER_INDICES="${UNFREEZE_QWEN_LAYER_INDICES-23}"
SEED="${SEED:-20260519}"
MAX_TRAIN_CASES="${MAX_TRAIN_CASES:-0}"
MAX_EVAL_CASES="${MAX_EVAL_CASES:-256}"
SKIP_TRAIN_EVAL="${SKIP_TRAIN_EVAL:-0}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-20}"
MIN_EVAL_CORE_GAIN="${MIN_EVAL_CORE_GAIN:-0.01}"
MIN_LANGUAGE_TOP1_AGREEMENT="${MIN_LANGUAGE_TOP1_AGREEMENT:-0.75}"
MAX_REPEATED_TOKEN_RUN="${MAX_REPEATED_TOKEN_RUN:-8}"
MIN_UNIQUE_RATIO="${MIN_UNIQUE_RATIO:-0.20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
MAX_GENERATION_PROMPTS="${MAX_GENERATION_PROMPTS:-6}"

export HF_HOME

UNFREEZE_ARGS=()
if [[ -n "${UNFREEZE_QWEN_LAYER_INDICES}" ]]; then
  UNFREEZE_ARGS+=(--unfreeze-qwen-layer-indices "${UNFREEZE_QWEN_LAYER_INDICES}")
fi
TRAIN_EVAL_ARGS=()
if [[ "${SKIP_TRAIN_EVAL}" == "1" ]]; then
  TRAIN_EVAL_ARGS+=(--skip-train-eval)
fi
CATEGORY_ARGS=()
if [[ "${BALANCED_CATEGORY_SAMPLING}" == "1" ]]; then
  CATEGORY_ARGS+=(--balanced-category-sampling)
fi

PYTHONPATH="${PYTHONPATH:-src}" "${PYTHON_BIN}" scripts/391_train_qwen35_integrated_public_mcq_healing.py \
  --model-id "${MODEL_ID}" \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --eval-jsonl "${EVAL_JSONL}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --mandatory-core \
  --max-train-cases "${MAX_TRAIN_CASES}" \
  --max-eval-cases "${MAX_EVAL_CASES}" \
  "${TRAIN_EVAL_ARGS[@]}" \
  --core-impl qwen_layer_wrapped \
  --qwen-core-layer-indices "${QWEN_CORE_LAYER_INDICES}" \
  --core-adapter-dim "${CORE_ADAPTER_DIM}" \
  --core-delta-adapter-mode "${CORE_DELTA_ADAPTER_MODE}" \
  --residual-scale "${RESIDUAL_SCALE}" \
  --checkpoint-load-mode "${CHECKPOINT_LOAD_MODE}" \
  "${UNFREEZE_ARGS[@]}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --qwen-lr "${QWEN_LR}" \
  --base-kl-weight "${BASE_KL_WEIGHT}" \
  --language-kl-weight "${LANGUAGE_KL_WEIGHT}" \
  --ce-focus "${CE_FOCUS}" \
  --margin-weight "${MARGIN_WEIGHT}" \
  --margin-value "${MARGIN_VALUE}" \
  --margin-focus "${MARGIN_FOCUS}" \
  "${CATEGORY_ARGS[@]}" \
  --category-regression-penalty "${CATEGORY_REGRESSION_PENALTY}" \
  --min-eval-category-gain "${MIN_EVAL_CATEGORY_GAIN}" \
  --min-eval-category-hit-delta "${MIN_EVAL_CATEGORY_HIT_DELTA}" \
  --category-guard-min-cases "${CATEGORY_GUARD_MIN_CASES}" \
  --eval-every-steps "${EVAL_EVERY_STEPS}" \
  --min-eval-core-gain "${MIN_EVAL_CORE_GAIN}" \
  --min-language-top1-agreement "${MIN_LANGUAGE_TOP1_AGREEMENT}" \
  --max-repeated-token-run "${MAX_REPEATED_TOKEN_RUN}" \
  --min-unique-ratio "${MIN_UNIQUE_RATIO}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-generation-prompts "${MAX_GENERATION_PROMPTS}" \
  --seed "${SEED}" \
  --restore-best-checkpoint
