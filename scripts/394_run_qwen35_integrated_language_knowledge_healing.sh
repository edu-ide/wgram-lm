#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B-Base}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-local_eval/qwen35_integrated_healing_l23_langkl_s100_20260516/last_core.pt}"
TEXT_JSONL="${TEXT_JSONL:-local_eval/external_language_corpus/qtrm_native_external_bilingual_9000_20260515.jsonl}"
MCQ_JSONL="${MCQ_JSONL:-local_eval/m7_public_reasoning_suite/external_mcq_train_pool_2000_20260516.jsonl}"
MCQ_VALIDATION_JSONL="${MCQ_VALIDATION_JSONL:-local_eval/m7_public_reasoning_suite/external_mcq_validation_pool_20260516.jsonl}"
OUT_DIR="${OUT_DIR:-local_eval/qwen35_integrated_language_knowledge_healing_external9000_s120_20260516}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
CHECKPOINT_LOAD_MODE="${CHECKPOINT_LOAD_MODE:-strict_shapes}"
CORE_IMPL="${CORE_IMPL:-qwen_shared_layer_wrapped}"
CORE_INSERTION_MODE="${CORE_INSERTION_MODE:-mid_layer_suffix}"
CORE_INSERT_AFTER_LAYER="${CORE_INSERT_AFTER_LAYER:-11}"
QWEN_CORE_LAYER_INDICES="${QWEN_CORE_LAYER_INDICES:-3}"
CORE_ADAPTER_DIM="${CORE_ADAPTER_DIM:-128}"
CORE_DELTA_ADAPTER_MODE="${CORE_DELTA_ADAPTER_MODE:-add}"
CORE_RESIDUAL_GATE_MODE="${CORE_RESIDUAL_GATE_MODE:-constant}"
CORE_RESIDUAL_GATE_DIM="${CORE_RESIDUAL_GATE_DIM:-128}"
CORE_RESIDUAL_GATE_INIT="${CORE_RESIDUAL_GATE_INIT:--2.0}"
RESIDUAL_GATE_LR_MULTIPLIER="${RESIDUAL_GATE_LR_MULTIPLIER:-1.0}"
TRAIN_ONLY_CORE_DELTA_ADAPTER="${TRAIN_ONLY_CORE_DELTA_ADAPTER:-0}"
CLONE_QWEN_CORE_LAYERS="${CLONE_QWEN_CORE_LAYERS:-0}"
N_CORE_LAYERS="${N_CORE_LAYERS:-1}"
H_CYCLES="${H_CYCLES:-3}"
L_CYCLES="${L_CYCLES:-6}"
OUTER_STEPS="${OUTER_STEPS:-3}"
CORE_CONVERGENCE_HALT_ENABLED="${CORE_CONVERGENCE_HALT_ENABLED:-1}"
CORE_CONVERGENCE_HALT_THRESHOLD="${CORE_CONVERGENCE_HALT_THRESHOLD:-0.2}"
CORE_CONVERGENCE_HALT_MIN_OUTER="${CORE_CONVERGENCE_HALT_MIN_OUTER:-1}"
CORE_STEP_CONDITIONING_ENABLED="${CORE_STEP_CONDITIONING_ENABLED:-1}"
CORE_STEP_CONDITIONING_MAX_STEPS="${CORE_STEP_CONDITIONING_MAX_STEPS:-64}"
CORE_STEP_CONDITIONING_SCALE="${CORE_STEP_CONDITIONING_SCALE:-1.0}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.05}"
UNFREEZE_QWEN_LAYER_INDICES="${UNFREEZE_QWEN_LAYER_INDICES-23}"
STEPS="${STEPS:-120}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MCQ_BATCH_SIZE="${MCQ_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
MAX_TEXT_ROWS="${MAX_TEXT_ROWS:-6000}"
EVAL_TEXT_ROWS="${EVAL_TEXT_ROWS:-128}"
MAX_MCQ_ROWS="${MAX_MCQ_ROWS:-2000}"
EVAL_MCQ_ROWS="${EVAL_MCQ_ROWS:-256}"
LR="${LR:-1.0e-5}"
QWEN_LR="${QWEN_LR:-1.0e-7}"
BASE_KL_WEIGHT="${BASE_KL_WEIGHT:-0.10}"
LANGUAGE_ANCHOR_WEIGHT="${LANGUAGE_ANCHOR_WEIGHT:-0.0}"
LANGUAGE_ANCHOR_BATCH_SIZE="${LANGUAGE_ANCHOR_BATCH_SIZE:-4}"
MCQ_WEIGHT="${MCQ_WEIGHT:-0.10}"
MCQ_CE_FOCUS="${MCQ_CE_FOCUS:-all}"
MCQ_LOSS_SPACE="${MCQ_LOSS_SPACE:-full_vocab}"
MCQ_MARGIN_WEIGHT="${MCQ_MARGIN_WEIGHT:-0.0}"
MCQ_MARGIN_VALUE="${MCQ_MARGIN_VALUE:-0.5}"
MCQ_MARGIN_FOCUS="${MCQ_MARGIN_FOCUS:-base_wrong}"
BASE_WRONG_MAX_TOP_MARGIN="${BASE_WRONG_MAX_TOP_MARGIN:--1.0}"
MCQ_NON_SELECTED_OPTION_KL_WEIGHT="${MCQ_NON_SELECTED_OPTION_KL_WEIGHT:-0.0}"
RESIDUAL_GATE_SELECTED_OPEN_WEIGHT="${RESIDUAL_GATE_SELECTED_OPEN_WEIGHT:-0.0}"
RESIDUAL_GATE_NON_SELECTED_CLOSED_WEIGHT="${RESIDUAL_GATE_NON_SELECTED_CLOSED_WEIGHT:-0.0}"
BASE_WRONG_MCQ_RETRIES="${BASE_WRONG_MCQ_RETRIES:-1}"
BASE_CORRECT_OPTION_KL_WEIGHT="${BASE_CORRECT_OPTION_KL_WEIGHT:-0.05}"
BASE_CORRECT_OPTION_KL_FOCUS="${BASE_CORRECT_OPTION_KL_FOCUS:-base_correct}"
BASE_CORRECT_KL_EXTRA_BATCH_SIZE="${BASE_CORRECT_KL_EXTRA_BATCH_SIZE:-0}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-40}"
RESTORE_BEST_CHECKPOINT="${RESTORE_BEST_CHECKPOINT:-1}"
SKIP_SAVE_CHECKPOINT="${SKIP_SAVE_CHECKPOINT:-0}"
CATEGORY_REGRESSION_PENALTY="${CATEGORY_REGRESSION_PENALTY:-0.25}"
TEXT_CE_REGRESSION_PENALTY="${TEXT_CE_REGRESSION_PENALTY:-1.0}"
MAX_CORE_CE_REGRESSION="${MAX_CORE_CE_REGRESSION:-0.01}"
MIN_EVAL_MCQ_GAIN="${MIN_EVAL_MCQ_GAIN:-0.0}"
MIN_BASE_WRONG_CORE_CORRECT="${MIN_BASE_WRONG_CORE_CORRECT:-0}"
MAX_BASE_CORRECT_CORE_WRONG="${MAX_BASE_CORRECT_CORE_WRONG:-1000000}"
MIN_EVAL_MCQ_CATEGORY_GAIN="${MIN_EVAL_MCQ_CATEGORY_GAIN:--1.0}"
MIN_EVAL_MCQ_CATEGORY_HIT_DELTA="${MIN_EVAL_MCQ_CATEGORY_HIT_DELTA:--1000000}"
CATEGORY_GUARD_MIN_CASES="${CATEGORY_GUARD_MIN_CASES:-8}"
MIN_LANGUAGE_TOP1_AGREEMENT="${MIN_LANGUAGE_TOP1_AGREEMENT:-0.75}"
MAX_REPEATED_TOKEN_RUN="${MAX_REPEATED_TOKEN_RUN:-8}"
MIN_UNIQUE_RATIO="${MIN_UNIQUE_RATIO:-0.20}"
MAX_GENERATION_PROMPTS="${MAX_GENERATION_PROMPTS:-6}"
SEED="${SEED:-20260521}"

export HF_HOME

MCQ_ARGS=()
if [[ -n "${MCQ_JSONL}" ]]; then
  MCQ_ARGS+=(--mcq-jsonl "${MCQ_JSONL}")
fi
if [[ -n "${MCQ_VALIDATION_JSONL}" ]]; then
  MCQ_ARGS+=(--mcq-validation-jsonl "${MCQ_VALIDATION_JSONL}")
fi
UNFREEZE_ARGS=()
UNFREEZE_ARGS+=(--unfreeze-qwen-layer-indices "${UNFREEZE_QWEN_LAYER_INDICES}")
RESTORE_ARGS=()
if [[ "${RESTORE_BEST_CHECKPOINT}" == "1" ]]; then
  RESTORE_ARGS+=(--restore-best-checkpoint)
fi
CLONE_CORE_ARGS=()
if [[ "${CLONE_QWEN_CORE_LAYERS}" == "1" ]]; then
  CLONE_CORE_ARGS+=(--clone-qwen-core-layers)
fi
SAVE_ARGS=()
if [[ "${SKIP_SAVE_CHECKPOINT}" == "1" ]]; then
  SAVE_ARGS+=(--skip-save-checkpoint)
fi
HALT_ARGS=()
if [[ "${CORE_CONVERGENCE_HALT_ENABLED}" == "1" ]]; then
  HALT_ARGS+=(--core-convergence-halt-enabled)
else
  HALT_ARGS+=(--no-core-convergence-halt)
fi
STEP_CONDITIONING_ARGS=()
if [[ "${CORE_STEP_CONDITIONING_ENABLED}" == "1" ]]; then
  STEP_CONDITIONING_ARGS+=(--core-step-conditioning-enabled)
else
  STEP_CONDITIONING_ARGS+=(--no-core-step-conditioning)
fi
TRAIN_ONLY_ADAPTER_ARGS=()
if [[ "${TRAIN_ONLY_CORE_DELTA_ADAPTER}" == "1" ]]; then
  TRAIN_ONLY_ADAPTER_ARGS+=(--train-only-core-delta-adapter)
fi

PYTHONPATH="${PYTHONPATH:-src}" "${PYTHON_BIN}" scripts/394_train_qwen35_integrated_language_knowledge_healing.py \
  --model-id "${MODEL_ID}" \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --checkpoint-load-mode "${CHECKPOINT_LOAD_MODE}" \
  --text-jsonl "${TEXT_JSONL}" \
  "${MCQ_ARGS[@]}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --mandatory-core \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --max-text-rows "${MAX_TEXT_ROWS}" \
  --eval-text-rows "${EVAL_TEXT_ROWS}" \
  --max-mcq-rows "${MAX_MCQ_ROWS}" \
  --eval-mcq-rows "${EVAL_MCQ_ROWS}" \
  --core-impl "${CORE_IMPL}" \
  --core-insertion-mode "${CORE_INSERTION_MODE}" \
  --core-insert-after-layer "${CORE_INSERT_AFTER_LAYER}" \
  --qwen-core-layer-indices "${QWEN_CORE_LAYER_INDICES}" \
  --core-adapter-dim "${CORE_ADAPTER_DIM}" \
  --core-delta-adapter-mode "${CORE_DELTA_ADAPTER_MODE}" \
  --core-residual-gate-mode "${CORE_RESIDUAL_GATE_MODE}" \
  --core-residual-gate-dim "${CORE_RESIDUAL_GATE_DIM}" \
  --core-residual-gate-init "${CORE_RESIDUAL_GATE_INIT}" \
  --residual-gate-lr-multiplier "${RESIDUAL_GATE_LR_MULTIPLIER}" \
  "${TRAIN_ONLY_ADAPTER_ARGS[@]}" \
  "${CLONE_CORE_ARGS[@]}" \
  --n-core-layers "${N_CORE_LAYERS}" \
  --h-cycles "${H_CYCLES}" \
  --l-cycles "${L_CYCLES}" \
  --outer-steps "${OUTER_STEPS}" \
  "${HALT_ARGS[@]}" \
  --core-convergence-halt-threshold "${CORE_CONVERGENCE_HALT_THRESHOLD}" \
  --core-convergence-halt-min-outer "${CORE_CONVERGENCE_HALT_MIN_OUTER}" \
  "${STEP_CONDITIONING_ARGS[@]}" \
  --core-step-conditioning-max-steps "${CORE_STEP_CONDITIONING_MAX_STEPS}" \
  --core-step-conditioning-scale "${CORE_STEP_CONDITIONING_SCALE}" \
  --residual-scale "${RESIDUAL_SCALE}" \
  "${UNFREEZE_ARGS[@]}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --mcq-batch-size "${MCQ_BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --lr "${LR}" \
  --qwen-lr "${QWEN_LR}" \
  --base-kl-weight "${BASE_KL_WEIGHT}" \
  --language-anchor-weight "${LANGUAGE_ANCHOR_WEIGHT}" \
  --language-anchor-batch-size "${LANGUAGE_ANCHOR_BATCH_SIZE}" \
  --mcq-weight "${MCQ_WEIGHT}" \
  --mcq-ce-focus "${MCQ_CE_FOCUS}" \
  --mcq-loss-space "${MCQ_LOSS_SPACE}" \
  --mcq-margin-weight "${MCQ_MARGIN_WEIGHT}" \
  --mcq-margin-value "${MCQ_MARGIN_VALUE}" \
  --mcq-margin-focus "${MCQ_MARGIN_FOCUS}" \
  --base-wrong-max-top-margin "${BASE_WRONG_MAX_TOP_MARGIN}" \
  --mcq-non-selected-option-kl-weight "${MCQ_NON_SELECTED_OPTION_KL_WEIGHT}" \
  --residual-gate-selected-open-weight "${RESIDUAL_GATE_SELECTED_OPEN_WEIGHT}" \
  --residual-gate-non-selected-closed-weight "${RESIDUAL_GATE_NON_SELECTED_CLOSED_WEIGHT}" \
  --base-wrong-mcq-retries "${BASE_WRONG_MCQ_RETRIES}" \
  --base-correct-option-kl-weight "${BASE_CORRECT_OPTION_KL_WEIGHT}" \
  --base-correct-option-kl-focus "${BASE_CORRECT_OPTION_KL_FOCUS}" \
  --base-correct-kl-extra-batch-size "${BASE_CORRECT_KL_EXTRA_BATCH_SIZE}" \
  --balanced-mcq-category-sampling \
  --eval-every-steps "${EVAL_EVERY_STEPS}" \
  --category-regression-penalty "${CATEGORY_REGRESSION_PENALTY}" \
  --text-ce-regression-penalty "${TEXT_CE_REGRESSION_PENALTY}" \
  --max-core-ce-regression "${MAX_CORE_CE_REGRESSION}" \
  --min-eval-mcq-gain "${MIN_EVAL_MCQ_GAIN}" \
  --min-base-wrong-core-correct "${MIN_BASE_WRONG_CORE_CORRECT}" \
  --max-base-correct-core-wrong "${MAX_BASE_CORRECT_CORE_WRONG}" \
  --min-eval-mcq-category-gain "${MIN_EVAL_MCQ_CATEGORY_GAIN}" \
  --min-eval-mcq-category-hit-delta "${MIN_EVAL_MCQ_CATEGORY_HIT_DELTA}" \
  --category-guard-min-cases "${CATEGORY_GUARD_MIN_CASES}" \
  --min-language-top1-agreement "${MIN_LANGUAGE_TOP1_AGREEMENT}" \
  --max-repeated-token-run "${MAX_REPEATED_TOKEN_RUN}" \
  --min-unique-ratio "${MIN_UNIQUE_RATIO}" \
  --max-generation-prompts "${MAX_GENERATION_PROMPTS}" \
  --seed "${SEED}" \
  "${RESTORE_ARGS[@]}" \
  "${SAVE_ARGS[@]}"
