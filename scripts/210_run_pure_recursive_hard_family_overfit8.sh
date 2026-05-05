#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}" \
PYTHONPATH="${PYTHONPATH:-src}" \
CONFIG="${CONFIG:-configs/qwen35_2b_4090_pure_recursive_transition_state_code_only_s080.yaml}" \
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_transition_state_code_only_s080/last.pt}" \
TRAIN_CASES="${TRAIN_CASES:-data/filtered/pure_recursive_hard_family_overfit8_cases.jsonl}" \
DATA="${DATA:-data/filtered/pure_recursive_hard_family_overfit8_preferences.jsonl}" \
OUT_DIR="${OUT_DIR:-/mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_hard_family_overfit8_s120}" \
RUN_NAME="${RUN_NAME:-pure_recursive_hard_family_overfit8_s120}" \
HELDOUT_CASES="${HELDOUT_CASES:-data/eval/pure_recursive_hard_family_overfit8_cases.jsonl}" \
EVAL_OUT="${EVAL_OUT:-/mnt/nvme1n1p2/qtrm-eval/pure_recursive_hard_family_overfit8_s120_depth_gate_8.jsonl}" \
TRAIN_CASES_PER_FAMILY="${TRAIN_CASES_PER_FAMILY:-4}" \
TRAIN_START_INDEX="${TRAIN_START_INDEX:-100}" \
TRAIN_INCLUDE_FAMILIES="${TRAIN_INCLUDE_FAMILIES:-arithmetic_chain,list_transform}" \
HELDOUT_CASES_PER_FAMILY="${HELDOUT_CASES_PER_FAMILY:-4}" \
HELDOUT_START_INDEX="${HELDOUT_START_INDEX:-100}" \
HELDOUT_INCLUDE_FAMILIES="${HELDOUT_INCLUDE_FAMILIES:-arithmetic_chain,list_transform}" \
MAX_CASES="${MAX_CASES:-8}" \
STEPS="${STEPS:-120}" \
ALL_DEPTH_CE_WEIGHT="${ALL_DEPTH_CE_WEIGHT:-0.10}" \
CHOICE_MARGIN_WEIGHT="${CHOICE_MARGIN_WEIGHT:-0.25}" \
CHOICE_MARGIN="${CHOICE_MARGIN:-0.10}" \
TRANSITION_STATE_CONTRAST_WEIGHT="${TRANSITION_STATE_CONTRAST_WEIGHT:-0.25}" \
TRANSITION_STATE_CONTRAST_MARGIN="${TRANSITION_STATE_CONTRAST_MARGIN:-0.05}" \
TRANSITION_STATE_CODE_CE_WEIGHT="${TRANSITION_STATE_CODE_CE_WEIGHT:-0.75}" \
CAUSAL_PREFIX_SUPERVISION="${CAUSAL_PREFIX_SUPERVISION:-1}" \
CAUSAL_PREFIX_MAX_TARGET_TOKENS="${CAUSAL_PREFIX_MAX_TARGET_TOKENS:-6}" \
FAMILY_REPEAT="${FAMILY_REPEAT:-arithmetic_chain=4,list_transform=8}" \
bash scripts/197_run_pure_recursive_depth_supervised_train.sh
