#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}" \
PYTHONPATH="${PYTHONPATH:-src}" \
CONFIG="${CONFIG:-configs/qwen35_2b_4090_pure_recursive_transition_state_s080.yaml}" \
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_value_state_sequence_margin_s240/last.pt}" \
TRAIN_CASES="${TRAIN_CASES:-data/filtered/pure_recursive_full_state_sequence_train_cases.jsonl}" \
DATA="${DATA:-data/filtered/pure_recursive_full_state_sequence_preferences.jsonl}" \
OUT_DIR="${OUT_DIR:-/mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_full_state_sequence_s240}" \
RUN_NAME="${RUN_NAME:-pure_recursive_full_state_sequence_s240}" \
HELDOUT_CASES="${HELDOUT_CASES:-data/eval/pure_recursive_full_state_sequence_heldout200_cases.jsonl}" \
EVAL_OUT="${EVAL_OUT:-/mnt/nvme1n1p2/qtrm-eval/pure_recursive_full_state_sequence_s240_heldout200_mean_depth_gate_8.jsonl}" \
TRAIN_CASES_PER_FAMILY="${TRAIN_CASES_PER_FAMILY:-16}" \
TRAIN_START_INDEX="${TRAIN_START_INDEX:-100}" \
TRAIN_INCLUDE_FAMILIES="${TRAIN_INCLUDE_FAMILIES:-arithmetic_chain,list_transform}" \
HELDOUT_CASES_PER_FAMILY="${HELDOUT_CASES_PER_FAMILY:-4}" \
HELDOUT_START_INDEX="${HELDOUT_START_INDEX:-200}" \
HELDOUT_INCLUDE_FAMILIES="${HELDOUT_INCLUDE_FAMILIES:-arithmetic_chain,list_transform}" \
MAX_CASES="${MAX_CASES:-8}" \
STEPS="${STEPS:-240}" \
ALL_DEPTH_CE_WEIGHT="${ALL_DEPTH_CE_WEIGHT:-0.05}" \
STAGED_INTERNAL_FIRST_TOKEN_CE_WEIGHT="${STAGED_INTERNAL_FIRST_TOKEN_CE_WEIGHT:-0.00}" \
STAGED_INTERNAL_SEQUENCE_CE_WEIGHT="${STAGED_INTERNAL_SEQUENCE_CE_WEIGHT:-0.80}" \
STAGED_INTERNAL_SEQUENCE_MAX_TARGET_TOKENS="${STAGED_INTERNAL_SEQUENCE_MAX_TARGET_TOKENS:-6}" \
CHOICE_MARGIN_WEIGHT="${CHOICE_MARGIN_WEIGHT:-0.30}" \
CHOICE_MARGIN="${CHOICE_MARGIN:-0.15}" \
CHOICE_MARGIN_MODE="${CHOICE_MARGIN_MODE:-sequence}" \
TRANSITION_STATE_CONTRAST_WEIGHT="${TRANSITION_STATE_CONTRAST_WEIGHT:-0.30}" \
TRANSITION_STATE_CONTRAST_MARGIN="${TRANSITION_STATE_CONTRAST_MARGIN:-0.05}" \
TRANSITION_STATE_CE_WEIGHT="${TRANSITION_STATE_CE_WEIGHT:-0.10}" \
CAUSAL_PREFIX_SUPERVISION="${CAUSAL_PREFIX_SUPERVISION:-1}" \
CAUSAL_PREFIX_MAX_TARGET_TOKENS="${CAUSAL_PREFIX_MAX_TARGET_TOKENS:-6}" \
FAMILY_REPEAT="${FAMILY_REPEAT:-arithmetic_chain=4,list_transform=10}" \
CHOICE_SCORE_NORMALIZATION="${CHOICE_SCORE_NORMALIZATION:-mean}" \
bash scripts/197_run_pure_recursive_depth_supervised_train.sh
