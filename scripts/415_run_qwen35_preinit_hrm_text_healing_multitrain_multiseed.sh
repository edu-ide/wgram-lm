#!/usr/bin/env bash
set -euo pipefail

# Multi-train-seed + multi-eval-seed healing gate. This targets the observed
# bottleneck where a single-train-seed checkpoint nearly passes aggregate
# multi-seed evaluation but remains below the 0.02 reasoning-gain threshold.

export HF_HOME="${HF_HOME:-/mnt/data4tb/hf-cache-qtrm}"
export PYTHONPATH="${PYTHONPATH:-src}"

INIT_CHECKPOINT="${INIT_CHECKPOINT:-local_eval/qwen35_preinit_trajcarry_mean_512select_lang0875_s100_20260519/last_core.pt}"
OUT_DIR="${OUT_DIR:-local_eval/qwen35_preinit_trajcarry_mean_hrmtext_multitrain_multiseed_s${STEPS:-80}_$(date +%Y%m%d_%H%M%S)}"

PYTHON="${PYTHON:-.venv/bin/python}" \
INIT_CHECKPOINT="${INIT_CHECKPOINT}" \
OUT_DIR="${OUT_DIR}" \
SEED="${SEED:-20260519}" \
TRAIN_SEED_OFFSETS="${TRAIN_SEED_OFFSETS:-0,1,2}" \
EVAL_SEED_OFFSETS="${EVAL_SEED_OFFSETS:-10000,10001,10002}" \
CORE_TRAJECTORY_CARRY_MODE="${CORE_TRAJECTORY_CARRY_MODE:-mean}" \
UNFREEZE_QWEN_LAYER_INDICES="${UNFREEZE_QWEN_LAYER_INDICES:-}" \
QWEN_LR="${QWEN_LR:-0.0}" \
LR="${LR:-8.0e-6}" \
KL_WEIGHT="${KL_WEIGHT:-0.10}" \
LANGUAGE_PROBE_SET="${LANGUAGE_PROBE_SET:-extended}" \
LANGUAGE_KL_WEIGHT="${LANGUAGE_KL_WEIGHT:-0.12}" \
LANGUAGE_KL_BATCH_SIZE="${LANGUAGE_KL_BATCH_SIZE:-4}" \
LANGUAGE_HEALING_WEIGHT="${LANGUAGE_HEALING_WEIGHT:-0.14}" \
LANGUAGE_HEALING_KL_WEIGHT="${LANGUAGE_HEALING_KL_WEIGHT:-0.05}" \
LANGUAGE_HEALING_BATCH_SIZE="${LANGUAGE_HEALING_BATCH_SIZE:-2}" \
SELECTION_LANGUAGE_WEIGHT="${SELECTION_LANGUAGE_WEIGHT:-0.5}" \
SELECTION_MIN_LANGUAGE_TOP1="${SELECTION_MIN_LANGUAGE_TOP1:-0.96875}" \
CORE_ADVANTAGE_WEIGHT="${CORE_ADVANTAGE_WEIGHT:-0.08}" \
CORE_ADVANTAGE_MARGIN="${CORE_ADVANTAGE_MARGIN:-0.02}" \
CORE_ADVANTAGE_MODE="${CORE_ADVANTAGE_MODE:-label_choice_margin}" \
FAMILY_LOSS_WEIGHTS="${FAMILY_LOSS_WEIGHTS:-chain5=1.2,checksum4=2.0,select_pair=1.2}" \
CHECKSUM_TRAJECTORY_WEIGHT="${CHECKSUM_TRAJECTORY_WEIGHT:-1.1}" \
STEPS="${STEPS:-80}" \
TRAIN_CASES="${TRAIN_CASES:-2048}" \
EVAL_CASES="${EVAL_CASES:-192}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-20}" \
MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}" \
DTYPE="${DTYPE:-bfloat16}" \
LOG_EVERY="${LOG_EVERY:-20}" \
bash scripts/410_run_qwen35_preinit_strict_trm_core_gate.sh
