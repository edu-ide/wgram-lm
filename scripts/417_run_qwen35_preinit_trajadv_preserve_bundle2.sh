#!/usr/bin/env bash
set -euo pipefail

# Bundle2 repair gate. The accepted base-error trajectory objective did not
# repeat on eval offsets 20000,20001,20002 because chain5 regressed below the
# core-off Qwen path. This adds a do-no-harm preservation margin on examples
# where the base digit-choice margin is already positive.

export HF_HOME="${HF_HOME:-/mnt/data4tb/hf-cache-qtrm}"
export PYTHONPATH="${PYTHONPATH:-src}"

OUT_DIR="${OUT_DIR:-local_eval/qwen35_preinit_recurrent_trajadv_preserve_bundle2_s${STEPS:-80}_$(date +%Y%m%d_%H%M%S)}"

PYTHON="${PYTHON:-.venv/bin/python}" \
OUT_DIR="${OUT_DIR}" \
EVAL_SEED_OFFSETS="${EVAL_SEED_OFFSETS:-20000,20001,20002}" \
TRAJECTORY_LOSS_BASE_ERROR_ONLY="${TRAJECTORY_LOSS_BASE_ERROR_ONLY:-1}" \
TRAJECTORY_ADVANTAGE_WEIGHT="${TRAJECTORY_ADVANTAGE_WEIGHT:-0.35}" \
TRAJECTORY_ADVANTAGE_MARGIN="${TRAJECTORY_ADVANTAGE_MARGIN:-0.02}" \
TRAJECTORY_MONOTONIC_WEIGHT="${TRAJECTORY_MONOTONIC_WEIGHT:-0.02}" \
CHECKSUM_TRAJECTORY_WEIGHT="${CHECKSUM_TRAJECTORY_WEIGHT:-0.5}" \
CORE_PRESERVATION_WEIGHT="${CORE_PRESERVATION_WEIGHT:-0.20}" \
CORE_PRESERVATION_MARGIN="${CORE_PRESERVATION_MARGIN:-0.0}" \
CORE_PRESERVATION_POSITIVE_MARGIN_ONLY="${CORE_PRESERVATION_POSITIVE_MARGIN_ONLY:-1}" \
LANGUAGE_HEALING_WEIGHT="${LANGUAGE_HEALING_WEIGHT:-0.08}" \
bash scripts/416_run_qwen35_preinit_recurrent_trajectory_advantage_multiseed.sh
