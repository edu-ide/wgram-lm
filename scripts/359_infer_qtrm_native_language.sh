#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_s3600_20260515/last.pt}"
DEVICE="${DEVICE:-cuda}"
THINK_STEPS="${THINK_STEPS:-4}"
MAX_NEW_CHARS="${MAX_NEW_CHARS:-220}"
PROMPT="${1:-Why should evidence be checked before trusting a claim?}"

PYTHONPATH="${PYTHONPATH:-src}" \
  "${PYTHON:-.venv/bin/python}" scripts/359_infer_qtrm_native_language.py \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  --think-steps "$THINK_STEPS" \
  --max-new-chars "$MAX_NEW_CHARS" \
  "$PROMPT"
