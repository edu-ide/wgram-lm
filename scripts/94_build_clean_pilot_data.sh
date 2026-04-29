#!/usr/bin/env bash
set -euo pipefail

cd ~/qtrm-workspace/qtrm_multimodal_memoryos
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH=${PYTHONPATH:-$PWD/src}

OUTPUT=${OUTPUT:-data/filtered/qtrm_clean_pilot.jsonl}
MAX_ROWS=${MAX_ROWS:-6000}
MIN_WORDS=${MIN_WORDS:-32}
MAX_WORDS=${MAX_WORDS:-420}
INPUTS=${INPUTS:-"data/raw/text_train.jsonl data/raw/math_train.jsonl data/raw/mm_train.jsonl"}

read -r -a INPUT_FILES <<< "$INPUTS"

python -m qtrm_mm.data.clean_filter \
  --input "${INPUT_FILES[@]}" \
  --output "$OUTPUT" \
  --max-rows "$MAX_ROWS" \
  --min-words "$MIN_WORDS" \
  --max-words "$MAX_WORDS" \
  --drop-images
