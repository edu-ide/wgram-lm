#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=${PYTHONPATH:-$PWD/src}
CONFIG=${1:-configs/smoke_multimodal.yaml}
python -m wgram_lm.training.train \
  --config "$CONFIG" \
  --data-jsonl data/raw/text_train.jsonl data/raw/math_train.jsonl
