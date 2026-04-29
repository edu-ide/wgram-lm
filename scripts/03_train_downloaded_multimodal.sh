#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=${PYTHONPATH:-$PWD/src}
CONFIG=${1:-configs/smoke_multimodal.yaml}
python -m qtrm_mm.training.train \
  --config "$CONFIG" \
  --multimodal \
  --data-jsonl data/raw/text_train.jsonl data/raw/math_train.jsonl data/raw/mm_train.jsonl
