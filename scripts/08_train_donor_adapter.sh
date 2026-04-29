#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
CONFIG=${1:-configs/qwen35_2b_4090.yaml}
shift 2>/dev/null || true

DATA_JSONL=${DATA_JSONL:-"data/raw/text_train.jsonl data/raw/math_train.jsonl data/raw/mm_train.jsonl"}
DATA_ARGS=()
MULTIMODAL=${MULTIMODAL:-1}
MULTIMODAL_ARGS=()
if [[ "$MULTIMODAL" == "1" ]]; then
  MULTIMODAL_ARGS=(--multimodal)
fi

if [[ "${ALLOW_SYNTHETIC:-0}" == "1" ]]; then
  echo "WARNING: ALLOW_SYNTHETIC=1: training on synthetic smoke data, not real language data."
else
  read -r -a DATA_FILES <<< "$DATA_JSONL"
  for f in "${DATA_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
      echo "Missing training data file: $f" >&2
      echo "Run scripts/01_download_datasets.sh first, or set ALLOW_SYNTHETIC=1 for smoke training." >&2
      exit 1
    fi
  done
  DATA_ARGS=(--data-jsonl "${DATA_FILES[@]}")
fi

echo "=== Training QTRM adapter with Qwen3.5-2B donor (4bit) ==="
echo "Config: $CONFIG"
if [[ ${#DATA_ARGS[@]} -gt 0 ]]; then
  echo "Data: ${DATA_FILES[*]}"
fi
python -m qtrm_mm.training.train \
  --config "$CONFIG" \
  "${MULTIMODAL_ARGS[@]}" \
  --use-donor \
  "${DATA_ARGS[@]}" \
  "$@"
