#!/usr/bin/env bash
set -euo pipefail

cd ~/qtrm-workspace/wgram-lm
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG=${1:-configs/qwen35_2b_4090.yaml}
if [[ $# -gt 0 ]]; then
  shift
fi

python -m wgram_lm.training.tiny_overfit --config "$CONFIG" "$@"
