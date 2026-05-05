#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/qwen35_2b_4090_hf_first_wave_warmup_v2_s400.yaml}
if [[ $# -gt 0 ]]; then
  shift
fi

export DATA_JSONL=${DATA_JSONL:-data/filtered/hf_distill_smoke/qtrm_hf_first_wave_mix_s400.jsonl}
export MULTIMODAL=${MULTIMODAL:-0}

bash scripts/08_train_donor_adapter.sh "$CONFIG" "$@"
