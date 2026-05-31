#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/smoke_multimodal.yaml}
CKPT=${2:-runs/smoke_multimodal/last.pt}
PROMPT=${3:-"Explain multimodal QTRM-MemoryOS."}
export PYTHONPATH=$PWD/src
python -m wgram_lm.infer --config "$CONFIG" --checkpoint "$CKPT" --prompt "$PROMPT" --max-new-tokens 32
