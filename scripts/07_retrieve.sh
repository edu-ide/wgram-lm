#!/usr/bin/env bash
set -euo pipefail
QUERY=${1:-"QTRM MemoryOS"}
INDEX_DIR=${2:-memory/text}
export PYTHONPATH=$PWD/src
python -m wgram_lm.memoryos.retrieve "$INDEX_DIR" "$QUERY" --top-k 5
