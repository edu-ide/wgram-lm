#!/usr/bin/env bash
set -euo pipefail
QUERY=${1:-"QTRM MemoryOS"}
INDEX_DIR=${2:-memory/text}
export PYTHONPATH=$PWD/src
python -m qtrm_mm.memoryos.retrieve "$INDEX_DIR" "$QUERY" --top-k 5
