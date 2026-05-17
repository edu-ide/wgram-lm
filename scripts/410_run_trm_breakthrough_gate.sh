#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

M6_REPORT="${M6_REPORT:-local_eval/m6_scoped_raw_reasoning_manifest_dgx512_20260517/report.json}"
M7B_REPORT="${M7B_REPORT:-local_eval/dgx_m7b_core_depth_gate_m7a_s300_256_20260517/m7b_gate_report.json}"
OUT_JSON="${OUT_JSON:-local_eval/trm_like_breakthrough_gate/report.json}"

PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" scripts/409_score_trm_breakthrough_gate.py \
  --m6-report "$M6_REPORT" \
  --m7b-report "$M7B_REPORT" \
  --out-json "$OUT_JSON"

