#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GATE="${GATE:-${1:-qtrm_native_tiny_lm_first}}"
PROFILE="${PROFILE:-${2:-standard}}"
OUT_ROOT="${OUT_ROOT:-local_eval/research_gate_runner}"
WRITE_WIKI="${WRITE_WIKI:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"
OPERATION_LEDGER="${OPERATION_LEDGER:-local_eval/research_gate_runner/results.tsv}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

args=(
  scripts/300_research_gate_runner.py
  --gate "${GATE}"
  --profile "${PROFILE}"
  --out-root "${OUT_ROOT}"
)

if [[ "${WRITE_WIKI}" == "1" ]]; then
  args+=(--write-wiki)
fi

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  args+=(--skip-existing)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry-run)
fi

if [[ -n "${OPERATION_LEDGER}" ]]; then
  args+=(--operation-ledger "${OPERATION_LEDGER}")
fi

PYTHONPATH="${PYTHONPATH:-src}" "${PYTHON_BIN}" "${args[@]}"
