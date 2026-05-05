#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_workspace_evidence_bounded_preference_s050.yaml}"
RUN_DIR="${RUN_DIR:-runs/qwen35_2b_4090_workspace_evidence_bounded_preference_s050}"
MAX_CASES="${MAX_CASES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

if [[ -n "${CHECKPOINT:-}" ]]; then
  CHECKPOINT_SPECS=("custom:${CHECKPOINT}")
else
  CHECKPOINT_SPECS=(
    "step100:${RUN_DIR}/step_000100.pt"
    "step200:${RUN_DIR}/step_000200.pt"
    "last:${RUN_DIR}/last.pt"
  )
fi

for spec in "${CHECKPOINT_SPECS[@]}"; do
  name="${spec%%:*}"
  checkpoint="${spec#*:}"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint for ${name}: ${checkpoint}" >&2
    exit 1
  fi

  echo "============================================================"
  echo "Bounded preference checkpoint quick compare: ${name}"
  echo "checkpoint=${checkpoint}"
  echo "max_cases=${MAX_CASES}"
  echo "============================================================"

  MAX_CASES="$MAX_CASES" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  CONFIG="$CONFIG" \
  CHECKPOINT="$checkpoint" \
  OUT="runs/eval/memory_reasoning_quick_workspace_evidence_bounded_preference_${name}_32tok_s050.jsonl" \
  AUDIT_OUT="runs/eval/memory_reasoning_quick_workspace_evidence_bounded_preference_${name}_32tok_s050_audit.jsonl" \
  PROOF_MD="docs/wiki/decisions/workspace-evidence-bounded-preference-${name}-quick-ablation.md" \
  PROOF_JSON="docs/wiki/decisions/workspace-evidence-bounded-preference-${name}-quick-ablation-summary.json" \
    bash scripts/117_run_workspace_evidence_path_probe.sh
done
